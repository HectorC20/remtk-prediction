/**
 * Puerta de Coherencia Semántica (juicio §3): decaimiento adaptativo del
 * historial por correlación coseno entre el mensaje actual u_t y el estado
 * acumulado H_{t-1}.
 *
 *   λ = σ(cos(u_t, H_{t-1})·β − γ)        q = (1−λ)·u_t + λ·H_{t-1}
 *
 * Temas ortogonales (cambio de asunto) colapsan λ→0 y el historial deja de
 * arrastrar la consulta; temas coherentes mantienen la inercia contextual.
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import type { AppConfig } from "src/config";

export class PuertaContextoService {
  constructor(private readonly config: AppConfig) {}

  /** λ ∈ (0,1) de la sigmoide sobre la coherencia coseno. */
  lambda(uText: Float32Array | undefined, prevState: Float32Array | undefined): number {
    if (!uText || !prevState || uText.length === 0 || prevState.length === 0) return 0;
    const cos = EmbeddingEngineService.cosine(uText, prevState);
    const x = cos * this.config.juicioGateBeta - this.config.juicioGateGamma;
    return 1 / (1 + Math.exp(-x));
  }

  /** true si el turno debe aislarse del historial (λ bajo el mínimo de texto). */
  gatesHistory(lambda: number, hasPrevState: boolean): boolean {
    return hasPrevState && lambda < this.config.juicioGateTextMin;
  }
}
