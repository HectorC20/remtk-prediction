/**
 * Abstención (juicio §2): dos señales matemáticas para NO devolver herramientas
 * cuando la entrada es ajena al catálogo (small-talk, divagación).
 *
 * A. Prototipo __NOOP__: arquetipos de interacción sin herramienta embebidos
 *    con e5; si su coseno máximo supera al mejor score de herramienta
 *    (+ margen), el enrutador emite ∅.
 *
 * B. Energía libre E(u) = −T·ln Σ_i exp(sim_i/T): distribuciones planas y
 *    bajas (nada destaca) superan τ → out-of-distribution. Una señal clara
 *    (pico alto sobre el resto) da energía muy negativa.
 *    DESACTIVADA por defecto (τ ≥ 0 = no-op): en la banda plana de e5-small,
 *    E está dominada por −max — consultas legítimas con top 0.83 dan
 *    E ≈ −0.86 y small-talk plano-alto E ≈ −0.96, sin τ que las separe.
 *    Medido en bench (docs/juicio.md). Se mantiene la fórmula para modelos
 *    con mayor rango dinámico de scores.
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import type { AppConfig } from "src/config";
import { NOOP_ARCHETYPES } from "src/shared/constants/juicio";

export class AbstencionService {
  private noopEmbs?: Float32Array[];

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly config: AppConfig,
  ) {}

  /** Coseno máximo del turno contra el pool __NOOP__ (0 si el modelo degrada). */
  async noopScore(uText: Float32Array | undefined): Promise<number> {
    if (!uText) return 0;
    const embs = await this.ensureNoopEmbs();
    let max = 0;
    for (const emb of embs) {
      const s = EmbeddingEngineService.cosine(uText, emb);
      if (s > max) max = s;
    }
    return max;
  }

  /** Energía libre de Helmholtz sobre los scores del ranking. */
  energy(scores: number[]): number {
    if (scores.length === 0) return 0;
    const T = this.config.juicioEnergyTemp;
    let sum = 0;
    for (const s of scores) sum += Math.exp(s / T);
    return -T * Math.log(sum);
  }

  /**
   * Veredicto de abstención: cualquiera de las dos señales basta.
   * Devuelve el motivo para la traza, o null si hay señal de herramienta.
   */
  async shouldAbstain(
    uText: Float32Array | undefined,
    rankedScores: number[],
  ): Promise<{ abstain: boolean; reason?: "noop" | "energy"; noop: number; energy: number }> {
    const noop = await this.noopScore(uText);
    const energy = this.energy(rankedScores);
    const top = rankedScores[0] ?? 0;
    if (noop > top + this.config.juicioNoopMargin) {
      return { abstain: true, reason: "noop", noop, energy };
    }
    const tau = this.config.juicioEnergyTau;
    if (tau < 0 && rankedScores.length > 0 && energy > tau) {
      return { abstain: true, reason: "energy", noop, energy };
    }
    return { abstain: false, noop, energy };
  }

  private async ensureNoopEmbs(): Promise<Float32Array[]> {
    if (this.noopEmbs) return this.noopEmbs;
    const embs = await Promise.all(
      NOOP_ARCHETYPES.map((a) => this.engine.embedQuery(a, "small").then((r) => r.embedding)),
    );
    this.noopEmbs = embs;
    return embs;
  }
}
