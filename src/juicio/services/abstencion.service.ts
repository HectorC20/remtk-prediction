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

export class AbstencionService {
  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly config: AppConfig,
  ) {}

  /** Coseno máximo del turno contra el pool __NOOP__ (0 al no haber arquetipos hardcodeados). */
  async noopScore(_uText: Float32Array | undefined): Promise<number> {
    return 0;
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
   * Veredicto de abstención: señal matemática de energía.
   * Devuelve el motivo para la traza, o null si hay señal de herramienta.
   */
  async shouldAbstain(
    uText: Float32Array | undefined,
    rankedScores: number[],
  ): Promise<{ abstain: boolean; reason?: "noop" | "energy"; noop: number; energy: number }> {
    const noop = await this.noopScore(uText);
    const energy = this.energy(rankedScores);
    const top = rankedScores[0] ?? 0;
    if (noop > 0 && noop > top + this.config.juicioNoopMargin) {
      return { abstain: true, reason: "noop", noop, energy };
    }
    const tau = this.config.juicioEnergyTau;
    if (tau < 0 && rankedScores.length > 0 && energy > tau) {
      return { abstain: true, reason: "energy", noop, energy };
    }
    return { abstain: false, noop, energy };
  }
}
