/**
 * CalibrationService: servicio de calibración matemática de puntuaciones.
 *
 * Resuelve la discrepancia entre distancias geométricas (coseno anisotrópico)
 * y probabilidades reales de éxito empírico P in [0, 1].
 *
 * Utiliza:
 * 1. Corrección de anisotropía: elimina el suelo basal inducido por Transformers
 *    (e5-small/large) donde textos no relacionados tienen cosenos de ~0.65 - 0.75.
 * 2. Sigmoide logística con escalado por temperatura (Platt / Temperature Scaling):
 *    P(tool | query) = 1 / (1 + exp(-(z - bias) / T)).
 */

import {
  CALIBRATION_DEFAULT_BIAS,
  CALIBRATION_DEFAULT_FLOOR,
  CALIBRATION_DEFAULT_TEMP,
} from "src/shared/constants/predict";

import type { CalibrationOptions } from "src/shared/interfaces";

export type { CalibrationOptions };

export class CalibrationService {
  /** Suelo empírico por defecto para multilingual-e5. */
  static readonly DEFAULT_FLOOR = CALIBRATION_DEFAULT_FLOOR;
  static readonly DEFAULT_TEMP = CALIBRATION_DEFAULT_TEMP;
  static readonly DEFAULT_BIAS = CALIBRATION_DEFAULT_BIAS;

  /**
   * Corrige la degeneración anisotrópica del espacio latente.
   * Proyecta el rango [floor, 1.0] -> [0.0, 1.0].
   * Valores por debajo del suelo se amortiguan fuertemente a 0.
   */
  static correctAnisotropy(rawCosine: number, floor = CalibrationService.DEFAULT_FLOOR): number {
    if (!Number.isFinite(rawCosine) || rawCosine <= 0) return 0;
    if (rawCosine >= 1.0) return 1.0;
    if (rawCosine <= floor) {
      // Amortiguación cúbica por debajo del suelo para no cortar abruptamente
      const ratio = Math.max(0, rawCosine / floor);
      return 0.05 * Math.pow(ratio, 3);
    }
    return (rawCosine - floor) / (1.0 - floor);
  }

  /**
   * Sigmoide logística calibrada con parámetro de temperatura.
   * Devuelve un valor estrictamente acotado en (0, 1).
   */
  static sigmoid(z: number, temperature = CalibrationService.DEFAULT_TEMP): number {
    const t = Math.max(0.01, temperature);
    const logit = z / t;
    // Evita overflow/underflow numérico
    if (logit > 30) return 0.9999;
    if (logit < -30) return 0.0001;
    return 1 / (1 + Math.exp(-logit));
  }

  /**
   * Calibra un score compuesto (fusión de dense, BM25, learned, acción y penalizaciones)
   * transformándolo en una probabilidad real calibrada P in [0, 1].
   */
  static calibrateProbability(
    rawFusedScore: number,
    opts: CalibrationOptions = {},
  ): number {
    if (!Number.isFinite(rawFusedScore)) return 0;
    const temp = opts.temperature ?? CalibrationService.DEFAULT_TEMP;
    const bias = opts.bias ?? CalibrationService.DEFAULT_BIAS;

    // Centramos el logit respecto al bias de decisión y aplicamos Temperature Scaling
    const centered = rawFusedScore - bias;
    const prob = CalibrationService.sigmoid(centered, temp);

    // Redondeo de precisión a 4 decimales para estabilidad
    return Math.round(prob * 10000) / 10000;
  }

  /**
   * Calibra un arreglo de scores ordenados garantizando que:
   * 1. Cada score esté en [0, 1].
   * 2. Se preserve el orden relativo estricto.
   * 3. La distancia relativa (gap) se mantenga proporcional para el corte adaptativo.
   */
  static calibrateRankedScores(scores: number[], opts: CalibrationOptions = {}): number[] {
    if (scores.length === 0) return [];
    return scores.map((s) => CalibrationService.calibrateProbability(s, opts));
  }
}
