export interface CalibrationOptions {
  /** Suelo empírico de anisotropía (coseno por debajo del cual es ruido basal). */
  anisotropyFloor?: number;
  /** Parámetro de temperatura T para suavizar o acentuar la distribución. */
  temperature?: number;
  /** Bias del logit para centrar la sigmoide en el punto de decisión óptimo. */
  bias?: number;
}
