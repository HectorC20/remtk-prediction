/**
 * Constantes y valores por defecto para CalibrationService.
 */

/**
 * Suelo empírico por defecto para multilingual-e5.
 * Cosenos < 0.68 entre frases cortas representan dispersión anisotrópica.
 */
export const CALIBRATION_DEFAULT_FLOOR = 0.68;

/** Parámetro de temperatura T para Platt/Temperature scaling. */
export const CALIBRATION_DEFAULT_TEMP = 0.35;

/** Bias del logit para centrar la sigmoide en el punto de decisión óptimo. */
export const CALIBRATION_DEFAULT_BIAS = 0.50;
