/** Tamaños de los modelos ONNX del pipeline. */

/** Tamaño pequeño (e5-small, default del pipeline). */
export const ONNXMODELSIZESMALL = "small";

/** Tamaño grande (e5-large, 1024 dims; carga bajo demanda, no en warmup). */
export const ONNXMODELSIZELARGE = "large";

/** Tamaño de modelo soportado (derivado de las constantes para no repetir literales). */
export type ModelSize = typeof ONNXMODELSIZESMALL | typeof ONNXMODELSIZELARGE;
