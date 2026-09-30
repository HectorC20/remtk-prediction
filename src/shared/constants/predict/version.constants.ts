/** Tamaños de los modelos ONNX del pipeline. */

/** Tamaño pequeño (e5-small). */
export const ONNXMODELSIZESMALL = "small";

/** Tamaño grande (e5-large, 1024 dims; carga bajo demanda, no en warmup). */
export const ONNXMODELSIZELARGE = "large";

/** gte-multilingual-base (768 dims, pooling CLS, sin prefijos query:/passage:) — default del pipeline. */
export const ONNXMODELSIZEGTE = "gte";

/** Tamaño de modelo soportado (derivado de las constantes para no repetir literales). */
export type ModelSize =
  | typeof ONNXMODELSIZESMALL
  | typeof ONNXMODELSIZELARGE
  | typeof ONNXMODELSIZEGTE;
