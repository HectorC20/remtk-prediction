/**
 * Metadatos físicos de los modelos de embeddings (ONNX).
 *
 * Antes vivían hardcodeados dentro de `embedding/embedding-engine.ts`; aquí son
 * la fuente única de dimensiones, tokenización, pooling, prefijos y rutas de
 * archivo en disco por tamaño de modelo.
 */
import type { ModelSize } from "./version.constants";

/** Dimensión del vector de salida por tamaño (e5-small 384, e5-large 1024, gte 768). */
export const onnxEmbedDim: Record<ModelSize, number> = { small: 384, large: 1024, gte: 768 };

/** Tope de tokens por secuencia de entrada (truncado). */
export const onnxMaxTokens = 512;

/** Carpeta del modelo en disco por tamaño. */
export const onnxModelDir: Record<ModelSize, string> = {
  small: "multilingual-e5-small-onnx",
  large: "multilingual-e5-large-onnx",
  gte: "gte-multilingual-base-onnx",
};

/** Nombre del archivo de pesos ONNX dentro de la carpeta del modelo. */
export const onnxWeightsFileBySize: Record<ModelSize, string> = {
  small: "model.onnx",
  large: "model.onnx",
  gte: "model_int8.onnx",
};

/** Pooling de la última capa: e5 promedia por máscara (mean); gte usa el token CLS. */
export const onnxPooling: Record<ModelSize, "mean" | "cls"> = {
  small: "mean",
  large: "mean",
  gte: "cls",
};

/** Prefijos de texto: e5 los exige ("query:"/"passage:"); gte no usa prefijo. */
export const onnxTextPrefix: Record<ModelSize, { query: string; passage: string }> = {
  small: { query: "query: ", passage: "passage: " },
  large: { query: "query: ", passage: "passage: " },
  gte: { query: "", passage: "" },
};
