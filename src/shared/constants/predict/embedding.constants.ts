/**
 * Metadatos físicos de los modelos de embeddings e5 (ONNX).
 *
 * Antes vivían hardcodeados dentro de `embedding/embedding-engine.ts`; aquí son
 * la fuente única de dimensiones, tokenización y rutas de archivo en disco.
 */
import type { ModelSize } from "./version.constants";

/** Dimensión del vector de salida por tamaño (e5-small → 384, e5-large → 1024). */
export const onnxEmbedDim: Record<ModelSize, number> = { small: 384, large: 1024 };

/** Tope de tokens por secuencia de entrada (truncado). */
export const onnxMaxTokens = 512;

/** Plantilla de carpeta del modelo en disco: `multilingual-e5-{size}-onnx`. */
export function onnxModelDirName(size: ModelSize): string {
  return `multilingual-e5-${size}-onnx`;
}

/** Nombre del archivo de pesos ONNX dentro de la carpeta del modelo. */
export const onnxWeightsFile = "model.onnx";
