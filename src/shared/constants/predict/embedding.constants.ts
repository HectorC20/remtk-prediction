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

/**
 * Umbrales dependientes de la ESCALA del coseno, calibrados por modelo.
 *
 * Cada modelo comprime el coseno de forma distinta (e5 satura alto), así que los
 * umbrales absolutos calibrados para e5 rompen con otros modelos. Medido en
 * `tests/e2e-nest-log-reproduction.test.ts`:
 *   e5-small/large → nodos/cosenos ~0.74-0.87
 *   gte            → escala intermedia (ver umbrales por modelo abajo)
 */
import { gapThresholdDefault, minScoreDefault } from "./general.predict";

/**
 * Piso de relevancia del umbral adaptativo por modelo (env
 * `ONNX_ADAPTIVE_MIN_SCORE`). Se conserva el valor histórico de e5 (0.8), que la
 * escala intermedia de gte sostiene.
 */
export const adaptiveMinScoreByModel: Record<ModelSize, number> = {
  small: minScoreDefault,
  large: minScoreDefault,
  gte: minScoreDefault,
};

/**
 * Gap natural del umbral adaptativo por modelo (env
 * `ONNX_ADAPTIVE_GAP_THRESHOLD`). La banda de relevancia (`gap × 2`) se deriva
 * de él; se mantiene 0.03.
 */
export const adaptiveGapThresholdByModel: Record<ModelSize, number> = {
  small: gapThresholdDefault,
  large: gapThresholdDefault,
  gte: gapThresholdDefault,
};

/**
 * Similitud mínima para inferir una arista PREREQUISITE en el grafo por modelo
 * (`graph-cache.service.inferEdges`). Valor histórico calibrado para e5.
 */
export const edgeInferenceMinSimByModel: Record<ModelSize, number> = {
  small: 0.835,
  large: 0.835,
  gte: 0.835,
};

/**
 * Umbral de coseno del juez de memoria (detección de negaciones técnicas) por
 * tamaño de modelo. La escala del coseno depende del modelo: e5-large satura
 * más alto que gte, así que el umbral va calibrado por tamaño en vez de ser una
 * constante absoluta en el orquestador. Medido en `tests/e2e-nest-log-reproduction.test.ts`:
 *   large → negación 0.884 / hecho 0.762   (umbral 0.87)
 *   gte   → negación 0.610 / hecho 0.436   (umbral 0.52)
 * `small` conserva el valor histórico (no recalibrado).
 */
export const memoryDenialThreshold: Record<ModelSize, number> = {
  small: 0.87,
  large: 0.87,
  gte: 0.52,
};

/**
 * Umbrales del detector de «pivote funcional autónomo» (`juicio.service.pre`),
 * que aísla el historial cuando el turno salta a otro dominio de herramientas.
 *
 * Un pivote exige que el turno mapee con CLARIDAD a una herramienta
 * (`topScore >= topScore`); sin esa condición, un turno débil dispara el pivote
 * de forma espuria. Los tres umbrales relativos conservan los valores
 * históricos de e5.
 */
export const pivotThresholdsByModel: Record<
  ModelSize,
  { shiftHigh: number; topScore: number; shiftLow: number; prominence: number }
> = {
  small: { shiftHigh: 0.04, topScore: 0.85, shiftLow: 0.05, prominence: 0.035 },
  large: { shiftHigh: 0.04, topScore: 0.85, shiftLow: 0.05, prominence: 0.035 },
  gte: { shiftHigh: 0.04, topScore: 0.85, shiftLow: 0.05, prominence: 0.035 },
};

/**
 * Umbrales del subsistema de SKILLS (`skill-memory.service`), calibrados por
 * modelo igual que el resto del pipeline. Los tres umbrales son sensibles a la
 * ESCALA del coseno; los valores históricos (0.82 / 0.015 / 0.75 / 0.6) se
 * midieron para e5, cuya escala satura alto, y se conservan para small/large/gte.
 *
 * `anchorMinSim` es un piso de relevancia (no decide solo: lo acompaña
 * `anchorMargin`, la ventaja mínima sobre el ancla rival de OTRO identificador).
 * `skillMatchMin` es el piso para atajar el planner con `execute_direct`.
 * `actionSimMin` habilita el refuerzo de la acción pendiente al resolver la
 * entidad de la sesión.
 */
export const skillThresholdsByModel: Record<
  ModelSize,
  { anchorMinSim: number; anchorMargin: number; skillMatchMin: number; actionSimMin: number }
> = {
  small: { anchorMinSim: 0.82, anchorMargin: 0.015, skillMatchMin: 0.75, actionSimMin: 0.6 },
  large: { anchorMinSim: 0.82, anchorMargin: 0.015, skillMatchMin: 0.75, actionSimMin: 0.6 },
  gte: { anchorMinSim: 0.82, anchorMargin: 0.015, skillMatchMin: 0.75, actionSimMin: 0.6 },
};
