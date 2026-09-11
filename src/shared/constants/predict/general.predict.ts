/**
 * Defaults de afinamiento del pipeline de predicción.
 *
 * Se aplican cuando la variable de entorno correspondiente no está definida;
 * el mapeo env → valor vive en `src/config.ts`.
 */

/** Mínimo de tools que entrega el umbral adaptativo (capa 3). */
export const minToolsDefault = 2;

/** Máximo de tools que entrega el umbral adaptativo (capa 3). */
export const maxToolsDefault = 30;

/** Tope máximo permitido de herramientas de salida (env `MAX_OUTPUT_TOOLS`). */
export const outputToolsCeiling = 50;

/** Default del tope final de herramientas de salida (recorte tras el umbral). */
export const maxOutputToolsDefault = outputToolsCeiling;

/** Gap natural para cortar el ranking (escala de coseno e5-small). */
export const gapThresholdDefault = 0.03;

/** Score mínimo aceptado en el coseno semántico. */
export const minScoreDefault = 0.8;

/** Refuerzo (suma) por confirmación léxica BM25 sobre el coseno semántico. */
export const keywordBoostDefault = 0.15;

/** Alpha de propagación de pre-requisitos en el grafo (S + alpha·Aᵀ·S). */
export const graphPropagationAlphaDefault = 0.2;

/** Tamaño del top-K tras la reducción cross-idioma (capa 2). */
export const keywordTopKDefault = 20;

/** Tope de candidatas que se recuperan del recall BM25 por consulta. */
export const recallLimitDefault = 50;

export const TOPIC_SHIFT_THRESHOLD = 0.86;
/** Máximo de mensajes previos incorporados al contexto de predicción. */
export const MAX_HISTORY_MESSAGES = 4;
/** Tope de caracteres por mensaje de historial (evita prompts gigantes). */
export const MAX_MESSAGE_CHARS = 300;

export const MAX_TRACES = 200;

/** Peso del match de tema (ancla de remtk-memory) en el score de interés compuesto. */
export const INTEREST_TOPIC_WEIGHT = 0.6;
/** Peso de la intención (arquetipos) en el score de interés compuesto. */
export const INTEREST_INTENT_WEIGHT = 0.4;
/** Tope de anclas de tema evaluadas por turno (acota el costo de embeddings). */
export const INTEREST_MAX_ANCHORS = 32;
/** Tope de matches de ancla devueltos (ordenados desc). */
export const INTEREST_TOP_MATCHES = 5;
