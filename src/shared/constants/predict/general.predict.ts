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

/**
 * Etapa A del pipeline: tope de categorías (grupos) que alimentan al decisor
 * de herramientas (env `MAX_CATEGORIES`).
 */
export const maxCategoriesDefault = 60;

/**
 * Tope máximo permitido de herramientas de la etapa B / salida final
 * (env `MAX_OUTPUT_TOOLS`).
 */
export const outputToolsCeiling = 150;

/** Default del tope final de herramientas de salida (recorte tras el umbral). */
export const maxOutputToolsDefault = outputToolsCeiling;

/** Gap natural para cortar el ranking (escala de coseno e5-small). */
export const gapThresholdDefault = 0.03;

/** Score mínimo aceptado en el coseno semántico. */
export const minScoreDefault = 0.8;

/** Refuerzo (suma) por confirmación léxica BM25 sobre el coseno semántico. */
export const keywordBoostDefault = 0.15;

/**
 * Refuerzo (suma) por afinidad entre las palabras clave de la consulta y el
 * NOMBRE de la herramienta (familia + entidad).
 *
 * Es la señal que discrimina DENTRO de un complemento: cuando todas las tools
 * comparten `group`, `category` y `tags` (caso de un plugin), ni el coseno ni
 * BM25 separan `mitumbes_item_*` de `schedule_*`; el nombre sí.
 *
 * La magnitud (0.1) supera `gapThresholdDefault` (0.03) a propósito: así la
 * familia correcta abre el gap natural sobre el que la etapa B recorta, en vez
 * de quedar mezclada con el resto del complemento.
 */
export const nameAffinityBoostDefault = 0.1;

/**
 * Penalización (resta) a las herramientas de una familia DISTINTA de la
 * nombrada en las palabras clave (env `FAMILY_GATE_PENALTY`).
 *
 * Solo actúa cuando la consulta nombra explícitamente el namespace de alguna
 * herramienta del catálogo (p. ej. `mitumbes`, como hace el replanteo de la 2ª
 * pasada): en ese momento las tools de otras familias (`schedule_*`) no tienen
 * que ver con esas palabras clave y deben caer fuera de la banda del umbral
 * adaptativo. La magnitud (0.2) es ~7× `gapThresholdDefault`: incluso cuando la
 * herramienta ajena encabeza el coseno, la resta la deja fuera de la tanda del
 * top. Sin familia nombrada la penalización es 0 y el ranking no cambia.
 */
export const familyGatePenaltyDefault = 0.2;

/** Alpha de propagación de pre-requisitos en el grafo (S + alpha·Aᵀ·S). */
export const graphPropagationAlphaDefault = 0.2;

/**
 * Tamaño del top-K tras la reducción cross-idioma (capa 2). Es el pool de
 * candidatas que alimenta la etapa A (categorías) y la etapa B (decisor de
 * herramientas), por lo que debe cubrir el tope de salida.
 */
export const keywordTopKDefault = maxOutputToolsDefault;

/** Tope de candidatas que se recuperan del recall BM25 por consulta. */
export const recallLimitDefault = 50;

export const TOPIC_SHIFT_THRESHOLD = 0.86;
/** Máximo de mensajes previos incorporados al contexto de predicción. */
export const MAX_HISTORY_MESSAGES = 4;
/** Tope de caracteres por mensaje de historial (evita prompts gigantes). */
export const MAX_MESSAGE_CHARS = 300;

export const MAX_TRACES = 200;

// ── Capa de aprendizaje léxico por canal (ver docs/entrenamiento-prediccion.md) ──

/** Interruptor maestro de la capa de aprendizaje léxico. */
export const learnEnabledDefault = true;

/** Peso del score aprendido en la fusión (comparable a `KEYWORD_BOOST`). */
export const learnWeightDefault = 0.25;

/** Tasa de refuerzo positivo (término → tool que funcionó). */
export const learnEtaDefault = 0.5;

/** Tasa de penalización negativa (término → tool predicha que no funcionó). */
export const learnNegativeGammaDefault = 0.15;

/** Decaimiento exponencial por día (evita que el perfil se fosilice). */
export const learnDecayLambdaDefault = 0.02;

/** Señales mínimas del canal antes de que la capa puntúe (cold start, R7). */
export const learnMinEventsDefault = 3;

/** Peso inicial de los términos de la semilla (el uso real lo domina pronto). */
export const learnSeedWeightDefault = 0.3;

/** Tope de términos por herramienta (poda por peso). */
export const learnMaxTermsPerToolDefault = 64;

/** Umbral de poda de términos del perfil. */
export const learnTermMinWeightDefault = 0.05;

/** Tope de acumulaciones por consulta (cota dura de latencia). */
export const learnMaxPostingsDefault = 200;

/** Persistencia del perfil en la colección Qdrant `tool_lexicon` (Fase 4). */
export const learnPersistDefault = false;

/** Milisegundos de un día (unidad de Δt para el decaimiento exponencial). */
export const LEARN_DAY_MS = 86_400_000;

/** Peso del match de tema (ancla de remtk-memory) en el score de interés compuesto. */
export const INTEREST_TOPIC_WEIGHT = 0.6;
/** Peso de la intención (arquetipos) en el score de interés compuesto. */
export const INTEREST_INTENT_WEIGHT = 0.4;
/** Tope de anclas de tema evaluadas por turno (acota el costo de embeddings). */
export const INTEREST_MAX_ANCHORS = 32;
/** Tope de matches de ancla devueltos (ordenados desc). */
export const INTEREST_TOP_MATCHES = 5;

// ── Estado continuo de sesión ───────────────────────────────────────
/** Ponderación del estado z_t previo en la combinación convexa 70/30. */
export const BLEND_PREV = 0.7;
/** Ponderación del vector nuevo en la combinación convexa 70/30. */
export const BLEND_NEW = 0.3;

// ── Ponderación topológica de aristas en el grafo de herramientas ────
/** Peso de arista de pre-requisito dirigida. */
export const PREREQUISITE_WEIGHT = 0.85;
/** Peso de arista de co-ocurrencia por grupo compartido. */
export const CO_OCCURRENCE_WEIGHT = 0.4;
/** Peso de arista de exclusión mutua / conflicto. */
export const MUTUALLY_EXCLUSIVE_WEIGHT = 1.0;

// ── Clasificador de turnos ──────────────────────────────────────────
/** Margen de ambigüedad para desempate entre confirmación y nueva consulta. */
export const CLASSIFICATION_MARGIN = 0.03;
/** Umbral de similitud para confirmación con matiz contextual. */
export const NUANCE_THRESHOLD = 0.55;
/** Umbral para clasificar como meta-pregunta de capacidades del sistema. */
export const META_QUESTION_THRESHOLD = 0.9;

// ── Complejidad de planes ───────────────────────────────────────────
export const COMPLEXITY_COUNT_HIGH = 20;
export const COMPLEXITY_AVG_HIGH = 0.65;
export const COMPLEXITY_COUNT_MID = 10;
export const COMPLEXITY_AVG_MID = 0.55;

// ── Debugger ────────────────────────────────────────────────────────
/** Número máximo de trazas históricas retornadas en el snapshot de /debug. */
export const DEBUG_MAX_SNAPSHOT_TRACES = 20;
