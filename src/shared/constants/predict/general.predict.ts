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

/**
 * Cota (ms) que espera `/predict` por el warm-up de un scope recién registrado.
 *
 * El grafo de pre-requisitos se construye en background tras `POST /tools`; sin
 * esta espera el primer turno de un catálogo frío cae en la ruta plana y entrega
 * la herramienta de mutación aislada, sin sus antecedentes de descubrimiento
 * (AGENT.md §2.C). 0 desactiva la espera.
 */
export const predictWarmupWaitMsDefault = 15_000;

/** Máximo permitido para `PREDICT_WARMUP_WAIT_MS`. */
export const predictWarmupWaitMsCeiling = 60_000;

export const TOPIC_SHIFT_THRESHOLD = 0.86;
/** Máximo de mensajes previos incorporados al contexto de predicción. */
export const MAX_HISTORY_MESSAGES = 4;
/** Tope de caracteres por mensaje de historial (evita prompts gigantes). */
export const MAX_MESSAGE_CHARS = 300;
/**
 * Tope de caracteres del enriquecimiento interno de la consulta (`intentContext`).
 * Es prosa que el consumidor produce ANTES de preguntar (interpretación del
 * planificador); se recorta porque su tamaño no lo limita ningún contrato de
 * mensaje y un bloque largo desplazaria el texto propio de la subtarea.
 */
export const MAX_INTENT_CONTEXT_CHARS = 600;
/**
 * Segmentos del enriquecimiento que se conservan como máximo para una consulta.
 * El bloque describe TODO el plan; lo que le sirve a una subtarea son uno o dos
 * segmentos, y pasar de ahí vuelve a diluir el texto propio de la tarea.
 */
export const MAX_INTENT_SEGMENTS = 3;

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

// ── Variante C · núcleo de 3 discriminadores (ruta conmutable) ───────
/**
 * Interruptor de la ruta Variante C (env `VARIANTE_C_ENABLED`). Default ON:
 * `predict()` enruta por el núcleo de 3 discriminadores (12/12 y 6/6 en los
 * arneses), que sustituye al ranking del pipeline por ser el ganador medido.
 * El pipeline previo queda como respaldo y sigue disponible apagando el flag
 * (env `VARIANTE_C_ENABLED=false`), con lo que el rollback es trivial.
 */
export const varianteCEnabledDefault = true;
/** Tope de candidatas del recall por unión (turno ∪ contexto). */
export const C_RECALL_K = 8;
/** Margen (max−media del coseno del turno) que marca un turno AUTÓNOMO. */
export const C_MARGIN = 0.13;
/**
 * Compuerta anafórica: si el turno se parece a un arquetipo de apoyo por encima
 * de este coseno, es DEPENDIENTE aunque tenga margen (debe resolver con el
 * contexto). Medido: autónomos ≤ 0.584; turno anafórico = 0.787.
 */
export const C_ANAF_TAU = 0.7;
/** Fusión AUTÓNOMA: D2 (MaxSim por cláusula) manda; D1 (coseno) desempata. */
export const C_W1_AUTO = 0.25;
export const C_W2_AUTO = 0.7;
/** Fusión DEPENDIENTE: D1 sobre el CONTEXTO manda; D2 desempata. */
export const C_W1_DEP = 0.85;
export const C_W2_DEP = 0.1;
/** Peso del estado (D3) en la fusión. */
export const C_W3 = 0.05;
/** Umbrales del detector de estado (D3) por coseno contra arquetipos. */
export const C_REJ_TAU = 0.6;
export const C_CONF_TAU = 0.6;
/** Umbral de abstención: si el mejor `fused` < τ ⇒ ∅. */
export const C_TAU = 0.4;
/** Banda multi-intención: se aceptan los candidatos dentro de δ del primero. */
export const C_DELTA = 0.12;
/** Unidades de intención del turno (conjunciones y puntuación). */
export const C_CLAUSE_SPLIT = /[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b/gi;
/** Arquetipos de ESTADO (D3): prototipos de rechazo/confirmación del turno. */
export const C_REJECT_ARCH = [
  "no, todavía no",
  "no gracias, mejor no",
  "no por ahora",
  "cancélalo, no quiero",
];
export const C_CONFIRM_ARCH = ["sí, hazlo, adelante", "dale, procede", "sí, por favor, confirma"];
/** Arquetipos ANAFÓRICOS: turnos de apoyo sin acción propia (compuerta). */
export const C_ANAPHORA_ARCH = [
  "hazlo con eso",
  "sí, ese mismo",
  "continúa con lo mismo",
  "aplica lo mismo a esa",
  "y eso también",
  "hazlo",
  "ahora con esa",
  "hazlo también para esa",
];

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
