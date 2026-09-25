/**
 * Configuración del servicio unificado, leída de variables de entorno.
 *
 * Los valores por defecto NO están hardcodeados aquí: viven en
 * `shared/constants/` agrupados por dominio, y este archivo solo mapea cada
 * env var sobre su default:
 *   - ports/    → puertos de los 3 listeners (predicción, modelos, Qdrant API)
 *   - models/   → tamaños de modelos ONNX
 *   - predict/  → afinamiento del pipeline (umbral adaptativo, keywords, recall)
 *   - qdrant/   → URL del motor Qdrant y nombres de colecciones
 */

import { ONNXMODELSIZESMALL } from "./shared/constants/predict/version.constants";
import {
  STRICTPORTEMBEDDING,
  STRICTPORTPREDICT,
  STRICTPORTQDRANT,
} from "./shared/constants/ports/general.port";
import {
  familyGatePenaltyDefault,
  gapThresholdDefault,
  keywordBoostDefault,
  keywordTopKDefault,
  learnDecayLambdaDefault,
  learnEnabledDefault,
  learnEtaDefault,
  learnMaxPostingsDefault,
  learnMaxTermsPerToolDefault,
  learnMinEventsDefault,
  learnNegativeGammaDefault,
  learnPersistDefault,
  learnSeedWeightDefault,
  learnTermMinWeightDefault,
  learnWeightDefault,
  maxCategoriesDefault,
  maxOutputToolsDefault,
  maxToolsDefault,
  minScoreDefault,
  minToolsDefault,
  nameAffinityBoostDefault,
  outputToolsCeiling,
  recallLimitDefault,
} from "./shared/constants/predict/general.predict";
import {
  KEYWORDSCOLLECTIONDEFAULT,
  MEMORIECOLLECTIONDEFAULTS,
  QDRANTURLDEFAULT,
  SYNONYMSCOLLECTIONDEFAULT,
  TOOLSCOLLECTIONDEFAULT,
} from "./shared/constants/qdrant/general.constant";
import {
  energyTauDefault,
  energyTempDefault,
  gateBetaDefault,
  gateGammaDefault,
  gateTextMinDefault,
  juicioEnabledDefault,
  juicioPostTopKDefault,
  maxsimMaxTokensDefault,
  maxsimWeightDefault,
  nliMarginDefault,
  noopMarginDefault,
  problemMarginDefault,
  rerankerAlphaDefault,
  rerankerModelPathDefault,
  specificityRhoDefault,
} from "./shared/constants/juicio";

export interface AppConfig {
  /** Puerto del API de predicción (endpoints /predict, /tools, /memory/predict...). */
  portPredict: number;
  /** Puerto del API de modelos (POST /embed, e5-large + e5-small). */
  portEmbed: number;
  /** Puerto del API de Qdrant (recall BM25, keywording, indexado de tools). */
  portQdrant: number;

  onnxEnabled: boolean;
  onnxModelsPath: string;
  /** Modelo del pipeline: solo e5-small (migrado de large por rendimiento). */
  onnxModelSize: typeof ONNXMODELSIZESMALL;

  adaptiveMinTools: number;
  adaptiveMaxTools: number;
  adaptiveGapThreshold: number;
  adaptiveMinScore: number;
  /** Refuerzo (suma) por match de keywords internas sobre el coseno semántico. */
  keywordBoost: number;
  /**
   * Refuerzo (suma) por afinidad entre las palabras clave de la consulta y el
   * NOMBRE de la herramienta (env `NAME_AFFINITY_BOOST`).
   */
  nameAffinityBoost: number;
  /**
   * Penalización (resta) a las herramientas de una familia (namespace) distinta
   * de la nombrada en las palabras clave (env `FAMILY_GATE_PENALTY`).
   */
  familyGatePenalty: number;
  /** Tamaño del top-K tras la reducción cross-idioma (capa 2). */
  keywordTopK: number;
  /** Tope de candidatas recuperadas del recall BM25 por consulta. */
  recallLimit: number;
  /** Tope final de tools devueltas tras el umbral adaptativo (env `MAX_OUTPUT_TOOLS`, rango 0-150). */
  maxOutputTools: number;
  /**
   * Etapa A: tope de categorías (grupos) que pasan al decisor de herramientas
   * (env `MAX_CATEGORIES`).
   */
  maxCategories: number;

  // ── Capa de aprendizaje léxico por canal (docs/entrenamiento-prediccion.md) ──
  /** Interruptor maestro: sin él, `seedScope`/`observe`/scoring quedan inertes. */
  learnEnabled: boolean;
  /** Peso del score aprendido en la fusión (comparable a `keywordBoost`). */
  learnWeight: number;
  /** Tasa de refuerzo positivo término → tool. */
  learnEta: number;
  /** Tasa de penalización negativa término → tool. */
  learnNegativeGamma: number;
  /** Decaimiento exponencial por día. */
  learnDecayLambda: number;
  /** Señales mínimas del canal antes de que la capa puntúe (cold start). */
  learnMinEvents: number;
  /** Peso inicial de los términos de la semilla. */
  learnSeedWeight: number;
  /** Tope de términos por herramienta. */
  learnMaxTermsPerTool: number;
  /** Umbral de poda de términos. */
  learnTermMinWeight: number;
  /** Tope de acumulaciones por consulta (cota de latencia). */
  learnMaxPostings: number;
  /** Persistencia del perfil en `tool_lexicon` (Fase 4). */
  learnPersist: boolean;

  // ── Capa de juicio (docs/juicio.md) ──────────────────────────────────────
  /** Interruptor maestro: sin él, pre/post de juicio son pasa-through. */
  juicioEnabled: boolean;
  /** Margen entre pools rechazo/confirmación del veredicto NLI de estado. */
  juicioNliMargin: number;
  /** Carpeta del modelo NLI ONNX (vacío = fallback por arquetipos). */
  juicioNliModelPath: string;
  /** Margen del prototipo __NOOP__ sobre el mejor score de herramienta. */
  juicioNoopMargin: number;
  /** Temperatura T de la energía libre sobre las similitudes. */
  juicioEnergyTemp: number;
  /** Umbral τ de energía para abstención out-of-distribution. */
  juicioEnergyTau: number;
  /** β de la sigmoide de la puerta de coherencia (pendiente). */
  juicioGateBeta: number;
  /** γ de la sigmoide de la puerta de coherencia (desplazamiento). */
  juicioGateGamma: number;
  /** λ mínima para concatenar historial al prompt. */
  juicioGateTextMin: number;
  /** Carpeta del cross-encoder ONNX (vacío = inactivo). */
  juicioRerankerModelPath: string;
  /** Peso del logit del cross-encoder en la fusión. */
  juicioRerankerAlpha: number;
  /** Peso del MaxSim token-level en la fusión del re-rank de juicio. */
  juicioMaxsimWeight: number;
  /** Tope de tokens por texto en el MaxSim. */
  juicioMaxsimMaxTokens: number;
  /** ρ de la penalización por difusión semántica. */
  juicioSpecificityRho: number;
  /** Margen exigido al problemSpace para reemplazar al score fusionado. */
  juicioProblemMargin: number;
  /** Tope de candidatas re-puntuadas en el post de juicio. */
  juicioPostTopK: number;

  qdrantEnabled: boolean;
  qdrantUrl: string;
  qdrantApiKey: string;
  // Colecciones del motor Qdrant: fijas por constantes (ya no se leen de env).
  toolsCollection: string;
  keywordsCollection: string;
  synonymsCollection: string;
  memoriesCollection: string;
}

function int(v: string | undefined, d: number): number {
  if (v === undefined || v === "") return d;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : d;
}

function float(v: string | undefined, d: number): number {
  if (v === undefined || v === "") return d;
  const n = Number.parseFloat(v);
  return Number.isFinite(n) ? n : d;
}

function bool(v: string | undefined, d: boolean): boolean {
  if (v === undefined || v === "") return d;
  return ["1", "true", "yes", "on"].includes(v.toLowerCase());
}

/** Lee un string de entorno recortando espacios; vacío o solo espacios → default. */
function str(v: string | undefined, d: string): string {
  if (v === undefined) return d;
  const s = v.trim();
  return s === "" ? d : s;
}

/** Resuelve la carpeta de modelos: env explícito → rutas del paquete → CWD. */
function resolveModelsPath(env: NodeJS.ProcessEnv): string {
  const explicit = (env.ONNX_MODELS_PATH ?? "").trim();
  if (explicit) return explicit;
  // Relativas al paquete: src/… → ../models, dist/src/… → ../../models.
  const fromPackage = [
    require("node:path").resolve(__dirname, "../models"),
    require("node:path").resolve(__dirname, "../../models"),
  ];
  for (const p of [...fromPackage, "./models", "../models"]) {
    try {
      if (require("node:fs").existsSync(p)) return p;
    } catch {
      /* noop */
    }
  }
  return fromPackage[0];
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  // Tope final de tools de salida: env opcional recortado al rango permitido [0, 150].
  const maxOutputTools = Math.max(
    0,
    Math.min(int(env.MAX_OUTPUT_TOOLS, maxOutputToolsDefault), outputToolsCeiling),
  );
  // Etapa A: al menos 1 categoría para que el decisor siempre tenga candidatas.
  const maxCategories = Math.max(1, int(env.MAX_CATEGORIES, maxCategoriesDefault));

  return {
    // ── Puertos de los listeners HTTP ───────────────────────────────────
    // PORT_PREDICT también hereda de PORT (nombre genérico del server Go).
    portPredict: int(env.PORT_PREDICT ?? env.PORT, STRICTPORTPREDICT),
    portEmbed: int(env.PORT_EMBED, STRICTPORTEMBEDDING),
    portQdrant: int(env.PORT_QDRANT, STRICTPORTQDRANT),

    // ── Modelos ONNX ────────────────────────────────────────────────────
    onnxEnabled: bool(env.ONNX_ENABLED, true),
    onnxModelsPath: resolveModelsPath(env),
    // Pipeline sobre e5-small (único tamaño activo).
    onnxModelSize: ONNXMODELSIZESMALL,

    // ── Umbral adaptativo y keywords (capas 2-3 del pipeline) ───────────
    adaptiveMinTools: int(env.ONNX_ADAPTIVE_MIN_TOOLS, minToolsDefault),
    adaptiveMaxTools: int(env.ONNX_ADAPTIVE_MAX_TOOLS, maxToolsDefault),
    adaptiveGapThreshold: float(env.ONNX_ADAPTIVE_GAP_THRESHOLD, gapThresholdDefault),
    adaptiveMinScore: float(env.ONNX_ADAPTIVE_MIN_SCORE, minScoreDefault),
    keywordBoost: float(env.KEYWORD_BOOST, keywordBoostDefault),
    nameAffinityBoost: float(env.NAME_AFFINITY_BOOST, nameAffinityBoostDefault),
    familyGatePenalty: float(env.FAMILY_GATE_PENALTY, familyGatePenaltyDefault),
    keywordTopK: int(env.KEYWORD_TOP_K, keywordTopKDefault),
    recallLimit: int(env.RECALL_LIMIT, recallLimitDefault),
    maxOutputTools,
    maxCategories,

    // ── Aprendizaje léxico por canal ────────────────────────────────────
    learnEnabled: bool(env.LEARN_ENABLED, learnEnabledDefault),
    learnWeight: float(env.LEARN_WEIGHT, learnWeightDefault),
    learnEta: float(env.LEARN_ETA, learnEtaDefault),
    learnNegativeGamma: float(env.LEARN_NEGATIVE_GAMMA, learnNegativeGammaDefault),
    learnDecayLambda: float(env.LEARN_DECAY_LAMBDA, learnDecayLambdaDefault),
    learnMinEvents: int(env.LEARN_MIN_EVENTS, learnMinEventsDefault),
    learnSeedWeight: float(env.LEARN_SEED_WEIGHT, learnSeedWeightDefault),
    learnMaxTermsPerTool: int(env.LEARN_MAX_TERMS_PER_TOOL, learnMaxTermsPerToolDefault),
    learnTermMinWeight: float(env.LEARN_TERM_MIN_WEIGHT, learnTermMinWeightDefault),
    learnMaxPostings: int(env.LEARN_MAX_POSTINGS, learnMaxPostingsDefault),
    learnPersist: bool(env.LEARN_PERSIST, learnPersistDefault),

    // ── Capa de juicio ────────────────────────────────────────────────────
    juicioEnabled: bool(env.JUICIO_ENABLED, juicioEnabledDefault),
    juicioNliMargin: float(env.JUICIO_NLI_MARGIN, nliMarginDefault),
    juicioNliModelPath: str(env.JUICIO_NLI_MODEL_PATH, ""),
    juicioNoopMargin: float(env.JUICIO_NOOP_MARGIN, noopMarginDefault),
    juicioEnergyTemp: float(env.JUICIO_ENERGY_T, energyTempDefault),
    juicioEnergyTau: float(env.JUICIO_ENERGY_TAU, energyTauDefault),
    juicioGateBeta: float(env.JUICIO_GATE_BETA, gateBetaDefault),
    juicioGateGamma: float(env.JUICIO_GATE_GAMMA, gateGammaDefault),
    juicioGateTextMin: float(env.JUICIO_GATE_TEXT_MIN, gateTextMinDefault),
    juicioRerankerModelPath: str(env.JUICIO_RERANKER_MODEL_PATH, rerankerModelPathDefault),
    juicioRerankerAlpha: float(env.JUICIO_RERANKER_ALPHA, rerankerAlphaDefault),
    juicioMaxsimWeight: float(env.JUICIO_MAXSIM_WEIGHT, maxsimWeightDefault),
    juicioMaxsimMaxTokens: int(env.JUICIO_MAXSIM_MAX_TOKENS, maxsimMaxTokensDefault),
    juicioSpecificityRho: float(env.JUICIO_SPECIFICITY_RHO, specificityRhoDefault),
    juicioProblemMargin: float(env.JUICIO_PROBLEM_MARGIN, problemMarginDefault),
    juicioPostTopK: int(env.JUICIO_POST_TOP_K, juicioPostTopKDefault),

    // ── Motor Qdrant externo ────────────────────────────────────────────
    qdrantEnabled: bool(env.QDRANT_ENABLED, true),
    qdrantUrl: str(env.QDRANT_URL, QDRANTURLDEFAULT),
    qdrantApiKey: str(env.QDRANT_API_KEY, ""),
    // Colecciones fijas (definidas en shared/constants), no configurables por env.
    toolsCollection: TOOLSCOLLECTIONDEFAULT,
    keywordsCollection: KEYWORDSCOLLECTIONDEFAULT,
    synonymsCollection: SYNONYMSCOLLECTIONDEFAULT,
    memoriesCollection: MEMORIECOLLECTIONDEFAULTS,
  };
}
