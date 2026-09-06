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
  gapThresholdDefault,
  keywordBoostDefault,
  keywordTopKDefault,
  maxOutputToolsDefault,
  maxToolsDefault,
  minScoreDefault,
  minToolsDefault,
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
  /** Tamaño del top-K tras la reducción cross-idioma (capa 2). */
  keywordTopK: number;
  /** Tope de candidatas recuperadas del recall BM25 por consulta. */
  recallLimit: number;
  /** Tope final de tools devueltas tras el umbral adaptativo (env `MAX_OUTPUT_TOOLS`, rango 0-50). */
  maxOutputTools: number;

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
  // Tope final de tools de salida: env opcional recortado al rango permitido [0, 50].
  const maxOutputTools = Math.max(
    0,
    Math.min(int(env.MAX_OUTPUT_TOOLS, maxOutputToolsDefault), outputToolsCeiling),
  );

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
    keywordTopK: int(env.KEYWORD_TOP_K, keywordTopKDefault),
    recallLimit: int(env.RECALL_LIMIT, recallLimitDefault),
    maxOutputTools,

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
