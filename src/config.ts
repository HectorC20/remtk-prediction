/** Configuración del servicio unificado, leída de variables de entorno. */

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
  onnxModelSize: "small";

  adaptiveMinTools: number;
  adaptiveMaxTools: number;
  adaptiveGapThreshold: number;
  adaptiveMinScore: number;
  /** Refuerzo (suma) por match de keywords internas sobre el coseno semántico. */
  keywordBoost: number;
  /** Tamaño del top-K tras la reducción cross-idioma (capa 2). */
  keywordTopK: number;

  qdrantEnabled: boolean;
  qdrantUrl: string;
  qdrantApiKey: string;
  toolsCollection: string;
  keywordsCollection: string;
  synonymsCollection: string;
  memoriesCollection: string;

  recallLimit: number;
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
  return {
    portPredict: int(env.PORT_PREDICT ?? env.PORT, 6776),
    portEmbed: int(env.PORT_EMBED, 6777),
    portQdrant: int(env.PORT_QDRANT, 6778),

    onnxEnabled: bool(env.ONNX_ENABLED, true),
    onnxModelsPath: resolveModelsPath(env),
    onnxModelSize: "small",

    adaptiveMinTools: int(env.ONNX_ADAPTIVE_MIN_TOOLS, 2),
    adaptiveMaxTools: int(env.ONNX_ADAPTIVE_MAX_TOOLS, 30),
    adaptiveGapThreshold: float(env.ONNX_ADAPTIVE_GAP_THRESHOLD, 0.03),
    adaptiveMinScore: float(env.ONNX_ADAPTIVE_MIN_SCORE, 0.8),
    keywordBoost: float(env.KEYWORD_BOOST, 0.15),
    keywordTopK: int(env.KEYWORD_TOP_K, 20),

    qdrantEnabled: bool(env.QDRANT_ENABLED, true),
    qdrantUrl: (env.QDRANT_URL ?? "http://localhost:6333").trim(),
    qdrantApiKey: (env.QDRANT_API_KEY ?? "").trim(),
    toolsCollection: (env.QDRANT_MCP_TOOLS_COLLECTION ?? "mcp_tools").trim(),
    keywordsCollection: (env.QDRANT_TOOL_KEYWORDS_COLLECTION ?? "tool_keywords").trim(),
    synonymsCollection: (env.QDRANT_QUERY_SYNONYMS_COLLECTION ?? "query_synonyms").trim(),
    memoriesCollection: (env.QDRANT_MEMORIES_COLLECTION ?? "contextual_memories").trim(),

    recallLimit: int(env.RECALL_LIMIT, 50),
  };
}
