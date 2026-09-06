/**
 * remtk-prediction — API pública de librería (in-process).
 *
 * Uso:
 * ```ts
 * import { createSystem, loadConfig } from "remtk-prediction";
 *
 * const system = await createSystem(loadConfig());
 * const result = await system.orchestrator.predict({
 *   sessionId: "s1",
 *   tenant: "t1",
 *   text: "envía un correo al equipo",
 * });
 * console.log(result.tools.map((t) => t.name));
 * ```
 *
 * Para arrancar los servidores HTTP usa `node dist/src/server.js`
 * (o importa `./server`).
 */

// Sistema completo (in-process, sin abrir puertos).
export { createSystem, type System } from "./main";

// Configuración.
export { loadConfig, type AppConfig } from "./config";

// Servicios núcleo reutilizables.
export { EmbeddingEngine, cosine, normalizeL2, type EmbedResult } from "./embedding/embedding-engine";
export { QdrantService } from "./qdrant/qdrant-service";
export { PredictionOrchestrator } from "./predict/orchestrator";
export { TurnClassifier, type TurnType } from "./predict/turn-classifier";
export { KeywordService, type KeywordReduceResult } from "./predict/keyword-service";
export { RerankService, adaptiveThreshold, estimateComplexity, type RerankResult } from "./predict/rerank";
export { ConfirmationCache } from "./predict/confirm-cache";
export { Debugger } from "./predict/debugger";
export { extractQueryKeywords, toolKeywords, normalizeToken } from "./predict/keywords";

// Tipos del contrato.
export type {
  ToolDefinition,
  ToolComplexity,
  ModelSize,
  ScoredTool,
  PredictionInput,
  PredictionResult,
  ChatMessage,
  MemoryDefinition,
  MemoryCandidate,
  MemoryPredictionInput,
  MemoryPredictionResult,
  Trace,
} from "./types";
