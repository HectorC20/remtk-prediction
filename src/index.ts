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
export { EmbeddingEngineService } from "./embedding/embedding.service";
export { QdrantService } from "./qdrant/qdrant-service";
export { PredictionOrchestrator } from "./predict/orchestrator";
export { TurnClassifier} from "./predict/turn-classifier";
export { KeywordService, type KeywordReduceResult } from "./predict/services/keyword.service";
export { ToolGraphCacheService } from "./predict/services/graph-cache.service";
export { SessionStateCacheService } from "./predict/services/session-state-cache.service";
export { RerankService, adaptiveThreshold, estimateComplexity, topologicalSort, type RerankResult, type GraphRerankResult } from "./predict/services/rerank.service";
export { ConfirmationCache } from "./predict/services/confirm-cache.service";
export { Debugger } from "./predict/helper/debugger.helper";
export { extractQueryKeywords, toolKeywords, normalizeToken } from "./predict/keywords";
export { MemoryCandidate, MemoryDefinition } from "src/shared/interfaces/index";
// Tipos del contrato.
export type {
  ToolDefinition,
  ToolComplexity,
  ScoredTool,
  PredictionInput,
  PredictionResult,
  Trace,
} from "./shared/interfaces/index";
export type {
  EdgeType,
  GraphEdge,
  ToolGraphNode,
  TenantToolGraph,
  GraphPredictionResult,
} from "./shared/interfaces/index";
export type { ModelSize } from "./shared/constants/predict/version.constants";
