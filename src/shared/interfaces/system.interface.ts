import type { EmbeddingEngineService } from "../../embedding/embedding-engine";
import type { ConfirmationCache } from "../../predict/services/confirm-cache.service";
import type { Debugger } from "../../predict/helper/debugger.helper";
import type { KeywordService } from "../../predict/services/keyword.service";
import type { ToolGraphCacheService } from "../../predict/services/graph-cache.service";
import type { SessionStateCacheService } from "../../predict/services/session-state-cache.service";
import type { TurnClassifier } from "../../predict/turn-classifier";
import type { QdrantService } from "../../qdrant/qdrant-service";
import { IPredictionOrchestrator } from "./orchestrator.interface";

export interface System {
  engine: EmbeddingEngineService;
  qdrant: QdrantService;
  classifier: TurnClassifier;
  keywords: KeywordService;
  graphCache: ToolGraphCacheService;
  sessionState: SessionStateCacheService;
  confirmCache: ConfirmationCache;
  debugger: Debugger;
  orchestrator: IPredictionOrchestrator; // <-- Cambia PredictionOrchestrator por IPredictionOrchestrator
}