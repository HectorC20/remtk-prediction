import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { TurnClassifier } from "src/predict/turn-classifier";
import { QdrantService } from "src/qdrant/qdrant-service";
import { KeywordService } from "src/predict/services/keyword.service";
import { ToolGraphCacheService } from "src/predict/services/graph-cache.service";
import { SessionStateCacheService } from "src/predict/services/session-state-cache.service";
import { Debugger } from "src/predict/helper/debugger.helper";
import { PredictionOrchestrator } from "src/predict/orchestrator";
import { IPredictionOrchestrator } from "../interfaces/orchestrator.interface";
import { RerankService } from "src/predict/services/rerank.service";
import { ConfirmationCache } from "src/predict/services/confirm-cache.service";
import { AppConfig } from "src/config";

export interface OrchestratorDependencies {
  engine: EmbeddingEngineService;
  qdrant: QdrantService;
  classifier: TurnClassifier;
  rerank: RerankService;
  confirmCache: ConfirmationCache;
  keywords: KeywordService;
  graphCache: ToolGraphCacheService;
  sessionState: SessionStateCacheService;
  debugger_: Debugger;
  cfg: AppConfig;
}

/** Construye y devuelve la instancia del orquestador bajo su interfaz */
export function createOrchestrator(deps: OrchestratorDependencies): IPredictionOrchestrator {
  return new PredictionOrchestrator(
    deps.engine,
    deps.qdrant,
    deps.classifier,
    deps.rerank,
    deps.confirmCache,
    deps.keywords,
    deps.graphCache,
    deps.sessionState,
    deps.debugger_,
    deps.cfg,
  );
}