import type { EmbeddingEngineService } from "../../embedding/embedding.service";
import type { ConfirmationCache } from "../../predict/services/confirm-cache.service";
import type { Debugger } from "../../predict/helper/debugger.helper";
import type { KeywordService } from "../../predict/services/keyword.service";
import type { ToolGraphCacheService } from "../../predict/services/graph-cache.service";
import type { SessionStateCacheService } from "../../predict/services/session-state-cache.service";
import type { TurnClassifier } from "../../predict/turn-classifier";
import type { QdrantService } from "../../qdrant/qdrant-service";
import type { LexicalProfileService } from "../../predict/services/lexical-profile.service";
import type { SkillMemoryService } from "../../predict/services/skill-memory.service";
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
  /** Perfil léxico aprendido por canal (cuarta señal del ranking). */
  lexical: LexicalProfileService;
  orchestrator: IPredictionOrchestrator;
  /** Memoria de Habilidades y Grafo de Entidades Paramétricas */
  skillMemory?: SkillMemoryService;
}