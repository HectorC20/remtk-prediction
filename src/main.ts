/**
 * remtk-prediction: Tool Prediction Server unificado en un solo proceso.
 *
 *  ┌────────────────────────────────────────────────────────────┐
 *  │                      PREDICTION (TS)                       │
 *  │                                                            │
 *  │  Puerto 6776  API de Predicción                            │
 *  │    /predict, /tools, /memory/predict, /health, /debug      │
 *  │                                                            │
 *  │  Puerto 6777  API de Modelos (e5-small)                    │
 *  │    POST /embed, GET /models                                │
 *  │                                                            │
 *  │  Puerto 6775  API de Qdrant (BM25, keywording, indexado)   │
 *  │    /tools/upsert, /tools/search, /memories/search, /status │
 *  └────────────────────────────────────────────────────────────┘
 */
import type { AppConfig } from "./config";
import { EmbeddingEngineService } from "./embedding/embedding-engine";
import { ConfirmationCache } from "./predict/services/confirm-cache.service";
import { Debugger } from "./predict/helper/debugger.helper";
import { KeywordService } from "./predict/services/keyword.service";
import { ToolGraphCacheService } from "./predict/services/graph-cache.service";
import { SessionStateCacheService } from "./predict/services/session-state-cache.service";
import { RerankService } from "./predict/services/rerank.service";
import { TurnClassifier } from "./predict/turn-classifier";
import { QdrantService } from "./qdrant/qdrant-service";
import { createOrchestrator } from "./shared/factory/orchestrator.factory";
import type { System } from "./shared/interfaces/system.interface";

// Contrato público del sistema: reexportado para `index.ts` y consumidores
// externos (p. ej. tests) que importan `{ createSystem, type System }`.
export type { System };

/** Construye el sistema completo (in-process, sin escuchar puertos). */
export async function createSystem(cfg: AppConfig): Promise<System> {
  const engine = new EmbeddingEngineService(cfg);
  engine.warmup();

  const qdrant = new QdrantService(cfg);
  if (qdrant.client.enabled) {
    await qdrant.ensureCollections();
  }

  const classifier = new TurnClassifier(engine);
  const confirmCache = new ConfirmationCache();
  const keywords = new KeywordService(engine, cfg.keywordTopK);
  const graphCache = new ToolGraphCacheService(engine);
  const sessionState = new SessionStateCacheService(engine);
  const debugger_ = new Debugger();
  const rerank = new RerankService(cfg);

  const orchestrator = createOrchestrator({
    engine,
    qdrant,
    classifier,
    rerank,
    confirmCache,
    keywords,
    graphCache,
    sessionState,
    debugger_,
    cfg,
  });

  return {
    engine,
    qdrant,
    classifier,
    keywords,
    graphCache,
    sessionState,
    confirmCache,
    debugger: debugger_,
    orchestrator,
  };
}