/**
 * remtk-prediction: Tool Prediction Server unificado en un solo proceso.
 *
 *  ┌────────────────────────────────────────────────────────────┐
 *  │                       PREDICTION (TS)                      │
 *  │                                                            │
 *  │  Puerto 6776  API de Predicción                            │
 *  │    /predict, /tools, /memory/predict, /health, /debug      │
 *  │                                                            │
 *  │  Puerto 6777  API de Modelos (e5-large + e5-small)         │
 *  │    POST /embed, GET /models                                │
 *  │                                                            │
 *  │  Puerto 6778  API de Qdrant (BM25, keywording, indexado)   │
 *  │    /tools/upsert, /tools/search, /memories/search, /status │
 *  └────────────────────────────────────────────────────────────┘
 */
import type { AppConfig } from "./config";
import { EmbeddingEngine } from "./embedding/embedding-engine";
import { ConfirmationCache } from "./predict/confirm-cache";
import { Debugger } from "./predict/debugger";
import { KeywordService } from "./predict/keyword-service";
import { PredictionOrchestrator } from "./predict/orchestrator";
import { RerankService } from "./predict/rerank";
import { TurnClassifier } from "./predict/turn-classifier";
import { QdrantService } from "./qdrant/qdrant-service";

export interface System {
  engine: EmbeddingEngine;
  qdrant: QdrantService;
  classifier: TurnClassifier;
  keywords: KeywordService;
  confirmCache: ConfirmationCache;
  debugger: Debugger;
  orchestrator: PredictionOrchestrator;
}

/** Construye el sistema completo (in-process, sin escuchar puertos). */
export async function createSystem(cfg: AppConfig): Promise<System> {
  const engine = new EmbeddingEngine(cfg);
  engine.warmup();

  const qdrant = new QdrantService(cfg);
  if (qdrant.client.enabled) {
    await qdrant.ensureCollections();
  }

  const classifier = new TurnClassifier(engine);
  const confirmCache = new ConfirmationCache();
  const keywords = new KeywordService(engine, cfg.keywordTopK);
  const debugger_ = new Debugger();
  const rerank = new RerankService(cfg);
  const orchestrator = new PredictionOrchestrator(
    engine,
    qdrant,
    classifier,
    rerank,
    confirmCache,
    keywords,
    debugger_,
    cfg,
  );

  return {
    engine,
    qdrant,
    classifier,
    keywords,
    confirmCache,
    debugger: debugger_,
    orchestrator,
  };
}

