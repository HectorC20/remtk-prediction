/**
 * PredictionOrchestrator: pipeline unificado de predicción, migrado de
 * prediction_orchestrator.go + memory_orchestrator.go del server Go.
 *
 * Flujo /predict:  clasificar turno → early-exit (confirmación cacheada) →
 *                  recall Qdrant → re-rank por embeddings e5 → umbral adaptativo.
 * Flujo /memory/predict: embed → topic shift → recall Qdrant → re-rank cosine.
 */
import { EmbeddingEngineService } from "../embedding/embedding.service";
import type { AppConfig } from "../config";
import { log, warn } from "../logger";
import type { QdrantService } from "../qdrant/qdrant-service";
import type {
  PredictionInput,
  Trace,
  ToolDefinition,
} from "../shared/interfaces/domain.interface";
import type { GraphPredictionResult } from "../shared/interfaces/graph.interface";
import { ConfirmationCache } from "./services/confirm-cache.service";
import { Debugger } from "./helper/debugger.helper";
import { ToolGraphCacheService } from "./services/graph-cache.service";
import { KeywordService } from "./services/keyword.service";
import { RerankService } from "./services/rerank.service";
import { SessionStateCacheService } from "./services/session-state-cache.service";
import { TurnClassifier} from "./turn-classifier";
import { ChatMessage, TopicState, MemoryDefinition, MemoryPredictionInput, MemoryPredictionResult } from "src/shared/interfaces/index";
import { TOPIC_SHIFT_THRESHOLD, MAX_HISTORY_MESSAGES, MAX_MESSAGE_CHARS } from "src/shared/constants/predict/index";
import { TurnType } from "src/shared/dictionary/turn.dictionary";

export class PredictionOrchestrator {
  private readonly topics = new Map<string, TopicState>();
  private readonly memoryCache = new Map<string, Float32Array>();

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly qdrant: QdrantService,
    private readonly classifier: TurnClassifier,
    private readonly rerank: RerankService,
    private readonly confirmCache: ConfirmationCache,
    private readonly keywords: KeywordService,
    private readonly graphCache: ToolGraphCacheService,
    private readonly sessionState: SessionStateCacheService,
    private readonly debugger_: Debugger,
    private readonly config: AppConfig,
  ) {}

  /** POST /tools: registra el catálogo, indexa Qdrant y calcula embeddings. */
  async registerTools(tenant: string, tools: ToolDefinition[]): Promise<{ indexed: number }> {
    await this.qdrant.indexTools(tenant, tools);
    const { recomputed, cached } = await this.keywords.upsertTools(tenant, tools);
    await this.graphCache.buildTenantGraph(tenant, tools);
    this.debugger_.bump({ embeddingsRecomputed: recomputed, embeddingsCached: cached });
    this.debugger_.setTenants(this.qdrant.countTools(tenant) > 0 ? Math.max(1, this.debugger_.snapshot().stats.tenants) : 0);
    log(`[tools] tenant=${tenant} indexados=${tools.length} embeddings recomputed=${recomputed} cached=${cached}`);
    return { indexed: tools.length };
  }

  countTools(tenant: string): number {
    return this.qdrant.countTools(tenant);
  }

  /** POST /predict */
  async predict(input: PredictionInput): Promise<GraphPredictionResult> {
    const start = Date.now();
    const trace: Trace = {
      sessionId: input.sessionId,
      tenant: input.tenant,
      turnType: TurnType.NewQuery,
      confirmHit: false,
      recall: 0,
      degraded: false,
      inputTokens: (input.text ?? "").length,
      outputTools: 0,
      modelSize: "hash",
      embeddingsRecomputed: 0,
      embeddingsCached: 0,
      complexity: "simple",
      latencyMs: 0,
      timestamp: new Date().toISOString(),
    };

    const text = (input.text ?? "").trim();
    if (text === "") {
      return {
        tools: [],
        complexity: "simple",
        modelSize: "hash",
        rankedScores: [],
        graph: { nodes: [], edges: [], executionOrder: [] },
      };
    }

    const turn = await this.classifier.classify(text);
    trace.turnType = turn;
    log(`[pipeline] predict session=${input.sessionId} tenant=${input.tenant} turn=${turn}`);

    // Early exit: confirmación vacía con plan previo cacheado.
    if (turn === TurnType.ConfirmationEmpty) {
      this.debugger_.bump({ confirmGets: 1 });
      const cached = this.confirmCache.get(input.sessionId);
      if (cached) {
        trace.confirmHit = true;
        this.debugger_.bump({ confirmHits: 1 });
        log(`[pipeline] early-exit: confirmación vacía con plan cacheado`);
        this.finalize(trace, start);
        return cached;
      }
    }

    // En turnos de confirmación (con matiz) o source=agent se predice sobre el plan previo.
    // En consulta nueva se incorporan los mensajes previos para dar contexto.
    const promptText =
      turn === TurnType.NewQuery ? this.withHistory(text, input.history) : input.priorPlan ?? text;

    // Estado latente de sesión (z_t): proyección suavizada del prompt entrante.
    let zt: Float32Array | undefined;
    let sessionModel = "hash";
    try {
      const s = await this.sessionState.resolve(input.sessionId, promptText);
      zt = s.zt;
      sessionModel = s.model;
    } catch (err) {
      warn(`[pipeline] estado de sesión degradado: ${String((err as Error)?.message ?? err)}`);
    }

    const graph = this.graphCache.get(input.tenant);

    // Ruta topológica: el grafo del tenant está precomputado en POST /tools.
    if (graph && zt) {
      const lexicalScores = await this.lexicalScores(input.tenant, promptText);
      const result = this.rerank.graphFilter({
        zt,
        graph,
        edges: this.graphCache.edges(input.tenant),
        lexicalScores,
        catalog: this.qdrant.catalog(input.tenant),
        modelSize: sessionModel,
      });
      trace.recall = graph.nodes.size;
      trace.modelSize = result.modelSize;
      trace.complexity = result.complexity;
      trace.outputTools = result.tools.length;
      trace.embeddingsCached = graph.nodes.size;
      this.debugger_.bump({ embeddingsCached: graph.nodes.size });

      if (turn === TurnType.NewQuery) {
        this.confirmCache.set(input.sessionId, result);
        this.debugger_.bump({ confirmSets: 1 });
      }

      this.finalize(trace, start);
      return result;
    }

    // Fallback plano (sin grafo): reduce por keywords + fusión BM25.
    let reduced: { name: string; score: number }[] = [];
    let modelSize = "hash";
    let recomputed = 0;
    let cached = 0;
    try {
      const r = await this.keywords.reduce(input.tenant, promptText, this.qdrant.catalog(input.tenant));
      reduced = r.candidates;
      modelSize = r.modelSize;
      recomputed = r.recomputed;
      cached = r.cached;
    } catch (err) {
      warn(`[pipeline] keyword reduce degradado: ${String((err as Error)?.message ?? err)}`);
    }
    trace.degraded = reduced.length === 0;
    if (trace.degraded) {
      reduced = this.qdrant.catalog(input.tenant).all().map((t) => ({ name: t.name, score: 0 }));
    }
    trace.recall = reduced.length;
    log(`[pipeline] reduce candidates=${reduced.length} degraded=${trace.degraded}`);

    const lexicalScores = await this.lexicalScores(input.tenant, promptText);
    const base = this.rerank.filter(reduced, lexicalScores, this.qdrant.catalog(input.tenant), modelSize);
    const order = base.tools.map((t) => t.name);
    const result: GraphPredictionResult = {
      ...base,
      graph: { nodes: order, edges: [], executionOrder: order },
    };
    trace.embeddingsRecomputed = recomputed;
    trace.embeddingsCached = cached;
    this.debugger_.bump({ embeddingsRecomputed: recomputed, embeddingsCached: cached });
    trace.modelSize = result.modelSize;
    trace.complexity = result.complexity;
    trace.outputTools = result.tools.length;

    // Cachea el plan para confirmaciones posteriores de la sesión.
    if (turn === TurnType.NewQuery) {
      this.confirmCache.set(input.sessionId, result);
      this.debugger_.bump({ confirmSets: 1 });
    }

    this.finalize(trace, start);
    return result;
  }

  /** Confirmación léxica BM25 (normalizada 0..1) para reforzar el score semántico. */
  private async lexicalScores(tenant: string, promptText: string): Promise<Map<string, number>> {
    const scores = new Map<string, number>();
    try {
      const lexical = await this.qdrant.retrieveTools(tenant, promptText, this.config.recallLimit);
      const maxLex = lexical.reduce((m, l) => Math.max(m, l.score), 0);
      for (const l of lexical) {
        scores.set(l.name, maxLex > 0 ? l.score / maxLex : 0);
      }
    } catch (err) {
      warn(`[pipeline] confirm léxica degradada: ${String((err as Error)?.message ?? err)}`);
    }
    return scores;
  }

  /** POST /memory/predict: memorias contextuales + detección de cambio de tema. */
  async predictMemory(input: MemoryPredictionInput): Promise<MemoryPredictionResult> {
    const text = (input.text ?? "").trim();
    if (text === "") {
      return { memories: [], topicShift: false, topicScore: 0, modelSize: "hash", rankedScores: [] };
    }

    const promptEmb = await this.engine.embedQuery(text);
    log(`[memory] embedded model=${promptEmb.model} dim=${promptEmb.dim}`);

    // Detección de cambio de tema por cosine contra el estado previo de la sesión.
    let topicScore = 0;
    let shift = false;
    const prev = this.topics.get(input.sessionId);
    if (prev) {
      topicScore = EmbeddingEngineService.cosine(promptEmb.embedding, prev.embedding);
      shift = topicScore < TOPIC_SHIFT_THRESHOLD;
    }
    this.topics.set(input.sessionId, this.updateTopic(prev, promptEmb.embedding, shift));

    // Recall BM25 directo en Qdrant (degradación suave si no responde).
    let candidates: { memory: MemoryDefinition; retrievalScore: number }[] = [];
    try {
      candidates = await this.qdrant.searchMemories(input.tenant, text, input.limit || 8);
    } catch (err) {
      log(`[memory] recall degradado (Qdrant no disponible): ${String((err as Error)?.message ?? err)}`);
      candidates = [];
    }

    // Re-rank: cosine entre prompt y cada memoria (embeddings cacheados por content).
    const scored = [];
    for (const c of candidates) {
      const emb = await this.memoryEmbedding(c.memory.content);
      if (!emb) {
        scored.push({ memory: { ...c.memory, score: c.retrievalScore }, score: c.retrievalScore });
        continue;
      }
      const s = EmbeddingEngineService.cosine(promptEmb.embedding, emb);
      scored.push({ memory: { ...c.memory, score: s }, score: s });
    }
    scored.sort((a, b) => b.score - a.score);

    const result: MemoryPredictionResult = {
      memories: scored.map((s) => s.memory),
      topicShift: shift,
      topicScore,
      modelSize: promptEmb.model,
      rankedScores: scored.map((s) => s.score),
    };
    log(`[memory] memories=${result.memories.length} topicShift=${shift} topicScore=${topicScore.toFixed(3)}`);
    return result;
  }

  /**
   * Incorpora los mensajes previos al texto actual para dar contexto a la
   * extracción de keywords y al match (capa 1+2). Recorta a los últimos mensajes.
   */
  private withHistory(text: string, history?: ChatMessage[]): string {
    if (!history || history.length === 0) return text;
    const recent = history
      .slice(-MAX_HISTORY_MESSAGES)
      .map((m) => (m?.content ?? "").trim())
      .filter((c) => c.length > 0)
      .map((c) => (c.length > MAX_MESSAGE_CHARS ? c.slice(0, MAX_MESSAGE_CHARS) : c));
    if (recent.length === 0) return text;
    return [...recent, text].join("\n");
  }

  private async memoryEmbedding(content: string): Promise<Float32Array | undefined> {
    if (!content) return undefined;
    const cached = this.memoryCache.get(content);
    if (cached) return cached;
    const res = await this.engine.embedPassage(content, undefined, { high: true });
    this.memoryCache.set(content, res.embedding);
    return res.embedding;
  }

  private updateTopic(prev: TopicState | undefined, emb: Float32Array, shift: boolean): TopicState {
    if (!prev) return { embedding: emb, lastScore: 0 };
    if (shift) return { embedding: emb, lastScore: 0 };
    // Sin cambio de tema: suaviza el estado con el nuevo embedding (blend 70/30).
    const blended = new Float32Array(emb.length);
    for (let i = 0; i < emb.length; i++) {
      blended[i] = prev.embedding[i] * 0.7 + emb[i] * 0.3;
    }
    return { embedding: EmbeddingEngineService.normalizeL2(blended), lastScore: 0 };
  }

  private finalize(trace: Trace, start: number): void {
    trace.latencyMs = Date.now() - start;
    this.debugger_.addTrace(trace);
    log(
      `[pipeline] predict session=${trace.sessionId} tenant=${trace.tenant} turn=${trace.turnType} ` +
        `confirm_hit=${trace.confirmHit} recall=${trace.recall} degraded=${trace.degraded} ` +
        `in=${trace.inputTokens} out=${trace.outputTools} model=${trace.modelSize} ` +
        `emb_compute=${trace.embeddingsRecomputed} emb_cached=${trace.embeddingsCached} ` +
        `complexity=${trace.complexity} ${trace.latencyMs}ms`,
    );
  }
}
