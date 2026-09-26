/**
 * JuicioService: orquesta la capa de juicio (docs/juicio.md) en dos momentos
 * del pipeline de /predict:
 *
 *   pre()   — antes del recall: (1) NLI de estado condicionado (rechazo /
 *             confirmación de la propuesta pendiente) y (3) puerta de
 *             coherencia semántica (λ) para aislar el turno del historial.
 *
 *   post()  — después del rerank: (2) abstención __NOOP__ + energía,
 *             (5) MaxSim token-level, (6) especificidad + indexación dual y
 *             (4) cross-encoder opcional, re-ordenando el top-K final.
 *
 * Todo el juicio se desactiva en cascada: sin `JUICIO_ENABLED` o con el motor
 * de embeddings degradado (hash), pre/post son pasa-through y el pipeline
 * queda exactamente como antes.
 */
import type { AppConfig } from "src/config";
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { log } from "src/logger";
import type { ChatMessage, GraphPredictionResult, ToolDefinition } from "src/shared/interfaces";
import type { JuicioContext, JuicioPostDiag } from "src/shared/interfaces/juicio.interface";
import { SessionStateCacheService } from "src/predict/services/session-state-cache.service";
import { AbstencionService } from "./services/abstencion.service";
import { CrossEncoderService } from "./services/cross-encoder.service";
import { EspecificidadService } from "./services/especificidad.service";
import { EstadoNliService } from "./services/estado-nli.service";
import { MaxSimService } from "./services/maxsim.service";
import { PuertaContextoService } from "./services/puerta-contexto.service";

export interface JuicioPreInput {
  scopeKey?: string;
  sessionId: string;
  text: string;
  source?: "human" | "agent";
  history?: ChatMessage[];
  /** Plan cacheado de la sesión (propuesta pendiente de confirmación). */
  cachedPlan?: GraphPredictionResult;
}

export class JuicioService {
  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly sessionState: SessionStateCacheService,
    private readonly nli: EstadoNliService,
    private readonly abstencion: AbstencionService,
    private readonly puerta: PuertaContextoService,
    private readonly maxsim: MaxSimService,
    private readonly especificidad: EspecificidadService,
    private readonly reranker: CrossEncoderService,
    private readonly config: AppConfig,
  ) {}

  /** Invalida las cachés del scope al re-registrar herramientas (POST /tools). */
  clearScope(scopeKey: string): void {
    this.maxsim.clearScope(scopeKey);
    this.especificidad.clearScope(scopeKey);
  }

  /** true si la capa puede operar este turno (flag + modelo real disponible). */
  private get operative(): boolean {
    return this.config.juicioEnabled && this.engine.isReady("small");
  }

  /**
   * Pre-pipeline. Devuelve el contexto de juicio del turno: veredicto NLI de
   * estado, atención por saliencia sobre tramos y λ de la puerta de coherencia.
   */
  async pre(input: JuicioPreInput): Promise<JuicioContext> {
    const ctx: JuicioContext = {
      estado: "neutral",
      historyGated: false,
      resolved: false,
    };
    if (!this.operative) return ctx;

    const res = await this.engine.embedQuery(input.text, "small", { high: true });
    if (res.model === "hash") return ctx;
    ctx.uText = res.embedding;

    // (1) NLI de estado: solo turnos humanos y solo si hay propuesta pendiente.
    if (input.source === "human") {
      const premise = this.pendingPremise(input);
      if (premise) {
        ctx.estado = await this.nli.verdict(input.text, premise);
        if (ctx.estado === "reject") {
          ctx.resolved = true;
          ctx.resolvedBy = "nli_reject";
          log(`[juicio] veredicto REJECT session=${input.sessionId} → propuesta descartada`);
        } else if (ctx.estado === "confirm" && input.cachedPlan) {
          ctx.resolved = true;
          ctx.resolvedBy = "nli_confirm";
          log(`[juicio] veredicto CONFIRM session=${input.sessionId} → plan cacheado`);
        }
      }
    }
    if (ctx.resolved) return ctx;

    // (2b) Atención por saliencia sobre tramos estructurales (textos extensos con relleno).
    if (input.scopeKey) {
      await this.focusSalientSpans(input.scopeKey, input.text, ctx);
    }

    // (3) Puerta de coherencia: λ entre el texto actual (enfocado) y el estado previo,
    //     reforzada con coherencia funcional sobre la variedad de herramientas del grafo.
    const prev = this.sessionState.peek(input.sessionId);
    ctx.gateLambda = this.puerta.lambda(ctx.uText, prev);
    if (prev && input.scopeKey && !ctx.allSpansNoop) {
      const [currProj, prevProj, noop] = await Promise.all([
        this.especificidad.projectManifold(input.scopeKey, ctx.uText),
        this.especificidad.projectManifold(input.scopeKey, prev),
        this.abstencion.noopScore(ctx.uText),
      ]);
      if (currProj && prevProj && currProj.topTool !== prevProj.topTool) {
        const scoreCurrOnPrev = currProj.scores.get(prevProj.topTool) ?? 0;
        const shiftMargin = currProj.topScore - scoreCurrOnPrev;
        const isFunctionalPivot =
          (currProj.topScore > noop - 0.01 && shiftMargin >= 0.02 && currProj.prominence >= 0.035) ||
          (shiftMargin >= 0.04 && currProj.prominence >= 0.035 && currProj.topScore >= 0.84);
        if (isFunctionalPivot) {
          ctx.gateLambda = 0.05;
        }
      }
    }
    ctx.historyGated = this.puerta.gatesHistory(ctx.gateLambda, prev !== undefined);
    if (ctx.historyGated) {
      log(`[juicio] puerta λ=${ctx.gateLambda.toFixed(3)} → turno aislado del historial`);
    }
    return ctx;
  }

  /**
   * Filtra el relleno en textos multi-oración proyectando cada tramo único
   * contra el grafo de herramientas y el prototipo __NOOP__.
   */
  private async focusSalientSpans(
    scopeKey: string,
    text: string,
    ctx: JuicioContext,
  ): Promise<void> {
    // Divide en oraciones completas evitando partir comas internas o puntos dentro de nombres de archivo/versiones
    const rawSpans = text
      .split(/(?:[\n…]+|(?<=[.?!;])\s+)/)
      .map((s) => s.trim())
      .filter((s) => s.length >= 18);
    if (rawSpans.length <= 1) return;

    // Deduplica tramos idénticos para evaluar cada oración única una sola vez.
    const uniqueTextByKey = new Map<string, string>();
    for (const s of rawSpans) {
      const k = s.toLowerCase();
      if (!uniqueTextByKey.has(k)) uniqueTextByKey.set(k, s);
    }

    interface SpanEval {
      text: string;
      emb: Float32Array;
      topScore: number;
      prominence: number;
      noop: number;
      actionable: boolean;
      salience: number;
    }

    const evalByKey = new Map<string, SpanEval>();
    for (const [k, spanText] of uniqueTextByKey.entries()) {
      const embRes = await this.engine.embedQuery(spanText, "small", { high: true });
      if (embRes.model === "hash") return;
      const proj = await this.especificidad.projectManifold(scopeKey, embRes.embedding);
      if (!proj) return;
      const noop = await this.abstencion.noopScore(embRes.embedding);
      const actionable =
        proj.topScore > noop && proj.topScore >= 0.845 && proj.prominence >= 0.032;
      const salience = proj.topScore + proj.prominence - 0.5 * noop;
      evalByKey.set(k, {
        text: spanText,
        emb: embRes.embedding,
        topScore: proj.topScore,
        prominence: proj.prominence,
        noop,
        actionable,
        salience,
      });
    }

    const evals = [...evalByKey.values()];
    const actionableEvals = evals.filter((e) => e.actionable);
    if (actionableEvals.length === 0) {
      ctx.allSpansNoop = true;
      return;
    }

    const maxSalience = Math.max(...actionableEvals.map((e) => e.salience));
    const kept: SpanEval[] = [];
    const seenKept = new Set<string>();
    for (const s of rawSpans) {
      const k = s.toLowerCase();
      if (seenKept.has(k)) continue;
      const ev = evalByKey.get(k);
      if (ev && ev.actionable && ev.salience >= maxSalience - 0.025) {
        seenKept.add(k);
        kept.push(ev);
      }
    }

    if (kept.length > 0 && kept.length < rawSpans.length) {
      ctx.focusedText = kept.map((k) => k.text).join(". ");
      if (kept.length === 1) {
        ctx.uText = kept[0].emb;
      } else {
        const focusedEmb = await this.engine.embedQuery(ctx.focusedText, "small", { high: true });
        if (focusedEmb.model !== "hash") ctx.uText = focusedEmb.embedding;
      }
      log(
        `[juicio] atención tramos=${rawSpans.length}->${kept.length} foco=${JSON.stringify(ctx.focusedText)}`,
      );
    }
  }

  /**
   * Post-pipeline: abstención + re-rank del top-K por señales de juicio.
   * `result` se clona superficialmente; el original del rerank no se toca.
   */
  async post(params: {
    scopeKey: string;
    text: string;
    ctx: JuicioContext;
    result: GraphPredictionResult;
    exclude?: string[];
  }): Promise<{ result: GraphPredictionResult; diag: JuicioPostDiag }> {
    const { scopeKey, text, ctx, result } = params;
    const diag: JuicioPostDiag = {
      abstained: false,
      noopScore: 0,
      energy: 0,
      maxsimApplied: false,
      rerankerApplied: false,
      specificityApplied: false,
    };
    if (!this.operative || result.tools.length === 0) return { result, diag };

    // (2) Abstención: prototipo __NOOP__, energía o ausencia de pico funcional en discurso multi-tramo.
    const verdict = await this.abstencion.shouldAbstain(ctx.uText, result.rankedScores);
    if (
      ctx.allSpansNoop &&
      !verdict.abstain &&
      (result.rankedScores[0] ?? 0) < this.config.adaptiveMinScore
    ) {
      verdict.abstain = true;
      verdict.reason = "noop";
    }
    diag.noopScore = verdict.noop;
    diag.energy = verdict.energy;
    log(
      `[juicio] señales noop=${verdict.noop.toFixed(3)} energy=${verdict.energy.toFixed(3)} ` +
        `top=${(result.rankedScores[0] ?? 0).toFixed(3)}`,
    );
    if (verdict.abstain) {
      diag.abstained = true;
      diag.abstainReason = verdict.reason;
      log(
        `[juicio] abstención (${verdict.reason}) noop=${verdict.noop.toFixed(3)} ` +
          `energy=${verdict.energy.toFixed(3)} top=${(result.rankedScores[0] ?? 0).toFixed(3)} → ∅`,
      );
      return { result: abstainedResult(result, verdict.reason ?? "noop"), diag };
    }

    // (4+5+6) Re-rank del top-K: MaxSim + cross-encoder + dual + especificidad.
    const effectiveText = ctx.focusedText ?? text;
    const k = Math.min(this.config.juicioPostTopK, result.tools.length);
    const top = result.tools.slice(0, k);
    const w = this.config.juicioMaxsimWeight;
    const alpha = this.config.juicioRerankerAlpha;

    const maxsimScores = await this.maxsim.score(scopeKey, effectiveText, top);
    diag.maxsimApplied = maxsimScores.size > 0;
    const rerankerOn = await this.reranker.ready();
    diag.rerankerApplied = rerankerOn;

    const adjusted: { tool: ToolDefinition; score: number }[] = [];
    for (let i = 0; i < top.length; i++) {
      const tool = top[i];
      let score = result.rankedScores[i] ?? 0;
      const ms = maxsimScores.get(tool.name);
      if (ms !== undefined) score = (1 - w) * score + w * ms;
      if (rerankerOn) {
        const ce = await this.reranker.score(effectiveText, `${tool.intentSummary}. ${tool.description}`);
        if (ce !== undefined) score = (1 - alpha) * score + alpha * ce;
      }
      const problem = await this.especificidad.problemScore(scopeKey, tool, ctx.uText);
      if (problem > score + this.config.juicioProblemMargin) score = problem;
      const penalty = this.especificidad.penalty(scopeKey, tool.name);
      if (penalty < 1) diag.specificityApplied = true;
      adjusted.push({ tool, score: score * penalty });
    }
    adjusted.sort((a, b) => b.score - a.score);

    // (5b) Alineación por sub-ventanas y anclas token-a-herramienta (ColBERT doblemente
    //      centrado) para consultas multi-intención (≥ 2 sub-ventanas ganadoras).
    const winPeaks = await this.especificidad.subWindowPeaks(scopeKey, effectiveText);
    if (winPeaks.size >= 2 && adjusted.length > 0) {
      const allTools = this.especificidad.catalogTools(scopeKey);
      const tokenAnchors = await this.maxsim.anchorTools(scopeKey, effectiveText, allTools);
      for (const [toolName, anchor] of tokenAnchors.entries()) {
        if (!winPeaks.has(toolName)) {
          winPeaks.set(toolName, {
            tool: anchor.tool,
            score: anchor.score,
            windowIdx: 100 + anchor.tokenIdx,
          });
        }
      }
      const top1Score = adjusted[0].score;
      const excludedLower = new Set((params.exclude ?? []).map((e) => e.toLowerCase()));
      const winEntries = [...winPeaks.entries()].sort((a, b) => a[1].windowIdx - b[1].windowIdx);
      for (let idx = 0; idx < winEntries.length; idx++) {
        const [toolName, peak] = winEntries[idx];
        if (excludedLower.has(toolName.toLowerCase())) continue;
        // Las herramientas ganadoras de su propia sub-ventana se promueven justo
        // debajo de la líder global, preservando `adjusted[0]` intacto.
        const targetScore = Math.max(peak.score, top1Score - 0.003 * (idx + 1));
        const cappedScore = Math.min(top1Score - 0.001 * (idx + 1), targetScore);
        const existing = adjusted.find((a) => a.tool.name === toolName);
        if (existing) {
          if (existing !== adjusted[0] && cappedScore > existing.score) {
            existing.score = cappedScore;
          }
        } else if (adjusted.length < this.config.maxOutputTools) {
          adjusted.push({ tool: peak.tool, score: cappedScore });
        }
      }
      adjusted.sort((a, b) => b.score - a.score);
    }

    const adjustedNames = new Set(adjusted.map((a) => a.tool.name));
    const rest: ToolDefinition[] = [];
    const restScores: number[] = [];
    for (let i = k; i < result.tools.length; i++) {
      if (!adjustedNames.has(result.tools[i].name)) {
        rest.push(result.tools[i]);
        restScores.push(result.rankedScores[i] ?? 0);
      }
    }
    const tools = [...adjusted.map((a) => a.tool), ...rest].slice(0, this.config.maxOutputTools);
    const rankedScores = [...adjusted.map((a) => a.score), ...restScores].slice(
      0,
      this.config.maxOutputTools,
    );
    const order = tools.map((t) => t.name);

    const judged: GraphPredictionResult = {
      ...result,
      tools,
      rankedScores,
      graph: { ...result.graph, nodes: order },
      context: result.context
        ? {
            ...result.context,
            intent: {
              ...result.context.intent,
              primaryAction: tools[0]?.category ?? "unknown",
              confidence: rankedScores[0] ?? 0,
            },
          }
        : result.context,
    };
    log(
      `[juicio] re-rank top=${adjusted
        .slice(0, 5)
        .map((a) => `${a.tool.name}(${a.score.toFixed(3)})`)
        .join(",")}`,
    );
    return { result: judged, diag };
  }

  /**
   * Premisa de la propuesta pendiente: el último mensaje del asistente en el
   * historial si existe; si no, el resumen del plan cacheado.
   */
  private pendingPremise(input: JuicioPreInput): string | undefined {
    const assistant = [...(input.history ?? [])].reverse().find((m) => m?.role === "assistant");
    if (assistant?.content?.trim()) {
      return `El asistente propone: ${assistant.content.trim()}`;
    }
    const plan = input.cachedPlan;
    if (plan && plan.tools.length > 0) {
      const acciones = plan.tools
        .slice(0, 3)
        .map((t) => t.intentSummary || t.name)
        .join(", ");
      return `El asistente propone ejecutar: ${acciones}`;
    }
    return undefined;
  }
}

/** Resultado vacío por abstención de juicio (intención "sin herramienta"). */
function abstainedResult(
  base: GraphPredictionResult,
  reason: "noop" | "energy",
): GraphPredictionResult {
  return {
    ...base,
    tools: [],
    rankedScores: [],
    graph: { nodes: [], edges: [], executionOrder: [] },
    context: {
      intent: {
        primaryAction: "unknown",
        confidence: base.rankedScores[0] ?? 0,
        summary: `Abstención de juicio (${reason}): la entrada no requiere herramientas`,
      },
      constraints: { negations: [], isConfirmation: false, isExploratory: true },
      dialogState: { phase: "discovery", topicShift: false },
      anticipation: { suggestedNextTools: [], reasoning: "Sin herramienta por abstención" },
    },
  };
}
