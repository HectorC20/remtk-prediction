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
import { adaptiveThreshold } from "src/predict/services/rerank.service";
import { CalibrationService } from "src/predict/services/calibration.service";
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

    // (2b) Atención por saliencia sobre tramos estructurales (textos extensos con relleno).
    if (input.scopeKey) {
      await this.focusSalientSpans(input.scopeKey, input.text, ctx);
    }

    // (3) Proyección sobre la variedad de herramientas (Manifold Geometry) y puerta de coherencia.
    const prev = this.sessionState.peek(input.sessionId);
    ctx.gateLambda = this.puerta.lambda(ctx.uText, prev);

    let currProj: { scores: Map<string, number>; topTool: string; topScore: number; prominence: number } | undefined;
    if (input.scopeKey && ctx.uText) {
      currProj = await this.especificidad.projectManifold(input.scopeKey, ctx.uText);
      if (currProj) {
        ctx.prominence = currProj.prominence;
      }
    }

    // (1) NLI de estado condicionado a la propuesta pendiente:
    // Solo turnos humanos con propuesta. Si el usuario confirma o rechaza la propuesta,
    // es una respuesta directa al asistente, nunca un pivote ajeno a otro dominio.
    if (input.source === "human") {
      const premise = this.pendingPremise(input);
      if (premise) {
        ctx.estado = await this.nli.verdict(input.text, premise);
        if (ctx.estado === "reject") {
          ctx.resolved = true;
          ctx.resolvedBy = "nli_reject";
          log(`[juicio] veredicto REJECT session=${input.sessionId} → propuesta descartada`);
          return ctx;
        } else if (input.cachedPlan && ctx.estado === "confirm") {
          ctx.resolved = true;
          ctx.resolvedBy = "nli_confirm";
          log(`[juicio] veredicto CONFIRM session=${input.sessionId} → plan cacheado`);
          return ctx;
        }
      }
    }

    // Si el usuario no está respondiendo a una propuesta pendiente, evalúa si es un pivote funcional
    if (prev && !ctx.allSpansNoop && ctx.estado !== "confirm" && ctx.estado !== "reject") {
      const [prevProj] = await Promise.all([
        this.especificidad.projectManifold(input.scopeKey, prev),
      ]);
      if (currProj && prevProj && currProj.topTool !== prevProj.topTool) {
        const scoreCurrOnPrev = currProj.scores.get(prevProj.topTool) ?? 0;
        const shiftMargin = currProj.topScore - scoreCurrOnPrev;
        const isFunctionalPivot =
          (shiftMargin >= 0.040 && currProj.topScore >= 0.85) ||
          (shiftMargin >= 0.050 && currProj.prominence >= 0.035);
        if (isFunctionalPivot) {
          ctx.gateLambda = 0.05;
          ctx.isAutonomousPivot = true;
        }
      }
    }

    // Juicio geométrico: solo se aísla el historial si el turno demuestra ser un pivote
    // funcional autónomo (prominencia en un nuevo dominio). Nunca por conteo rígido de palabras ni diccionarios.
    const isAutonomous = ctx.isAutonomousPivot === true;
    ctx.historyGated = isAutonomous && this.puerta.gatesHistory(ctx.gateLambda, prev !== undefined);
    if (ctx.historyGated) {
      log(`[juicio] puerta λ=${ctx.gateLambda?.toFixed(3) ?? "0"} → turno aislado del historial`);
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
    source?: "human" | "agent";
  }): Promise<{ result: GraphPredictionResult; diag: JuicioPostDiag }> {
    const { scopeKey, text, ctx, result, source } = params;
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
        const summary = tool.intentSummary ? `${tool.intentSummary}. ` : "";
        const ce = await this.reranker.score(effectiveText, `${tool.name}: ${summary}${tool.description}`);
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
      const top1Score = adjusted[0].score;
      const relevanceFloor = Math.max(
        this.config.adaptiveMinScore,
        top1Score - this.config.adaptiveGapThreshold * 2,
      );
      // Anclas a nivel de token sobre las herramientas del pool candidato
      const candidateTools = result.tools.length > 0 ? result.tools : this.especificidad.catalogTools(scopeKey);
      const tokenAnchors = await this.maxsim.anchorTools(scopeKey, effectiveText, candidateTools);
      for (const [toolName, anchor] of tokenAnchors.entries()) {
        if (!winPeaks.has(toolName) && anchor.score >= relevanceFloor) {
          winPeaks.set(toolName, {
            tool: anchor.tool,
            score: anchor.score,
            windowIdx: 100 + anchor.tokenIdx,
          });
        }
      }
      const excludedLower = new Set((params.exclude ?? []).map((e) => e.toLowerCase()));
      const winEntries = [...winPeaks.entries()].sort((a, b) => a[1].windowIdx - b[1].windowIdx);
      for (let idx = 0; idx < winEntries.length; idx++) {
        const [toolName, peak] = winEntries[idx];
        if (excludedLower.has(toolName.toLowerCase())) continue;
        if (peak.score < relevanceFloor) continue;
        // La herramienta de sub-ventana preserva su afinidad real acotada bajo el líder
        const cappedScore = Math.min(peak.score, top1Score - 0.005);
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

    // Poda por umbral adaptativo sobre adjusted: preserva la agudeza del juicio y
    // descarta herramientas irrelevantes o destructivas que caen tras un gap natural.
    const scoredAdjusted = adjusted.map((a) => ({ name: a.tool.name, score: a.score }));
    const maxAdaptive = 5;
    const maxOutput = 5;
    const selectedNames = new Set(
      adaptiveThreshold(
        scoredAdjusted,
        this.config.adaptiveMinTools,
        maxAdaptive,
        this.config.adaptiveGapThreshold,
      ),
    );
    const filteredAdjusted = adjusted.filter((a) => selectedNames.has(a.tool.name));
    const finalAdjusted = filteredAdjusted.length > 0 ? filteredAdjusted : adjusted;
    const toolMap = new Map<string, ToolDefinition>();
    for (const a of finalAdjusted) toolMap.set(a.tool.name, a.tool);

    // Preservar herramientas antecedentes del pre-plan DAG (AGENT.md)
    if (result.graph?.edges) {
      for (const tName of [...toolMap.keys()]) {
        for (const e of result.graph.edges) {
          if (e.type === "PREREQUISITE" && e.to === tName) {
            if (!toolMap.has(e.from) && toolMap.size < maxOutput) {
              const preTool = result.tools.find((t) => t.name === e.from);
              if (preTool) toolMap.set(e.from, preTool);
            }
          }
        }
      }
    }

    // Ordenar respetando el orden topológico (Kahn) del grafo precomputado
    const execOrder = result.graph?.executionOrder ?? [];
    const tools = [...toolMap.values()]
      .sort((a, b) => {
        const idxA = execOrder.indexOf(a.name);
        const idxB = execOrder.indexOf(b.name);
        if (idxA !== -1 && idxB !== -1) return idxA - idxB;
        if (idxA !== -1) return -1;
        if (idxB !== -1) return 1;
        return 0;
      })
      .slice(0, maxOutput);

    const order = tools.map((t) => t.name);
    const rankedScores = tools.map((t) => {
      const found = adjusted.find((a) => a.tool.name === t.name);
      return found ? found.score : 0.85;
    });

    const judged: GraphPredictionResult = {
      ...result,
      tools,
      rankedScores,
      calibratedScores: CalibrationService.calibrateRankedScores(rankedScores),
      graph: { ...result.graph, nodes: order },
      context: result.context
        ? {
            ...result.context,
            intent: {
              ...result.context.intent,
              primaryAction: tools[0]?.category ?? "unknown",
              category: tools[0]?.group ?? result.context.intent?.category,
              confidence: Math.max(result.context?.intent?.confidence ?? 0.85, CalibrationService.calibrateProbability(rankedScores[0] ?? 0.85)),
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
