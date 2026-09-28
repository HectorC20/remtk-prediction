/**
 * API de Predicción (puerto 6776) — controlador Nest.
 *
 * Contratos 1:1 con el server Go original (migrado de predict-server.service.ts):
 * /predict, /tools, /tools/count, /memory/predict, /health, /debug. Sin prefijo
 * global: mismos paths raíz que el listener Express original.
 *
 * PredictAppModule expone el contexto (orquestador + engine + debugger) bajo el
 * token PREDICT_CONTEXT; el motor se inyecta con @Inject explícito porque tsx/
 * esbuild no emite design:paramtypes.
 */
import "reflect-metadata";
import {
  BadRequestException,
  Body,
  Controller,
  Get,
  Headers,
  HttpCode,
  Inject,
  InternalServerErrorException,
  Post,
  Query,
} from "@nestjs/common";
import type { EmbeddingEngineService } from "../embedding/embedding.service";
import type { Debugger } from "./helper/debugger.helper";
import type { LexicalProfileService } from "./services/lexical-profile.service";
import type { ToolDefinition } from "../shared/interfaces/domain.interface";
import type { IPredictionOrchestrator } from "../shared/interfaces/orchestrator.interface";
import type { SpatialPredictDto, VisualToolPayload } from "src/shared/interfaces";
import { SpatialPredictService } from "./services/spatial-predict.service";
import { SkillMemoryService } from "./services/skill-memory.service";
import { normalizeAgentId, parseKeyRemtk, resolveScopeKey } from "../shared/scope";
import type { SkillPredictInput } from "../shared/interfaces/skill.interface";

/** Contexto que necesita este controlador (provisto por PredictAppModule). */
export interface PredictContext {
  orchestrator: IPredictionOrchestrator;
  engine: EmbeddingEngineService;
  debugger: Debugger;
  lexical: LexicalProfileService;
  spatial?: SpatialPredictService;
  skillMemory?: SkillMemoryService;
}

/** Token de inyección del contexto (no es una clase inyectable por tipo). */
export const PREDICT_CONTEXT = Symbol("predict-context");

@Controller()
export class PredictV1Controller {
  constructor(@Inject(PREDICT_CONTEXT) private readonly ctx: PredictContext) {}

  @Get("health")
  health(): { status: string } {
    return { status: "ok" };
  }

  /**
   * POST /spatial/predict
   * Predicción de acciones en lienzo visual interactivo con coordenadas y contexto anafórico.
   */
  @Post("spatial/predict")
  @HttpCode(200)
  async predictSpatial(@Body() body: SpatialPredictDto): Promise<VisualToolPayload> {
    if (!body || typeof body.query !== "string" || body.query.trim() === "") {
      throw new BadRequestException({ error: "El campo query (string) es requerido" });
    }
    const service = this.ctx.spatial ?? new SpatialPredictService(this.ctx.engine);
    const cursorStr = body.cursor ? `(${body.cursor.x.toFixed(2)}, ${body.cursor.y.toFixed(2)})` : "(0.50, 0.50)";
    console.log(`\n================== [SPATIAL HARNESS] ==================`);
    console.log(`📥 TEXTO TRANSCRIBIENDO/RECIBIDO : "${body.query}"`);
    console.log(`📍 CONTEXTO ESPACIAL             : Cursor=${cursorStr} | Seleccionado=${body.selectedId ?? "ninguno"}`);
    
    const result = await service.predict(body);
    
    console.log(`🛠️  HERRAMIENTA UTILIZADA        : [${result.tool}]`);
    console.log(`📦 PARÁMETROS                    :`, JSON.stringify(result.parameters));
    console.log(`🎯 CONFIANZA / LATENCIA          : ${(result.confidence * 100).toFixed(0)}% | ${result.executionTimeMs}ms`);
    console.log(`=======================================================\n`);
    return result;
  }

  @Post("predict/spatial")
  @HttpCode(200)
  async predictSpatialAlias(@Body() body: SpatialPredictDto): Promise<VisualToolPayload> {
    return this.predictSpatial(body);
  }

  /**
   * POST /predict
   * Body: { sessionId, tenant, agentId?, keyRemtk?, text, source: "human"|"agent", priorPlan?, keywords?, intentContext?, exclude? }
   * Headers: key-remtk / x-remtk-key
   * Respuesta: { tools, complexity, modelSize, rankedScores }
   */
  @Post("predict")
  @HttpCode(200)
  async predict(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { tenant, agentId } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    const { sessionId, text, source, priorPlan, history, keywords, exclude, intentContext } = b;
    if (typeof sessionId !== "string" || !tenant || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant (o key-remtk) y text (string) requeridos" });
    }
    try {
      return await this.ctx.orchestrator.predict({
        sessionId,
        tenant,
        text,
        source: source === "agent" ? "agent" : "human",
        agentId,
        priorPlan: typeof priorPlan === "string" ? priorPlan : undefined,
        intentContext: typeof intentContext === "string" ? intentContext : undefined,
        keywords: Array.isArray(keywords) ? asNames(keywords) : undefined,
        exclude: Array.isArray(exclude) ? asNames(exclude) : undefined,
        history: Array.isArray(history)
          ? (history as { role?: unknown; content?: unknown }[])
              .filter((h) => h && typeof h.content === "string")
              .map((h) => ({ role: String(h.role ?? "user"), content: h.content as string }))
          : undefined,
      });
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /tools
   * Body: { tenant, agentId?, keyRemtk?, tools: ToolDefinition[] }
   * Headers: key-remtk / x-remtk-key
   * Respuesta: { indexed: number }
   */
  @Post("tools")
  @HttpCode(200)
  async registerTools(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { tenant, agentId } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    if (!tenant || !Array.isArray(b.tools)) {
      throw new BadRequestException({ error: "tenant (o key-remtk) y tools[] requeridos" });
    }
    try {
      return await this.ctx.orchestrator.registerTools(
        tenant,
        b.tools as ToolDefinition[],
        agentId,
      );
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /predict/feedback
   * Body: { tenant, agentId?, keyRemtk?, sessionId, text, used: string[], rejected?: string[] }
   * Headers: key-remtk / x-remtk-key
   * Respuesta: { ok: true, events }
   */
  @Post("predict/feedback")
  @HttpCode(200)
  feedback(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): unknown {
    const b = body ?? {};
    const { tenant, agentId } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    const { sessionId, text } = b;
    if (typeof sessionId !== "string" || !tenant || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant (o key-remtk) y text (string) requeridos" });
    }
    try {
      return this.ctx.orchestrator.feedback({
        sessionId,
        tenant,
        text,
        agentId,
        used: asNames(b.used),
        rejected: asNames(b.rejected),
      });
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /** GET /tools/count?tenant=&agentId= → {count} */
  @Get("tools/count")
  toolsCount(
    @Query("tenant") tenant?: string,
    @Query("agentId") agentId?: string,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): { count: number } {
    const parsed = parseKeyRemtk(keyRemtkHeader ?? xRemtkKeyHeader, tenant, agentId);
    return { count: parsed.tenant ? this.ctx.orchestrator.countTools(parsed.tenant, parsed.agentId) : 0 };
  }

  /** GET /tools/ready?tenant=&agentId= → espera a que el warm-up del scope concluya */
  @Get("tools/ready")
  async toolsReady(
    @Query("tenant") tenant?: string,
    @Query("agentId") agentId?: string,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<{ ready: boolean }> {
    const parsed = parseKeyRemtk(keyRemtkHeader ?? xRemtkKeyHeader, tenant, agentId);
    if (parsed.tenant) await this.ctx.orchestrator.waitForWarmup(parsed.tenant, parsed.agentId);
    return { ready: true };
  }

  /**
   * POST /memory/predict
   * Body: { sessionId, tenant, agentId?, keyRemtk?, text, limit }
   * Headers: key-remtk / x-remtk-key
   * Respuesta: { memories, topicShift, topicScore, modelSize, rankedScores, activeEntities, skillContext }
   */
  @Post("memory/predict")
  @HttpCode(200)
  async predictMemory(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { tenant, agentId, scopeKey } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    const { sessionId, text } = b;
    if (typeof sessionId !== "string" || !tenant || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant (o key-remtk) y text (string) requeridos" });
    }
    const skillService = this.ctx.skillMemory ?? new SkillMemoryService(this.ctx.engine);
    try {
      const memRes = await this.ctx.orchestrator.predictMemory({
        sessionId,
        tenant,
        text,
        agentId,
        limit: Number.isFinite(Number(b.limit)) ? Number(b.limit) : 8,
      });

      // Enriquecer con el contexto activo de habilidades y entidades de sesión para remtk-memory
      const activeEntities = skillService.getSessionEntities(sessionId, scopeKey);
      let skillContext;
      try {
        skillContext = await skillService.predictSkill({
          sessionId,
          tenant,
          agentId,
          keyRemtk: scopeKey,
          text,
        });
      } catch {
        // Fallback suave
      }

      return {
        ...memRes,
        activeEntities,
        skillContext,
      };
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /memory/judge
   * Evalúa semánticamente si un hecho o texto representa ruido de negación de herramientas
   * no apto para persistir en memoria contextual de largo plazo.
   */
  @Post("memory/judge")
  @HttpCode(200)
  async judgeMemory(@Body() body: Record<string, unknown>): Promise<{ isNoise: boolean; noiseScore: number }> {
    const text = String(body?.text ?? "").trim();
    if (!text) return { isNoise: false, noiseScore: 0 };
    try {
      return await this.ctx.orchestrator.judgeMemory(text);
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /interest
   * Body: { sessionId, tenant, agentId?, keyRemtk?, text, anchors?: string[], limit? }
   * Headers: key-remtk / x-remtk-key
   * Respuesta: { interestScore, topic, topicScore, intentScore, matches, method }
   */
  @Post("interest")
  @HttpCode(200)
  async predictInterest(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { tenant, agentId } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    const { sessionId, text } = b;
    if (typeof sessionId !== "string" || !tenant || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant (o key-remtk) y text (string) requeridos" });
    }
    try {
      return await this.ctx.orchestrator.predictInterest({
        sessionId,
        tenant,
        text,
        agentId,
        anchors: Array.isArray(b.anchors)
          ? (b.anchors as unknown[]).filter((a): a is string => typeof a === "string")
          : undefined,
        limit: Number.isFinite(Number(b.limit)) ? Number(b.limit) : undefined,
      });
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /skills/learn
   * Registra o refuerza una habilidad aprendida con su grafo y anclas de parámetros.
   * Headers: key-remtk / x-remtk-key
   */
  @Post("skills/learn")
  @HttpCode(200)
  async learnSkill(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { tenant, agentId, scopeKey } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    const { id, name, description, intentSummary, tools, parameterGrounding } = b;
    if (
      !tenant ||
      typeof id !== "string" ||
      typeof name !== "string" ||
      !Array.isArray(tools)
    ) {
      throw new BadRequestException({ error: "tenant (o key-remtk), id, name y tools[] son requeridos" });
    }
    const skillService = this.ctx.skillMemory ?? new SkillMemoryService(this.ctx.engine);
    try {
      return await skillService.learnSkill(
        scopeKey,
        {
          id,
          name,
          description: typeof description === "string" ? description : name,
          intentSummary: typeof intentSummary === "string" ? intentSummary : name,
          tools: asNames(tools),
          parameterGrounding: parameterGrounding as Record<string, { id: string; name: string }[]> | undefined,
        },
        agentId,
      );
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /skills/predict
   * Predice la habilidad adecuada y resuelve entidades y parámetros pre-calculados.
   * Headers: key-remtk / x-remtk-key
   */
  @Post("skills/predict")
  @HttpCode(200)
  async predictSkill(
    @Body() body: SkillPredictInput,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { tenant, agentId, scopeKey } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      b.tenant,
      b.agentId,
      b.keyRemtk,
    );
    const { sessionId, text } = b;
    if (typeof sessionId !== "string" || !tenant || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant (o key-remtk) y text son requeridos" });
    }
    const skillService = this.ctx.skillMemory ?? new SkillMemoryService(this.ctx.engine);
    try {
      return await skillService.predictSkill({
        ...b,
        tenant,
        agentId,
        keyRemtk: scopeKey,
      });
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /entities/track
   * Registra o actualiza el ciclo de vida y acción pendiente de una entidad en la sesión.
   * Headers: key-remtk / x-remtk-key
   */
  @Post("entities/track")
  @HttpCode(200)
  async trackEntity(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): Promise<unknown> {
    const b = body ?? {};
    const { scopeKey, agentId } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof b.tenant === "string" ? b.tenant : undefined,
      typeof b.agentId === "string" ? b.agentId : undefined,
      typeof b.keyRemtk === "string" ? b.keyRemtk : (typeof b.key_remtk === "string" ? b.key_remtk : undefined),
    );
    const { sessionId, id, type, name, slug, state, attributes } = b;
    if (typeof sessionId !== "string" || typeof id !== "string" || typeof type !== "string" || typeof name !== "string") {
      throw new BadRequestException({ error: "sessionId, id, type y name son requeridos" });
    }
    const skillService = this.ctx.skillMemory ?? new SkillMemoryService(this.ctx.engine);
    try {
      return await skillService.trackEntity(
        sessionId,
        {
          id,
          type,
          name,
          slug: typeof slug === "string" ? slug : undefined,
          state: typeof state === "object" && state !== null ? (state as any) : undefined,
          attributes: typeof attributes === "object" && attributes !== null ? (attributes as any) : undefined,
        },
        scopeKey,
        agentId,
      );
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /**
   * POST /entities/clear
   * Limpia el grafo de entidades de una sesión específica.
   * Headers: key-remtk / x-remtk-key
   */
  @Post("entities/clear")
  @HttpCode(200)
  clearEntities(
    @Body() body: Record<string, unknown>,
    @Headers("key-remtk") keyRemtkHeader?: string,
    @Headers("x-remtk-key") xRemtkKeyHeader?: string,
  ): { ok: boolean } {
    const { scopeKey, agentId } = parseKeyRemtk(
      keyRemtkHeader ?? xRemtkKeyHeader,
      typeof body?.tenant === "string" ? body.tenant : undefined,
      typeof body?.agentId === "string" ? body.agentId : undefined,
      typeof body?.keyRemtk === "string" ? body.keyRemtk : (typeof body?.key_remtk === "string" ? body.key_remtk : undefined),
    );
    const sessionId = typeof body?.sessionId === "string" ? body.sessionId : "";
    if (!sessionId) throw new BadRequestException({ error: "sessionId requerido" });
    const skillService = this.ctx.skillMemory ?? new SkillMemoryService(this.ctx.engine);
    skillService.clearSession(sessionId, scopeKey, agentId);
    return { ok: true };
  }

  /**
   * GET /debug?tenant=&agentId= → engine, stats, trazas de las últimas
   * predicciones y el resumen del perfil léxico aprendido del canal (F3).
   */
  @Get("debug")
  async debug(
    @Query("tenant") tenant?: string,
    @Query("agentId") agentId?: string,
  ): Promise<{
    engine: { size: string; modelPath: string };
    stats: unknown;
    traces: unknown;
    lexical?: unknown;
  }> {
    const engineInfo = await this.ctx.engine.getInfo(this.ctx.engine.defaultSize);
    const snapshot = this.ctx.debugger.snapshot();
    const t = typeof tenant === "string" ? tenant.trim() : "";
    return {
      engine: { size: engineInfo.model, modelPath: engineInfo.modelPath },
      stats: snapshot.stats,
      traces: snapshot.traces,
      lexical: t === "" ? undefined : this.ctx.lexical.snapshot(resolveScopeKey(t, normalizeAgentId(agentId))),
    };
  }
}

/** Normaliza una lista de nombres de herramienta recibida por HTTP. */
function asNames(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === "string");
}
