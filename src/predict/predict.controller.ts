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
  HttpCode,
  Inject,
  InternalServerErrorException,
  Post,
  Query,
} from "@nestjs/common";
import type { EmbeddingEngineService } from "../embedding/embedding.service";
import type { Debugger } from "./helper/debugger.helper";
import type { ToolDefinition } from "../shared/interfaces/domain.interface";
import type { IPredictionOrchestrator } from "../shared/interfaces/orchestrator.interface";

/** Contexto que necesita este controlador (provisto por PredictAppModule). */
export interface PredictContext {
  orchestrator: IPredictionOrchestrator;
  engine: EmbeddingEngineService;
  debugger: Debugger;
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
   * POST /predict
   * Body: { sessionId, tenant, text, source: "human"|"agent", priorPlan? }
   * Respuesta: { tools, complexity, modelSize, rankedScores }
   */
  @Post("predict")
  @HttpCode(200)
  async predict(@Body() body: Record<string, unknown>): Promise<unknown> {
    const b = body ?? {};
    const { sessionId, tenant, text, source, priorPlan, history } = b;
    if (typeof sessionId !== "string" || typeof tenant !== "string" || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant y text (string) requeridos" });
    }
    try {
      return await this.ctx.orchestrator.predict({
        sessionId,
        tenant,
        text,
        source: source === "agent" ? "agent" : "human",
        priorPlan: typeof priorPlan === "string" ? priorPlan : undefined,
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
   * Body: { tenant, tools: ToolDefinition[] }
   * Respuesta: { indexed: number }
   */
  @Post("tools")
  @HttpCode(200)
  async registerTools(@Body() body: Record<string, unknown>): Promise<unknown> {
    const b = body ?? {};
    if (typeof b.tenant !== "string" || !Array.isArray(b.tools)) {
      throw new BadRequestException({ error: "tenant y tools[] requeridos" });
    }
    try {
      return await this.ctx.orchestrator.registerTools(b.tenant, b.tools as ToolDefinition[]);
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /** GET /tools/count?tenant= → {count} */
  @Get("tools/count")
  toolsCount(@Query("tenant") tenant?: string): { count: number } {
    const t = typeof tenant === "string" ? tenant : "";
    return { count: t ? this.ctx.orchestrator.countTools(t) : 0 };
  }

  /**
   * POST /memory/predict
   * Body: { sessionId, tenant, text, limit }
   * Respuesta: { memories, topicShift, topicScore, modelSize, rankedScores }
   */
  @Post("memory/predict")
  @HttpCode(200)
  async predictMemory(@Body() body: Record<string, unknown>): Promise<unknown> {
    const b = body ?? {};
    const { sessionId, tenant, text } = b;
    if (typeof sessionId !== "string" || typeof tenant !== "string" || typeof text !== "string") {
      throw new BadRequestException({ error: "sessionId, tenant y text (string) requeridos" });
    }
    try {
      return await this.ctx.orchestrator.predictMemory({
        sessionId,
        tenant,
        text,
        limit: Number.isFinite(Number(b.limit)) ? Number(b.limit) : 8,
      });
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /** GET /debug → engine, stats y trazas de las últimas predicciones. */
  @Get("debug")
  async debug(): Promise<{
    engine: { size: string; modelPath: string };
    stats: unknown;
    traces: unknown;
  }> {
    const engineInfo = await this.ctx.engine.getInfo(this.ctx.engine.defaultSize);
    const snapshot = this.ctx.debugger.snapshot();
    return {
      engine: { size: engineInfo.model, modelPath: engineInfo.modelPath },
      stats: snapshot.stats,
      traces: snapshot.traces,
    };
  }
}
