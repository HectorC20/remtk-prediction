/**
 * API de Predicción (puerto 6776): contratos 1:1 con el server Go original.
 * Rutas: /predict, /tools, /tools/count, /memory/predict, /health, /debug.
 */
import express, { type Express } from "express";
import type { PredictionOrchestrator } from "./orchestrator";
import type { Debugger } from "./debugger";
import type { EmbeddingEngine } from "../embedding/embedding-engine";
import type { ToolDefinition } from "../types";

export function createPredictServer(
  orchestrator: PredictionOrchestrator,
  debugger_: Debugger,
  engine: EmbeddingEngine,
): Express {
  const app = express();
  app.use(express.json({ limit: "2mb" }));

  app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  /**
   * POST /predict
   * Body: { sessionId, tenant, text, source: "human"|"agent", priorPlan? }
   * Respuesta: { tools, complexity, modelSize, rankedScores }
   */
  app.post("/predict", async (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const { sessionId, tenant, text, source, priorPlan, history } = body;
    if (typeof sessionId !== "string" || typeof tenant !== "string" || typeof text !== "string") {
      res.status(400).json({ error: "sessionId, tenant y text (string) requeridos" });
      return;
    }
    try {
      const result = await orchestrator.predict({
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
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: String((err as Error)?.message ?? err) });
    }
  });

  /**
   * POST /tools
   * Body: { tenant, tools: ToolDefinition[] }
   * Respuesta: { indexed: number }
   */
  app.post("/tools", async (req, res) => {
    const body = (req.body ?? {}) as { tenant?: unknown; tools?: unknown };
    if (typeof body.tenant !== "string" || !Array.isArray(body.tools)) {
      res.status(400).json({ error: "tenant y tools[] requeridos" });
      return;
    }
    try {
      const result = await orchestrator.registerTools(body.tenant, body.tools as ToolDefinition[]);
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: String((err as Error)?.message ?? err) });
    }
  });

  /** GET /tools/count?tenant= → {count} */
  app.get("/tools/count", (req, res) => {
    const tenant = typeof req.query.tenant === "string" ? req.query.tenant : "";
    res.json({ count: tenant ? orchestrator.countTools(tenant) : 0 });
  });

  /**
   * POST /memory/predict
   * Body: { sessionId, tenant, text, limit }
   * Respuesta: { memories, topicShift, topicScore, modelSize, rankedScores }
   */
  app.post("/memory/predict", async (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const { sessionId, tenant, text } = body;
    if (typeof sessionId !== "string" || typeof tenant !== "string" || typeof text !== "string") {
      res.status(400).json({ error: "sessionId, tenant y text (string) requeridos" });
      return;
    }
    try {
      const result = await orchestrator.predictMemory({
        sessionId,
        tenant,
        text,
        limit: Number.isFinite(Number(body.limit)) ? Number(body.limit) : 8,
      });
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: String((err as Error)?.message ?? err) });
    }
  });

  /** GET /debug → engine, stats y trazas de las últimas predicciones. */
  app.get("/debug", async (_req, res) => {
    const engineInfo = await engine.getInfo(engine.defaultSize);
    const snapshot = debugger_.snapshot();
    res.json({
      engine: { size: engineInfo.model, modelPath: engineInfo.modelPath },
      stats: snapshot.stats,
      traces: snapshot.traces,
    });
  });

  return app;
}
