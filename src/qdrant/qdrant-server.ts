/** API de Qdrant (puerto 6778): recall BM25, keywording e indexado de tools. */
import express, { type Express } from "express";
import type { QdrantService } from "./qdrant-service";
import type { ToolDefinition } from "../types";

export function createQdrantServer(service: QdrantService): Express {
  const app = express();
  app.use(express.json({ limit: "2mb" }));

  app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  app.get("/status", async (_req, res) => {
    res.json(await service.status());
  });

  /** POST /tools/upsert {tenant, tools: ToolDefinition[]} → {indexed: number} */
  app.post("/tools/upsert", async (req, res) => {
    const body = (req.body ?? {}) as { tenant?: unknown; tools?: unknown };
    if (typeof body.tenant !== "string" || !Array.isArray(body.tools)) {
      res.status(400).json({ error: "tenant y tools[] requeridos" });
      return;
    }
    await service.indexTools(body.tenant, body.tools as ToolDefinition[]);
    res.json({ indexed: service.countTools(body.tenant) });
  });

  /** GET /tools/count?tenant= → {count} */
  app.get("/tools/count", (req, res) => {
    const tenant = typeof req.query.tenant === "string" ? req.query.tenant : "";
    res.json({ count: tenant ? service.countTools(tenant) : 0 });
  });

  /** POST /tools/search {tenant, text, limit?} → {tools: {name, score}[]} */
  app.post("/tools/search", async (req, res) => {
    const body = (req.body ?? {}) as { tenant?: unknown; text?: unknown; limit?: unknown };
    if (typeof body.tenant !== "string" || typeof body.text !== "string") {
      res.status(400).json({ error: "tenant y text requeridos" });
      return;
    }
    const limit = Number.isFinite(Number(body.limit)) ? Number(body.limit) : 20;
    try {
      const tools = await service.retrieveTools(body.tenant, body.text, limit);
      res.json({ tools });
    } catch (err) {
      res.status(500).json({ error: String((err as Error)?.message ?? err) });
    }
  });

  /** POST /memories/search {userId, text, limit?} → {memories: MemoryDefinition[]} */
  app.post("/memories/search", async (req, res) => {
    const body = (req.body ?? {}) as { userId?: unknown; text?: unknown; limit?: unknown };
    if (typeof body.userId !== "string" || typeof body.text !== "string") {
      res.status(400).json({ error: "userId y text requeridos" });
      return;
    }
    const limit = Number.isFinite(Number(body.limit)) ? Number(body.limit) : 8;
    try {
      const candidates = await service.searchMemories(body.userId, body.text, limit);
      res.json({ memories: candidates.map((c) => c.memory) });
    } catch (err) {
      res.status(500).json({ error: String((err as Error)?.message ?? err) });
    }
  });

  /** POST /tools/synonyms {text} → {expanded} */
  app.post("/tools/synonyms", async (req, res) => {
    const body = (req.body ?? {}) as { text?: unknown };
    if (typeof body.text !== "string") {
      res.status(400).json({ error: "text requerido" });
      return;
    }
    res.json({ expanded: await service.expandSynonyms(body.text) });
  });

  return app;
}
