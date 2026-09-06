/** API de Modelos (puerto 6777): POST /embed con e5-large y e5-small en un solo proceso. */
import express, { type Express } from "express";
import type { EmbeddingEngine } from "./embedding-engine";
import type { ModelSize } from "../types";

export function createEmbedServer(engine: EmbeddingEngine): Express {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  /** GET /models → estado de ambos modelos empaquetados. */
  app.get("/models", async (_req, res) => {
    // const [large, small] = await Promise.all([engine.getInfo("large"), engine.getInfo("small")]);
    const small = await engine.getInfo("small");
    res.json({ small });
  });

  /**
   * POST /embed
   * Body: { text: string, size?: "large" | "small" }  (sin prefijo; el caller aplica query:/passage:)
   * Respuesta: { embedding: number[], model: "large"|"small"|"hash", dim, modelPath }
   */
  app.post("/embed", async (req, res) => {
    const body = (req.body ?? {}) as { text?: unknown; size?: unknown };
    if (typeof body.text !== "string" || body.text.trim() === "") {
      res.status(400).json({ error: "text requerido" });
      return;
    }
    const size: ModelSize | undefined =
      // body.size === "small" ? body.size : undefined;
      "small";

    try {
      const r = await engine.embed(body.text, size);
      res.json({
        embedding: Array.from(r.embedding),
        model: r.model,
        dim: r.dim,
        modelPath: r.modelPath,
      });
    } catch (err) {
      res.status(500).json({ error: String((err as Error)?.message ?? err) });
    }
  });

  return app;
}
