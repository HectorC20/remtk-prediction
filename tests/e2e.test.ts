/**
 * Test end-to-end del servicio unificado: levanta los 3 listeners en puertos
 * efímeros y valida predicción + memoria + modelos + Qdrant API, simulando el
 * envío de 12 herramientas desde NestJS. Hermético (sin ONNX real ni Qdrant).
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import type { Express } from "express";
import { createSystem, type System } from "../src/main";
import { createPredictServer } from "../src/predict/predict-server";
import { createEmbedServer } from "../src/embedding/embed-server";
import { createQdrantServer } from "../src/qdrant/qdrant-server";
import { EXAMPLE_TOOLS } from "./example-tools";

function listen(app: Express): Promise<{ port: number; close: () => Promise<void> }> {
  return new Promise((resolve, reject) => {
    const server = app.listen(0, () => {
      const addr = server.address();
      resolve({
        port: typeof addr === "object" && addr ? addr.port : 0,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
    server.on("error", reject);
  });
}

let system: System;
let predictUrl = "";
let embedUrl = "";
let qdrantUrl = "";
let closes: (() => Promise<void>)[] = [];

before(async () => {
  system = await createSystem({
    portPredict: 0,
    portEmbed: 0,
    portQdrant: 0,
    onnxEnabled: false, // hermético: fallback hash determinístico
    onnxModelsPath: "../models",
    onnxModelSize: "small",
    adaptiveMinTools: 2,
    adaptiveMaxTools: 5,
    adaptiveGapThreshold: 0.15,
    adaptiveMinScore: 0,
    keywordBoost: 0.15,
    keywordTopK: 20,
    qdrantEnabled: false,
    qdrantUrl: "http://localhost:6333",
    qdrantApiKey: "",
    toolsCollection: "mcp_tools",
    keywordsCollection: "tool_keywords",
    synonymsCollection: "query_synonyms",
    memoriesCollection: "contextual_memories",
    recallLimit: 50,
  });

  const [p, e, q] = await Promise.all([
    listen(createPredictServer(system.orchestrator, system.debugger, system.engine)),
    listen(createEmbedServer(system.engine)),
    listen(createQdrantServer(system.qdrant)),
  ]);
  predictUrl = `http://localhost:${p.port}`;
  embedUrl = `http://localhost:${e.port}`;
  qdrantUrl = `http://localhost:${q.port}`;
  closes = [p.close, e.close, q.close];
});

after(async () => {
  await Promise.all(closes.map((c) => c()));
});

async function post(url: string, path: string, body: unknown): Promise<{ status: number; data: unknown }> {
  const res = await fetch(`${url}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, data: await res.json() };
}

test("health de los 3 puertos responde ok", async () => {
  for (const url of [predictUrl, embedUrl, qdrantUrl]) {
    const res = await fetch(`${url}/health`);
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { status: "ok" });
  }
});

test("el catálogo de ejemplo tiene al menos 10 herramientas", () => {
  assert.ok(EXAMPLE_TOOLS.length >= 10, `esperaba >=10, hay ${EXAMPLE_TOOLS.length}`);
});

test("POST /tools indexa las 12 herramientas y /tools/count las reporta", async () => {
  const res = await post(predictUrl, "/tools", { tenant: "tenant-demo", tools: EXAMPLE_TOOLS });
  assert.equal(res.status, 200);
  assert.equal((res.data as { indexed: number }).indexed, EXAMPLE_TOOLS.length);

  const count = await fetch(`${predictUrl}/tools/count?tenant=tenant-demo`);
  const countData = (await count.json()) as { count: number };
  assert.equal(countData.count, EXAMPLE_TOOLS.length);
});

test("POST /predict selecciona herramientas del catálogo", async () => {
  const res = await post(predictUrl, "/predict", {
    sessionId: "session-1",
    tenant: "tenant-demo",
    text: "busca información sobre el clima en lima",
    source: "human",
  });
  assert.equal(res.status, 200);
  const result = res.data as {
    tools: { name: string }[];
    complexity: string;
    modelSize: string;
    rankedScores: number[];
  };
  assert.ok(result.tools.length >= 1, "debe seleccionar al menos una herramienta");
  assert.ok(result.tools.length <= 5, `máximo 5, obtuvo ${result.tools.length}`);
  assert.equal(typeof result.complexity, "string");
  assert.ok(["large", "small", "hash"].includes(result.modelSize));

  // Las herramientas devueltas deben existir en el catálogo.
  const names = new Set(EXAMPLE_TOOLS.map((t) => t.name));
  for (const tool of result.tools) {
    assert.ok(names.has(tool.name), `${tool.name} no está en el catálogo`);
  }
});

test("POST /predict con confirmación responde tools (early-exit se valida con modelo real)", async () => {
  const first = await post(predictUrl, "/predict", {
    sessionId: "session-confirm",
    tenant: "tenant-demo",
    text: "envía un correo electrónico al equipo",
    source: "human",
  });
  assert.equal(first.status, 200);
  const second = await post(predictUrl, "/predict", {
    sessionId: "session-confirm",
    tenant: "tenant-demo",
    text: "continúa",
    source: "human",
  });
  assert.equal(second.status, 200);
  assert.ok((second.data as { tools: unknown[] }).tools);
});

test("POST /memory/predict devuelve memorias, topicShift y scores", async () => {
  const res = await post(predictUrl, "/memory/predict", {
    sessionId: "session-mem",
    tenant: "user-123",
    text: "recuerda que prefiero respuestas cortas",
    limit: 8,
  });
  assert.equal(res.status, 200);
  const result = res.data as {
    memories: unknown[];
    topicShift: boolean;
    topicScore: number;
    modelSize: string;
    rankedScores: number[];
  };
  assert.ok(Array.isArray(result.memories));
  assert.equal(typeof result.topicShift, "boolean");
  assert.equal(typeof result.topicScore, "number");
  assert.ok(["large", "small", "hash"].includes(result.modelSize));
  assert.ok(Array.isArray(result.rankedScores));
});

test("POST /embed (6777) responde embedding con dim de small", async () => {
  const res = await post(embedUrl, "/embed", { text: "hola mundo" });
  assert.equal(res.status, 200);
  const data = res.data as { embedding: number[]; model: string; dim: number; modelPath: string };
  assert.equal(data.model, "hash");
  assert.equal(data.dim, 384);
  assert.equal(data.embedding.length, 384);

  const small = await post(embedUrl, "/embed", { text: "hola", size: "small" });
  const smallData = small.data as { dim: number };
  assert.equal(smallData.dim, 384);
});

test("GET /models (6777) expone el modelo small", async () => {
  const res = await fetch(`${embedUrl}/models`);
  assert.equal(res.status, 200);
  const data = (await res.json()) as { small?: { dim?: number } };
  assert.ok(data && typeof data === "object");
  assert.equal(data.small?.dim, 384);
});

test("API Qdrant (6778) responde status y recall de tools", async () => {
  const status = await fetch(`${qdrantUrl}/status`);
  assert.equal(status.status, 200);

  const upsert = await post(qdrantUrl, "/tools/upsert", { tenant: "tenant-qdrant", tools: EXAMPLE_TOOLS });
  assert.equal(upsert.status, 200);

  const search = await post(qdrantUrl, "/tools/search", { tenant: "tenant-qdrant", text: "envía un correo" });
  assert.equal(search.status, 200);
  const data = search.data as { tools: unknown[] };
  assert.ok(Array.isArray(data.tools));
});

test("GET /debug expone engine, stats y trazas", async () => {
  const res = await fetch(`${predictUrl}/debug`);
  assert.equal(res.status, 200);
  const data = (await res.json()) as { engine: unknown; stats: unknown; traces: unknown };
  assert.ok(data.engine);
  assert.ok(data.stats);
  assert.ok(Array.isArray(data.traces));
});
