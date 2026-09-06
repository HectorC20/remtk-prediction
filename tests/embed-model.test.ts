/**
 * Verificación del modelo real: carga multilingual-e5-small (384 dims) desde
 * ../models y comprueba que el pipeline usa ONNX (no hash). Requiere el modelo
 * completo (model.onnx + model.onnx_data). Corre con: pnpm test:model
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { EmbeddingEngine, cosine } from "../src/embedding/embedding-engine";
import { createSystem } from "../src/main";
import { EXAMPLE_TOOLS } from "./example-tools";

const CFG = {
  portPredict: 0,
  portEmbed: 0,
  portQdrant: 0,
  onnxEnabled: true,
  onnxModelsPath: "../models",
  onnxModelSize: "small" as const,
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
};

test("e5-small real: modelo=small, dim=384, vector normalizado", { timeout: 300_000 }, async () => {
  const engine = new EmbeddingEngine(CFG);
  const res = await engine.embed("hola mundo");
  assert.equal(res.model, "small", "debe usar ONNX small, no hash");
  assert.equal(res.dim, 384);
  assert.equal(res.embedding.length, 384);
  const norm = Math.sqrt(res.embedding.reduce((s, v) => s + v * v, 0));
  assert.ok(Math.abs(norm - 1) < 1e-3, `esperaba L2=1, obtuve ${norm}`);
});

test("semántica real: textos similares > textos distintos (e5-small)", { timeout: 300_000 }, async () => {
  const engine = new EmbeddingEngine(CFG);
  const [a, b, c] = await Promise.all([
    engine.embed("envía un correo electrónico al equipo"),
    engine.embed("enviar email a los colaboradores"),
    engine.embed("genera una imagen de un paisaje"),
  ]);
  const similar = cosine(a.embedding, b.embedding);
  const different = cosine(a.embedding, c.embedding);
  assert.ok(similar > different, `similar=${similar.toFixed(3)} debe superar different=${different.toFixed(3)}`);
  assert.ok(similar > 0.5, `similitud semántica demasiado baja: ${similar.toFixed(3)}`);
});

test("pipeline /predict usa el modelo real small (modelSize=small)", { timeout: 300_000 }, async () => {
  const system = await createSystem(CFG);
  await system.orchestrator.registerTools("tenant-model", EXAMPLE_TOOLS);

  const result = await system.orchestrator.predict({
    sessionId: "session-model",
    tenant: "tenant-model",
    text: "busca información sobre el clima y envía un correo con el resultado",
    source: "human",
  });

  assert.equal(result.modelSize, "small", `esperaba small, obtuvo ${result.modelSize}`);
  assert.ok(result.tools.length >= 1, "debe seleccionar herramientas con el modelo real");
  const trace = system.debugger.snapshot().traces[0];
  assert.ok(trace, "debe existir una traza");
  assert.equal(trace.modelSize, "small");
});

test("pipeline /predict: confirmación vacía reutiliza el plan cacheado (early-exit)", { timeout: 300_000 }, async () => {
  const system = await createSystem(CFG);
  await system.orchestrator.registerTools("tenant-confirm", EXAMPLE_TOOLS);

  const first = await system.orchestrator.predict({
    sessionId: "session-confirm",
    tenant: "tenant-confirm",
    text: "envía un correo electrónico al equipo",
    source: "human",
  });
  assert.ok(first.tools.length >= 1);

  const second = await system.orchestrator.predict({
    sessionId: "session-confirm",
    tenant: "tenant-confirm",
    text: "confirma y continúa", // clasifica como confirmation_empty con e5-small real
    source: "human",
  });

  const namesA = first.tools.map((t) => t.name);
  const namesB = second.tools.map((t) => t.name);
  assert.deepEqual(namesB, namesA, "la confirmación vacía debe devolver el plan cacheado");

  const traces = system.debugger.snapshot().traces;
  const confirmTrace = traces.find((t) => t.turnType === "confirmation_empty");
  assert.ok(confirmTrace, "debe haber una traza de confirmación vacía");
  assert.equal(confirmTrace.confirmHit, true, "debe marcar confirm_hit=true (early-exit)");
});
