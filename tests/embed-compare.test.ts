/**
 * Comparativa real e5-small vs e5-large (ONNX, CPU): carga ambos modelos y
 * verifica dims, normalización, calidad semántica y latencia (large debe ser
 * más lento pero con vector 1024). Requiere ambos modelos en ./models.
 * Corre con: pnpm test:compare
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { EmbeddingEngineService } from "../src/embedding/embedding-engine";
import type { EmbedResult } from "../src/shared/interfaces/embedding.interface";

const CFG = {
  portPredict: 0,
  portEmbed: 0,
  portQdrant: 0,
  onnxEnabled: true,
  onnxModelsPath: "./models",
  onnxModelSize: "small" as const,
  adaptiveMinTools: 2,
  adaptiveMaxTools: 5,
  adaptiveGapThreshold: 0.15,
  adaptiveMinScore: 0,
  keywordBoost: 0.15,
  keywordTopK: 20,
  recallLimit: 50,
  maxOutputTools: 50,
  qdrantEnabled: false,
  qdrantUrl: "http://localhost:6333",
  qdrantApiKey: "",
  toolsCollection: "mcp_tools",
  keywordsCollection: "tool_keywords",
  synonymsCollection: "query_synonyms",
  memoriesCollection: "contextual_memories",
};

const SIZES: Array<{ size: "small" | "large"; dim: number }> = [
  { size: "small", dim: 384 },
  { size: "large", dim: 1024 },
];

const smallExists = existsSync(join("models", "multilingual-e5-small-onnx", "model.onnx"));
const largeExists = existsSync(join("models", "multilingual-e5-large-onnx", "model.onnx"));
const available = smallExists && largeExists;

// Un único motor compartido: small se carga una vez y large bajo demanda.
const engine = new EmbeddingEngineService(CFG);

const TEXT_SIM_A = "envía un correo electrónico al equipo";
const TEXT_SIM_B = "enviar email a los colaboradores del grupo";
const TEXT_DIF = "genera una imagen de un paisaje nevado";

function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** Texto que tras truncar a 512 tokens representa un embed "largo". */
const TEXT_LONG = Array.from(
  { length: 200 },
  () => "enviar correo electrónico a los colaboradores del proyecto",
).join(" ");

test("small y large cargan ONNX real y producen su dim correcta", { skip: !available, timeout: 300_000 }, async () => {
  for (const { size, dim } of SIZES) {
    const res: EmbedResult = await engine.embed("hola mundo", size);
    assert.equal(res.model, size, `esperaba ONNX ${size}, obtuvo ${res.model}`);
    assert.equal(res.dim, dim, `dim de ${size} debe ser ${dim}`);
    assert.equal(res.embedding.length, dim, `vector de ${size} debe medir ${dim}`);
  }
});

test("ambos tamaños emiten vectores L2-normalizados", { skip: !available, timeout: 300_000 }, async () => {
  for (const { size } of SIZES) {
    const res = await engine.embed(TEXT_SIM_A, size);
    const n = norm(res.embedding);
    assert.ok(Math.abs(n - 1) < 1e-3, `norma de ${size}=${n.toFixed(4)}, esperaba 1`);
  }
});

test("calidad semántica: afines > disímiles en ambos tamaños (large ≥ small)", { skip: !available, timeout: 300_000 }, async () => {
  let smallCos = 0;
  for (const { size } of SIZES) {
    const [a, b, c] = await Promise.all([
      engine.embed(TEXT_SIM_A, size),
      engine.embed(TEXT_SIM_B, size),
      engine.embed(TEXT_DIF, size),
    ]);
    const similar = EmbeddingEngineService.cosine(a.embedding, b.embedding);
    const different = EmbeddingEngineService.cosine(a.embedding, c.embedding);
    assert.ok(similar > different, `${size}: similar=${similar.toFixed(3)} debe superar different=${different.toFixed(3)}`);
    assert.ok(similar > 0.5, `${size}: similitud semántica demasiado baja: ${similar.toFixed(3)}`);
    if (size === "small") smallCos = similar;
    else assert.ok(similar >= smallCos - 0.05, `large (${similar.toFixed(3)}) no debe ser peor que small (${smallCos.toFixed(3)})`);
  }
});

test("latencia: large es más lento que small (texto 512 tokens, CPU)", { skip: !available, timeout: 300_000 }, async () => {
  const runs = 3;
  const avg: Record<string, number> = {};
  for (const { size } of SIZES) {
    await engine.embed(TEXT_LONG, size); // warmup de esta longitud
    const times: number[] = [];
    for (let i = 0; i < runs; i++) {
      const t0 = performance.now();
      await engine.embed(TEXT_LONG, size);
      times.push(performance.now() - t0);
    }
    avg[size] = times.reduce((s, t) => s + t, 0) / runs;
    console.log(`[compare] e5-${size} texto largo: avg ${avg[size].toFixed(0)} ms (${times.map((t) => `${t.toFixed(0)} ms`).join(", ")})`);
  }
  assert.ok(avg.large > avg.small, `esperaba large > small: large=${avg.large.toFixed(0)}ms small=${avg.small.toFixed(0)}ms`);
  assert.ok(avg.large < 20_000, `large demasiado lento: ${avg.large.toFixed(0)}ms`);
  console.log(`[compare] ratio large/small ≈ ${(avg.large / avg.small).toFixed(1)}x`);
});
