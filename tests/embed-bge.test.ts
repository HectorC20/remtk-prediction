/**
 * Evaluación real de BGE-M3 int8 ONNX (gpahal/bge-m3-onnx-int8, CPU) frente a
 * e5-small: dim (1024), latencia (corto/largo) y separación semántica.
 * Modelo en ./models/multilingual-bge-m3-int8-onnx. Corre con: pnpm test:bge
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";

const DIR = "models/multilingual-bge-m3-int8-onnx";
const ONNX = "model_quantized.onnx";
const available = existsSync(join(DIR, ONNX)) && existsSync(join(DIR, "tokenizer.json"));

const TEXT_SIM_A = "envía un correo electrónico al equipo";
const TEXT_SIM_B = "enviar email a los colaboradores del grupo";
const TEXT_DIF = "genera una imagen de un paisaje nevado";
const TEXT_LONG = Array.from(
  { length: 200 },
  () => "enviar correo electrónico a los colaboradores del proyecto",
).join(" ");

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

function toFloat32(embedding: Float32Array): number[] {
  return Array.from(embedding);
}

function l2Normalize(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / n);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

/** Ejecuta el ONNX y reduce a un vector [dim]: prioriza salida 2D; si la
 *  salida es 3D (last_hidden_state) hace mean-pooling ponderado por mask. */
async function embedOnce(
  session: ort.InferenceSession,
  text: string,
): Promise<{ vector: number[]; dim: number }> {
  const encoded = encodeText(text);
  const ids = encoded.ids;
  const mask = ids.map(() => 1);
  const len = ids.length;
  const feed: Record<string, ort.Tensor> = {};
  for (const name of session.inputNames) {
    if (name === "input_ids") {
      feed[name] = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, len]);
    } else if (name === "attention_mask") {
      feed[name] = new ort.Tensor("int64", BigInt64Array.from(mask.map(BigInt)), [1, len]);
    } else if (name === "token_type_ids") {
      feed[name] = new ort.Tensor("int64", new BigInt64Array(len), [1, len]);
    }
  }
  const out = await session.run(feed);
  const names = session.outputNames;
  // Preferencia: salida 2D ya es "sentence_embedding".
  let pooled: Float32Array | null = null;
  let dim = 0;
  for (const n of names) {
    const t = out[n];
    if (t.dims.length === 2) {
      pooled = t.data as Float32Array;
      dim = t.dims[1] ?? 0;
      break;
    }
  }
  if (pooled == null) {
    // last_hidden_state [1, seq, hidden] → mean-pooling sobre tokens válidos.
    for (const n of names) {
      const t = out[n];
      if (t.dims.length === 3) {
        const [b, seq, h] = t.dims as [number, number, number];
        const flat = t.data as Float32Array;
        const sums = new Float64Array(h);
        let count = 0;
        for (let s = 0; s < seq; s++) {
          if (!mask[s] && s >= mask.length) break;
          count++;
          for (let j = 0; j < h; j++) sums[j] += flat[s * h + j];
        }
        pooled = Float32Array.from(sums, (x) => x / (count || 1));
        dim = h;
        break;
      }
    }
  }
  assert.ok(pooled, `sin salida usable en ${names.join(", ")}`);
  return { vector: l2Normalize(toFloat32(pooled)), dim };
}

let tokenizer: Tokenizer | null = null;
function encodeText(text: string): { ids: number[] } {
  if (!tokenizer) {
    const json = JSON.parse(readFileSync(join(DIR, "tokenizer.json"), "utf8"));
    const config = JSON.parse(readFileSync(join(DIR, "tokenizer_config.json"), "utf8"));
    tokenizer = new Tokenizer(json, config);
  }
  const enc = tokenizer.encode(text);
  return { ids: enc.ids.slice(0, 512) };
}

test("BGE-M3 int8 ONNX: carga y dim=1024", { skip: !available, timeout: 300_000 }, async () => {
  const mb = (statSync(join(DIR, ONNX)).size / 1024 / 1024).toFixed(0);
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(join(DIR, ONNX), { executionProviders: ["cpu"] });
  console.log(`[bge] model_quantized.onnx: ${mb} MB — carga ${(performance.now() - t0).toFixed(0)} ms`);
  const { dim } = await embedOnce(session, "hola mundo");
  assert.equal(dim, 1024, `esperaba dim 1024, obtuve ${dim}`);
  console.log(`[bge] dim=${dim}`);
});

test("BGE-M3 int8 ONNX: latencia corto y largo (CPU)", { skip: !available, timeout: 300_000 }, async () => {
  const session = await ort.InferenceSession.create(join(DIR, ONNX), { executionProviders: ["cpu"] });
  const short: number[] = [];
  const long: number[] = [];
  await embedOnce(session, TEXT_SIM_A);
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await embedOnce(session, TEXT_SIM_A);
    short.push(performance.now() - t0);
  }
  await embedOnce(session, TEXT_LONG);
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await embedOnce(session, TEXT_LONG);
    long.push(performance.now() - t0);
  }
  const avgShort = short.reduce((s, t) => s + t, 0) / short.length;
  const avgLong = long.reduce((s, t) => s + t, 0) / long.length;
  console.log(`[bge] corto (~16 tok): avg ${avgShort.toFixed(0)} ms (${short.map((t) => `${t.toFixed(0)}`).join(", ")})`);
  console.log(`[bge] largo (512 tok): avg ${avgLong.toFixed(0)} ms (${long.map((t) => `${t.toFixed(0)}`).join(", ")})`);
  assert.ok(avgLong < 20_000, `largo demasiado lento: ${avgLong.toFixed(0)}ms`);
});

test("separación semántica BGE-M3 vs e5-small (mismo corpus)", { skip: !available, timeout: 300_000 }, async () => {
  const session = await ort.InferenceSession.create(join(DIR, ONNX), { executionProviders: ["cpu"] });
  const [a, b, c] = await Promise.all([
    embedOnce(session, TEXT_SIM_A).then((r) => r.vector),
    embedOnce(session, TEXT_SIM_B).then((r) => r.vector),
    embedOnce(session, TEXT_DIF).then((r) => r.vector),
  ]);
  const sim = cosine(a, b);
  const dif = cosine(a, c);
  const marginBge = sim - dif;
  console.log(`[bge] similar=${sim.toFixed(3)} different=${dif.toFixed(3)} margen=${marginBge.toFixed(3)}`);

  const engine = new EmbeddingEngineService(CFG);
  const [ea, eb, ec] = await Promise.all([
    engine.embed(TEXT_SIM_A).then((r) => r.embedding),
    engine.embed(TEXT_SIM_B).then((r) => r.embedding),
    engine.embed(TEXT_DIF).then((r) => r.embedding),
  ]);
  const simS = EmbeddingEngineService.cosine(ea, eb);
  const difS = EmbeddingEngineService.cosine(ea, ec);
  const marginSmall = simS - difS;
  console.log(`[bge] e5-small similar=${simS.toFixed(3)} different=${difS.toFixed(3)} margen=${marginSmall.toFixed(3)}`);

  assert.ok(sim > dif, `BGE-M3: similar=${sim.toFixed(3)} debe superar different=${dif.toFixed(3)}`);
  assert.ok(marginBge >= marginSmall - 0.05, `margen BGE (${marginBge.toFixed(3)}) por debajo de e5-small (${marginSmall.toFixed(3)})`);
});
