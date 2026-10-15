/**
 * Compara la estructura ACTUAL (dos corridas del mismo modelo sobre el mismo
 * texto: una para el vector pooled y otra para los vectores por token) contra la
 * estructura ALTERNA (una sola corrida de la que salen ambos).
 *
 * Verifica dos cosas por separado:
 *   1. Equivalencia numérica — el pooled y los tokens deben ser idénticos, o la
 *      alternativa no sirve (cambiaría los scores del juicio).
 *   2. Coste — milisegundos de cada estructura sobre secuencias corta y larga.
 *
 * No quita ningún modelo: el bi-encoder, el cross-encoder y el MaxSim siguen
 * consumiendo exactamente las mismas señales.
 *
 * Uso: tsx bench/pasada-unica-vs-doble.ts
 */
import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { onnxModelDir, onnxWeightsFileBySize, onnxPooling, onnxEmbedDim, onnxMaxTokens } from "../src/shared/constants/predict/embedding.constants";

const SIZE = "gte" as const;
const DIR = join("models", onnxModelDir[SIZE]);
const DIM = onnxEmbedDim[SIZE];

function normalizeL2(v: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n);
  if (n > 1e-9) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

function meanPool(data: Float32Array, seq: number, dim: number, mask: number[]): Float32Array {
  const out = new Float32Array(dim);
  let valid = 0;
  for (let s = 0; s < seq; s++) {
    if (mask[s] === 0) continue;
    valid++;
    for (let d = 0; d < dim; d++) out[d] += data[s * dim + d];
  }
  if (valid > 1) for (let d = 0; d < dim; d++) out[d] /= valid;
  return out;
}

/** Pooling idéntico al de embedding.service.runModel (CLS para gte, mean para e5). */
function pool(data: Float32Array, dims: readonly number[], mask: number[]): Float32Array {
  const vec =
    dims.length === 3
      ? onnxPooling[SIZE] === "cls"
        ? data.slice(0, dims[2])
        : meanPool(data, dims[1], dims[2], mask)
      : data;
  return normalizeL2(vec);
}

/** Filtrado de tokens idéntico al de embedding.service.runModelTokens. */
function tokenVectors(data: Float32Array, dims: readonly number[], mask: number[]): Float32Array[] {
  if (dims.length !== 3) return [];
  const seq = dims[1];
  const dim = dims[2];
  const out: Float32Array[] = [];
  for (let s = 0; s < seq; s++) {
    if (mask[s] === 0) continue;
    out.push(normalizeL2(data.slice(s * dim, (s + 1) * dim)));
  }
  if (out.length > 2) {
    out.shift();
    out.pop();
  }
  return out;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) d += a[i] * b[i];
  return d;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main() {
  console.log("=== Estructura ACTUAL (2 corridas) vs ALTERNA (1 corrida) ===");
  console.log(`modelo: ${DIR}/${onnxWeightsFileBySize[SIZE]}  dim=${DIM}\n`);

  const tokenizer = new Tokenizer(
    JSON.parse(readFileSync(join(DIR, "tokenizer.json"), "utf8")),
    JSON.parse(readFileSync(join(DIR, "tokenizer_config.json"), "utf8")),
  );
  const session = await InferenceSession.create(join(DIR, onnxWeightsFileBySize[SIZE]), {
    executionProviders: ["cpu"],
  });
  console.log(`outputs del modelo: ${session.outputNames.join(", ")}`);
  console.log(`RSS tras cargar   : ${(process.memoryUsage().rss / 1048576).toFixed(0)} MB\n`);

  const feedsFor = (text: string) => {
    const enc = tokenizer.encode(text);
    const ids = enc.ids.slice(0, onnxMaxTokens);
    const mask = (enc.attention_mask ?? ids.map(() => 1)).slice(0, onnxMaxTokens);
    const len = ids.length;
    const inputIds = new BigInt64Array(len);
    const attentionMask = new BigInt64Array(len);
    const tokenTypeIds = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(ids[i]);
      attentionMask[i] = BigInt(mask[i]);
      tokenTypeIds[i] = 0n;
    }
    const feeds: Record<string, Tensor> = {};
    if (session.inputNames.includes("input_ids")) feeds.input_ids = new Tensor("int64", inputIds, [1, len]);
    if (session.inputNames.includes("attention_mask")) feeds.attention_mask = new Tensor("int64", attentionMask, [1, len]);
    if (session.inputNames.includes("token_type_ids")) feeds.token_type_ids = new Tensor("int64", tokenTypeIds, [1, len]);
    if (session.inputNames.includes("position_ids")) {
      const pos = new BigInt64Array(len);
      for (let i = 0; i < len; i++) pos[i] = BigInt(i);
      feeds.position_ids = new Tensor("int64", pos, [1, len]);
    }
    return { feeds, mask, len };
  };

  /** Estructura ACTUAL: reproduce embedding.service runRaw con wantTokens=false y =true. */
  async function actual(text: string) {
    const { feeds, mask } = feedsFor(text);
    const a = await session.run(feeds);
    const pooledOut = a["sentence_embedding"] ?? a["last_hidden_state"] ?? a["token_embeddings"];
    const b = await session.run(feeds);
    const tokOut = b["token_embeddings"] ?? b["last_hidden_state"] ?? b["sentence_embedding"];
    return {
      pooled: pool(pooledOut.data as Float32Array, pooledOut.dims, mask),
      tokens: tokenVectors(tokOut.data as Float32Array, tokOut.dims, mask),
    };
  }

  /** Estructura ALTERNA: una corrida, ambos artefactos del mismo forward pass. */
  async function alterna(text: string) {
    const { feeds, mask } = feedsFor(text);
    const r = await session.run(feeds);
    const pooledOut = r["sentence_embedding"] ?? r["last_hidden_state"] ?? r["token_embeddings"];
    const tokOut = r["token_embeddings"] ?? r["last_hidden_state"] ?? r["sentence_embedding"];
    return {
      pooled: pool(pooledOut.data as Float32Array, pooledOut.dims, mask),
      tokens: tokenVectors(tokOut.data as Float32Array, tokOut.dims, mask),
    };
  }

  const CASOS = [
    { etiqueta: "corta", text: "envía el resumen de ventas por correo a finanzas" },
    {
      etiqueta: "larga-512",
      text:
        "necesito enviar el resumen de ventas por correo a finanzas " +
        "cuéntame también el detalle de cada paso del proceso y sus dependencias. ".repeat(20),
    },
  ];

  for (const caso of CASOS) {
    const A = await actual(caso.text);
    const B = await alterna(caso.text);

    const cos = cosine(A.pooled, B.pooled);
    const diff = maxAbsDiff(A.pooled, B.pooled);
    const mismosTokens = A.tokens.length === B.tokens.length;
    let diffTokens = 0;
    if (mismosTokens) {
      for (let i = 0; i < A.tokens.length; i++) diffTokens = Math.max(diffTokens, maxAbsDiff(A.tokens[i], B.tokens[i]));
    }

    console.log(`--- caso ${caso.etiqueta} (${caso.text.length} chars) ---`);
    console.log(`  pooled  : coseno=${cos.toFixed(8)}  max|Δ|=${diff.toExponential(2)}  dim=${A.pooled.length}`);
    console.log(`  tokens  : actual=${A.tokens.length} alterna=${B.tokens.length}  max|Δ|=${diffTokens.toExponential(2)}`);
    console.log(`  idéntico: ${cos > 0.999999 && diff < 1e-5 && mismosTokens && diffTokens < 1e-5 ? "SÍ" : "NO"}`);

    // Calentamiento y medición.
    for (let i = 0; i < 2; i++) {
      await actual(caso.text);
      await alterna(caso.text);
    }
    const N = 3;
    let msActual = 0;
    let msAlterna = 0;
    for (let i = 0; i < N; i++) {
      let t0 = performance.now();
      await actual(caso.text);
      msActual += performance.now() - t0;
      t0 = performance.now();
      await alterna(caso.text);
      msAlterna += performance.now() - t0;
    }
    console.log(`  ACTUAL  (2 corridas): ${(msActual / N).toFixed(0)} ms`);
    console.log(`  ALTERNA (1 corrida) : ${(msAlterna / N).toFixed(0)} ms`);
    console.log(`  ahorro              : ${(((msActual - msAlterna) / msActual) * 100).toFixed(0)}%\n`);
  }

  console.log(`RSS final: ${(process.memoryUsage().rss / 1048576).toFixed(0)} MB`);
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
