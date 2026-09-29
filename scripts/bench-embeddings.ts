/**
 * Benchmark A/B de modelos de EMBEDDINGS para comprensión de intención
 * (selección de herramienta) sobre el catálogo REAL de MiTumbes + ruido
 * (TOTAL_ACTIVE_TOOLS = 232 tools) en CPU/ONNX. Cuantizados, ≤10 GB, sin GPU.
 *
 * Uso:
 *   npx tsx scripts/bench-embeddings.ts            # descarga (si falta) + mide
 *   npx tsx scripts/bench-embeddings.ts --ensure   # solo descarga
 *
 * Cada modelo declara su pooling y su formato de texto (query/doc), porque
 * difieren: e5 => mean + "query:/passage:"; bge-m3/gte/arctic => CLS sin
 * prefijo; Qwen3 => último token + instrucción; EmbeddingGemma => mean +
 * "task: ... | query:" / "title: none | text:".
 */
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { performance } from "node:perf_hooks";
import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-real-catalog";

const HF = "https://huggingface.co";
const MAX_TOKENS = 512;
const DOC_CHARS = 480;

type Pooling = "mean" | "cls" | "last";
interface FileSpec {
  url: string;
  dest: string;
}
interface ModelCfg {
  name: string;
  dir: string;
  onnx: string;
  pooling: Pooling;
  dim: number;
  qFmt: (q: string) => string;
  dFmt: (d: string) => string;
  kvEmpty?: [number, number, number, number];
  files: FileSpec[];
}

const E5_TASK = "Given a web search query, retrieve relevant passages that answer the query";
const E5I_TASK = "Given a web search query, retrieve relevant passages that answer the query";

const MODELS: ModelCfg[] = [
  {
    name: "e5-small",
    dir: "multilingual-e5-small-onnx",
    onnx: "model.onnx",
    pooling: "mean",
    dim: 384,
    qFmt: (q) => `query: ${q}`,
    dFmt: (d) => `passage: ${d}`,
    files: [],
  },
  {
    name: "e5-large",
    dir: "multilingual-e5-large-onnx",
    onnx: "model.onnx",
    pooling: "mean",
    dim: 1024,
    qFmt: (q) => `query: ${q}`,
    dFmt: (d) => `passage: ${d}`,
    files: [],
  },
  {
    name: "bge-m3",
    dir: "multilingual-bge-m3-int8-onnx",
    onnx: "model_quantized.onnx",
    pooling: "cls",
    dim: 1024,
    qFmt: (q) => q,
    dFmt: (d) => d,
    files: [],
  },
  {
    name: "qwen3-0.6b",
    dir: "qwen3-embedding-0.6b-onnx",
    onnx: "model.onnx",
    pooling: "last",
    dim: 1024,
    qFmt: (q) => `Instruct: ${E5_TASK}\nQuery: ${q}`,
    dFmt: (d) => `passage: ${d}`,
    kvEmpty: [1, 8, 0, 128],
    files: [],
  },
  {
    name: "e5-large-instruct",
    dir: "multilingual-e5-large-instruct-onnx",
    onnx: "model.onnx",
    pooling: "mean",
    dim: 1024,
    qFmt: (q) => `Instruct: ${E5I_TASK}\nQuery: ${q}`,
    dFmt: (d) => d,
    files: [
      `${HF}/intfloat/multilingual-e5-large-instruct/resolve/main/onnx/model.onnx`,
      `${HF}/intfloat/multilingual-e5-large-instruct/resolve/main/onnx/model.onnx_data`,
      `${HF}/intfloat/multilingual-e5-large-instruct/resolve/main/onnx/tokenizer.json`,
      `${HF}/intfloat/multilingual-e5-large-instruct/resolve/main/onnx/tokenizer_config.json`,
    ].map((url) => ({ url, dest: baseName(url) })),
  },
  {
    name: "embeddinggemma-300m",
    dir: "embeddinggemma-300m-onnx",
    onnx: "model_quantized.onnx",
    pooling: "mean",
    dim: 768,
    qFmt: (q) => `task: search result | query: ${q}`,
    dFmt: (d) => `title: none | text: ${d}`,
    files: [
      { url: `${HF}/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_quantized.onnx`, dest: "model_quantized.onnx" },
      { url: `${HF}/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_quantized.onnx_data`, dest: "model_quantized.onnx_data" },
      { url: `${HF}/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json`, dest: "tokenizer.json" },
      { url: `${HF}/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer_config.json`, dest: "tokenizer_config.json" },
      { url: `${HF}/onnx-community/embeddinggemma-300m-ONNX/resolve/main/config.json`, dest: "config.json" },
    ],
  },
  {
    name: "gte-multilingual-base",
    dir: "gte-multilingual-base-onnx",
    onnx: "model_int8.onnx",
    pooling: "cls",
    dim: 768,
    qFmt: (q) => q,
    dFmt: (d) => d,
    files: [
      `${HF}/onnx-community/gte-multilingual-base/resolve/main/onnx/model_int8.onnx`,
      `${HF}/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json`,
      `${HF}/onnx-community/gte-multilingual-base/resolve/main/tokenizer_config.json`,
    ].map((url) => ({ url, dest: baseName(url) })),
  },
  {
    name: "arctic-embed-l-v2.0",
    dir: "snowflake-arctic-embed-l-v2.0-onnx",
    onnx: "model_int8.onnx",
    pooling: "cls",
    dim: 1024,
    qFmt: (q) => q,
    dFmt: (d) => d,
    files: [
      `${HF}/Snowflake/snowflake-arctic-embed-l-v2.0/resolve/main/onnx/model_int8.onnx`,
      `${HF}/Snowflake/snowflake-arctic-embed-l-v2.0/resolve/main/tokenizer.json`,
      `${HF}/Snowflake/snowflake-arctic-embed-l-v2.0/resolve/main/tokenizer_config.json`,
    ].map((url) => ({ url, dest: baseName(url) })),
  },
];

function baseName(url: string): string {
  return url.split("/").pop() as string;
}

// ── Descarga idempotente ─────────────────────────────────────────────────────
async function download(url: string, dest: string): Promise<void> {
  const tmp = `${dest}.tmp`;
  const res = await fetch(url, { redirect: "follow", signal: AbortSignal.timeout(60 * 60 * 1000) });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} para ${url}`);
  await pipeline(Readable.fromWeb(res.body as any), createWriteStream(tmp));
  if (statSync(tmp).size === 0) throw new Error(`archivo vacío: ${dest}`);
  renameSync(tmp, dest);
}

async function ensure(m: ModelCfg): Promise<boolean> {
  const target = join("models", m.dir);
  mkdirSync(target, { recursive: true });
  let ok = true;
  for (const f of m.files) {
    const out = join(target, f.dest);
    if (existsSync(out) && statSync(out).size > 0) continue;
    try {
      const mb = "?";
      console.log(`[dl] ${m.name}: ${f.dest} (${mb})`);
      await download(f.url, out);
    } catch (e) {
      rmSync(`${out}.tmp`, { force: true });
      console.log(`[dl] WARN ${m.name}: ${(e as Error).message}`);
      ok = false;
    }
  }
  if (!existsSync(join(target, m.onnx))) {
    console.log(`[dl] ${m.name}: falta ${m.onnx} — se omitirá`);
    return false;
  }
  return ok;
}

// ── Carga y embedding genérico ───────────────────────────────────────────────
interface Loaded {
  session: ort.InferenceSession;
  tok: Tokenizer;
}
const cache = new Map<string, Loaded>();

function bigint(n: number[]): BigInt64Array {
  return BigInt64Array.from(n.map((x) => BigInt(x)));
}

async function load(m: ModelCfg): Promise<Loaded> {
  const hit = cache.get(m.name);
  if (hit) return hit;
  const dir = join("models", m.dir);
  const session = await ort.InferenceSession.create(join(dir, m.onnx), { executionProviders: ["cpu"], graphOptimizationLevel: "all" });
  const j = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8"));
  const cfgPath = join(dir, "tokenizer_config.json");
  const c = existsSync(cfgPath) ? JSON.parse(readFileSync(cfgPath, "utf8")) : undefined;
  const tok = c ? new Tokenizer(j, c) : new Tokenizer(j);
  const l = { session, tok };
  cache.set(m.name, l);
  return l;
}

async function embed(m: ModelCfg, text: string): Promise<number[]> {
  const { session, tok } = await load(m);
  let ids = tok.encode(text).ids.slice(0, MAX_TOKENS);
  if (ids.length === 0) ids = [tok.encode("a").ids[0] ?? 0];
  const len = ids.length;
  const names = session.inputNames;
  const feeds: Record<string, ort.Tensor> = {};
  if (names.includes("input_ids")) feeds.input_ids = new ort.Tensor("int64", bigint(ids), [1, len]);
  if (names.includes("attention_mask")) feeds.attention_mask = new ort.Tensor("int64", new BigInt64Array(len).fill(1n), [1, len]);
  if (names.includes("token_type_ids")) feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(len).fill(0n), [1, len]);
  if (names.includes("position_ids")) feeds.position_ids = new ort.Tensor("int64", bigint(Array.from({ length: len }, (_, i) => i)), [1, len]);
  if (m.kvEmpty) {
    for (let i = 0; i < 64; i++) {
      const k = `past_key_values.${i}.key`;
      if (!names.includes(k)) break;
      feeds[k] = new ort.Tensor("float32", new Float32Array(0), m.kvEmpty);
      feeds[`past_key_values.${i}.value`] = new ort.Tensor("float32", new Float32Array(0), m.kvEmpty);
    }
  }

  const out = await session.run(feeds);
  const tensor = out["sentence_embedding"] ?? out["last_hidden_state"] ?? out[Object.keys(out)[0]];
  const dims = tensor.dims as number[];
  const flat = tensor.data as Float32Array;
  let vec: number[];
  if (dims.length === 2) {
    vec = Array.from(flat.subarray(0, dims[1]));
  } else {
    const [, seq, dim] = dims as [number, number, number];
    if (m.pooling === "cls") {
      vec = Array.from(flat.subarray(0, dim));
    } else if (m.pooling === "last") {
      const off = (seq - 1) * dim;
      vec = Array.from(flat.subarray(off, off + dim));
    } else {
      vec = new Array(dim).fill(0);
      for (let t = 0; t < seq; t++) for (let d = 0; d < dim; d++) vec[d] += flat[t * dim + d];
      for (let d = 0; d < dim; d++) vec[d] /= seq;
    }
  }
  let n = 0;
  for (const x of vec) n += x * x;
  n = Math.sqrt(n) || 1;
  return vec.map((x) => x / n);
}

// ── Corpus y consultas reales ────────────────────────────────────────────────
const CATALOG = TOTAL_ACTIVE_TOOLS.map((t) => ({
  id: t.name,
  text: `${t.name}. ${(t.description ?? "").replace(/\s+/g, " ")}`.slice(0, DOC_CHARS),
}));
const INDEX_OF = new Map(CATALOG.map((c, i) => [c.id, i]));

const QUERIES: Array<{ q: string; tool: string }> = [
  { q: "lista las diapositivas activas del hero de la portada", tool: "mitumbes_hero_obtener" },
  { q: "muéstrame la configuración actual de las diapositivas del hero de la portada", tool: "mitumbes_hero_obtener" },
  { q: "obtén todas las diapositivas del hero con su orden y estado", tool: "mitumbes_hero_obtener" },
  { q: "cambia las diapositivas que se muestran en la portada principal", tool: "mitumbes_hero_configurar" },
  { q: "configura las diapositivas del hero de la portada con nuevos ítems y lugares", tool: "mitumbes_hero_configurar" },
  { q: "quiero mostrar otros eventos destacados en el carrusel de la página de eventos", tool: "mitumbes_evento_hero_configurar" },
  { q: "programa que un evento destacado solo aparezca los fines de semana", tool: "mitumbes_evento_hero_programar" },
  { q: "añade una nueva sección al menú de navegación de arriba", tool: "mitumbes_navbar_configurar" },
  { q: "devuelve el menú de navegación a su estado original", tool: "mitumbes_navbar_restablecer" },
  { q: "crea un lugar turístico nuevo en el sistema", tool: "mitumbes_lugar_crear" },
  { q: "lista todos los lugares registrados", tool: "mitumbes_lugar_listar" },
  { q: "elimina definitivamente un aliado del directorio", tool: "mitumbes_aliado_eliminar" },
  { q: "fija los mejores ítems al inicio de una categoría", tool: "mitumbes_categoria_ordenar_items" },
  { q: "consulta cuántos clics y visitas han tenido los anuncios publicitarios", tool: "mitumbes_publicidad_estadisticas" },
  { q: "adjunta una imagen a la ficha de un ítem", tool: "mitumbes_item_imagen_adjuntar" },
];

function cos(a: number[], b: number[]): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  return d;
}

function rankOf(qv: number[], docs: number[][], target: number): number {
  let best = -Infinity;
  for (let i = 0; i < docs.length; i++) if (i === target) best = cos(qv, docs[i]);
  let better = 0;
  for (let i = 0; i < docs.length; i++) if (i !== target && cos(qv, docs[i]) > best) better++;
  return better + 1;
}

interface Result {
  name: string;
  dim: number;
  top1: number;
  top3: number;
  top5: number;
  mrr: number;
  heroOk: number;
  heroTot: number;
  indexS: number;
  qMs: number;
  sizeMb: number;
}

async function bench(m: ModelCfg): Promise<Result | null> {
  if (!(await ensure(m))) return null;
  const t0 = performance.now();
  const vec = await embed(m, "prueba de dimensiones");
  const loadMs = performance.now() - t0;
  if (vec.length !== m.dim) console.log(`[warn] ${m.name}: dim ${vec.length} ≠ config ${m.dim}`);

  const tIdx = performance.now();
  const docs: number[][] = [];
  for (const c of CATALOG) docs.push(await embed(m, m.dFmt(c.text)));
  const indexS = (performance.now() - tIdx) / 1000;

  const ranks: number[] = [];
  const qTimes: number[] = [];
  let heroOk = 0;
  let heroTot = 0;
  const detail: string[] = [];
  for (const { q, tool } of QUERIES) {
    const tidx = INDEX_OF.get(tool);
    if (tidx === undefined) throw new Error(`tool fuera de catálogo: ${tool}`);
    const tq = performance.now();
    const qv = await embed(m, m.qFmt(q));
    qTimes.push(performance.now() - tq);
    const r = rankOf(qv, docs, tidx);
    ranks.push(r);
    const isHero = tool.includes("hero");
    if (isHero) {
      heroTot++;
      if (r <= 5) heroOk++;
    }
    detail.push(`    #${String(r).padStart(3)}  ${tool}`);
  }
  const top1 = ranks.filter((r) => r === 1).length / ranks.length;
  const top3 = ranks.filter((r) => r <= 3).length / ranks.length;
  const top5 = ranks.filter((r) => r <= 5).length / ranks.length;
  const mrr = ranks.reduce((s, r) => s + 1 / r, 0) / ranks.length;
  const sizeMb = statSync(join("models", m.dir, m.onnx)).size / 1024 / 1024;

  console.log(`\n── ${m.name} ─────────────────────────────────────────`);
  console.log(`  dim=${vec.length} load=${loadMs.toFixed(0)}ms index(232)=${indexS.toFixed(1)}s onnx=${sizeMb.toFixed(0)}MB`);
  console.log(`  ranks por consulta:\n${detail.join("\n")}`);
  console.log(
    `  top1=${(top1 * 100).toFixed(0)}% top3=${(top3 * 100).toFixed(0)}% top5=${(top5 * 100).toFixed(0)}% MRR=${mrr.toFixed(3)}`,
  );
  console.log(`  hero en top5: ${heroOk}/${heroTot} · latencia query avg=${(qTimes.reduce((s, t) => s + t, 0) / qTimes.length).toFixed(0)}ms`);

  return { name: m.name, dim: vec.length, top1, top3, top5, mrr, heroOk, heroTot, indexS, qMs: qTimes.reduce((s, t) => s + t, 0) / qTimes.length, sizeMb };
}

(async () => {
  const ensureOnly = process.argv.includes("--ensure");
  console.log(`Modelos a evaluar: ${MODELS.length} · catálogo ${CATALOG.length} tools · ${QUERIES.length} consultas\n`);
  if (ensureOnly) {
    for (const m of MODELS) await ensure(m);
    console.log("Descarga completada.");
    return;
  }
  const results: Result[] = [];
  for (const m of MODELS) {
    try {
      const r = await bench(m);
      if (r) results.push(r);
    } catch (e) {
      console.log(`\n[ERROR] ${m.name}: ${(e as Error).message}`);
    }
  }
  results.sort((a, b) => b.mrr - a.mrr);
  console.log("\n════════ RANKING (por MRR) ════════");
  console.log("  #  modelo                 top1  top3  top5   MRR   hero  idx(s)  q(ms)  MB");
  results.forEach((r, i) => {
    console.log(
      `  ${String(i + 1).padStart(2)}  ${r.name.padEnd(22)} ${(r.top1 * 100).toFixed(0).padStart(4)}% ${(r.top3 * 100).toFixed(0).padStart(4)}% ${(r.top5 * 100).toFixed(0).padStart(4)}% ${r.mrr.toFixed(3)}  ${r.heroOk}/${r.heroTot}  ${r.indexS.toFixed(1).padStart(5)}  ${r.qMs.toFixed(0).padStart(5)}  ${r.sizeMb.toFixed(0)}`,
    );
  });
})().catch((e) => {
  console.error(String((e && e.stack) || e));
  process.exit(1);
});
