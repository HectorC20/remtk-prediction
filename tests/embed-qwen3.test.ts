/**
 * A/B real: Qwen3-Embedding-0.6B (ONNX int8, CPU) frente a e5-large para
 * COMPRENSIÓN DE INTENCIÓN. Carga ambos modelos y mide dim/normalización,
 * latencia, separación semántica y —lo central— la recuperación (retrieval)
 * del tool correcto sobre el catálogo REAL de MiTumbes (98 tools).
 * Modelo en ./models/qwen3-embedding-0.6b-onnx. Corre con: pnpm test:qwen3
 *
 * Diferencias de integración de Qwen3 respecto a e5 (verificado empíricamente):
 *  - Exige `position_ids` (0..n-1) y NO acepta `token_type_ids`.
 *  - El export incluye caché KV: hay que enviar 28 pares `past_key_values.N.*`
 *    vacíos con forma [1, 8, 0, 128] (si no, ORT falla por input faltante).
 *  - Pooling = ÚLTIMO token ([EOS]) de `last_hidden_state`, no mean-pooling.
 *  - Las consultas llevan instrucción en inglés: "Instruct: …\nQuery: …". El
 *    texto de instrucción importa: la genérica de búsqueda web ("Given a web
 *    search query, retrieve relevant passages that answer the query") rinde
 *    mucho mejor que una instrucción de tarea larga (diagnóstico previo).
 *  - Los docs llevan prefijo "passage: " (mejora medible en este catálogo).
 *  - tokenizer_config ya aplica el sufijo `<|endoftext|>` (no añadirlo a mano).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { juicioConfigDefaults } from "../src/shared/constants/juicio";
import { MITUMBES_VERBATIM_TOOLS } from "./fixtures/mitumbes-real-catalog";

const DIR = "models/qwen3-embedding-0.6b-onnx";
const ONNX = "model.onnx";
const available = existsSync(join(DIR, ONNX)) && existsSync(join(DIR, "tokenizer.json"));

// Geometría del export de Qwen3-0.6B (num_hidden_layers=28, 8 kv-heads, head_dim=128).
const NUM_LAYERS = 28;
const NUM_KV_HEADS = 8;
const HEAD_DIM = 128;
const MAX_TOKENS = 512;
const INSTRUCTION =
  "Given a web search query, retrieve relevant passages that answer the query";

const CFG = {
  portPredict: 0,
  portEmbed: 0,
  portQdrant: 0,
  onnxEnabled: true,
  onnxModelsPath: "./models",
  onnxModelSize: "large" as const,
  adaptiveMinTools: 2,
  adaptiveMaxTools: 5,
  adaptiveGapThreshold: 0.15,
  adaptiveMinScore: 0,
  keywordBoost: 0.15,
  nameAffinityBoost: 0.1,
  familyGatePenalty: 0.2,
  keywordTopK: 20,
  recallLimit: 50,
  maxOutputTools: 50,
  maxCategories: 60,
  qdrantEnabled: false,
  qdrantUrl: "http://localhost:6333",
  qdrantApiKey: "",
  toolsCollection: "mcp_tools",
  keywordsCollection: "tool_keywords",
  synonymsCollection: "query_synonyms",
  memoriesCollection: "contextual_memories",
  learnEnabled: false,
  learnWeight: 0.25,
  learnEta: 0.5,
  learnNegativeGamma: 0.15,
  learnDecayLambda: 0.02,
  learnMinEvents: 3,
  learnSeedWeight: 0.3,
  learnMaxTermsPerTool: 64,
  learnTermMinWeight: 0.05,
  learnMaxPostings: 200,
  learnPersist: false,
  predictWarmupWaitMs: 0,
  ...juicioConfigDefaults,
};

const TEXT_SIM_A = "envía un correo electrónico al equipo";
const TEXT_SIM_B = "enviar email a los colaboradores del grupo";
const TEXT_DIF = "genera una imagen de un paisaje nevado";
const TEXT_LONG = Array.from(
  { length: 200 },
  () => "enviar correo electrónico a los colaboradores del proyecto",
).join(" ");

function l2(v: number[]): number[] {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// ── Cargador Qwen3 (inputs correctos + last-token pooling) ───────────────────

let qwenSession: ort.InferenceSession | null = null;
let qwenTokenizer: Tokenizer | null = null;

async function qwen(): Promise<{ session: ort.InferenceSession; tokenizer: Tokenizer }> {
  if (!qwenSession) {
    qwenSession = await ort.InferenceSession.create(join(DIR, ONNX), { executionProviders: ["cpu"] });
  }
  if (!qwenTokenizer) {
    const json = JSON.parse(readFileSync(join(DIR, "tokenizer.json"), "utf8"));
    const config = JSON.parse(readFileSync(join(DIR, "tokenizer_config.json"), "utf8"));
    qwenTokenizer = new Tokenizer(json, config);
  }
  return { session: qwenSession, tokenizer: qwenTokenizer };
}

/** Embedding Qwen3: instrucción oficial (solo queries) + prefijo "passage: "
 *  en docs + position_ids + past-KV vacío + pooling del último token ([EOS]). */
async function qwenEmbed(text: string, isQuery: boolean): Promise<number[]> {
  const { session, tokenizer } = await qwen();
  const full = isQuery ? `Instruct: ${INSTRUCTION}\nQuery: ${text}` : `passage: ${text}`;
  const ids = tokenizer.encode(full).ids.slice(0, MAX_TOKENS);
  const len = Math.max(1, ids.length);

  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, len]),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(len).fill(1n), [1, len]),
    position_ids: new ort.Tensor(
      "int64",
      BigInt64Array.from({ length: len }, (_, i) => BigInt(i)),
      [1, len],
    ),
  };
  for (let i = 0; i < NUM_LAYERS; i++) {
    feeds[`past_key_values.${i}.key`] = new ort.Tensor("float32", new Float32Array(0), [1, NUM_KV_HEADS, 0, HEAD_DIM]);
    feeds[`past_key_values.${i}.value`] = new ort.Tensor("float32", new Float32Array(0), [1, NUM_KV_HEADS, 0, HEAD_DIM]);
  }

  const out = await session.run(feeds);
  const lh = out["last_hidden_state"];
  assert.ok(lh, `Qwen3 sin last_hidden_state (${Object.keys(out).join(",")})`);
  const [, seq, dim] = lh.dims as [number, number, number];
  const flat = lh.data as Float32Array;
  const off = (seq - 1) * dim; // padding_side=left + EOS final → último token
  return l2(Array.from(flat.subarray(off, off + dim)));
}

// ── Corpus de intención sobre el catálogo real ──────────────────────────────

/** Recorta cada doc a un presupuesto común de caracteres para que ambos modelos
 *  vean exactamente el mismo texto (nombre + inicio de la descripción). */
const DOC_CHARS = 480;

const CATALOG = MITUMBES_VERBATIM_TOOLS.map((t) => ({
  id: t.id,
  name: t.name,
  text: `${t.name}. ${t.description}`.replace(/\s+/g, " ").slice(0, DOC_CHARS),
}));
const INDEX_OF = new Map(CATALOG.map((c, i) => [c.name, i]));

const INTENTS: Array<{ q: string; tool: string }> = [
  { q: "quiero cambiar las diapositivas que aparecen en la portada principal del sitio", tool: "mitumbes_hero_configurar" },
  { q: "muéstrame la configuración actual de las diapositivas del hero de la portada", tool: "mitumbes_hero_obtener" },
  { q: "necesito mostrar otros eventos destacados en el carrusel de la página de eventos", tool: "mitumbes_evento_hero_configurar" },
  { q: "quiero que un evento destacado solo aparezca los fines de semana", tool: "mitumbes_evento_hero_programar" },
  { q: "añade una nueva sección al menú de navegación de arriba", tool: "mitumbes_navbar_configurar" },
  { q: "devuélveme el menú de navegación a su estado original", tool: "mitumbes_navbar_restablecer" },
  { q: "crea un lugar turístico nuevo en el sistema", tool: "mitumbes_lugar_crear" },
  { q: "lista todos los lugares registrados", tool: "mitumbes_lugar_listar" },
  { q: "elimina definitivamente un aliado del directorio", tool: "mitumbes_aliado_eliminar" },
  { q: "fija los mejores ítems al inicio de una categoría", tool: "mitumbes_categoria_ordenar_items" },
  { q: "consulta cuántos clics y visitas han tenido los anuncios publicitarios", tool: "mitumbes_publicidad_estadisticas" },
  { q: "adjunta una imagen a la ficha de un ítem", tool: "mitumbes_item_imagen_adjuntar" },
];

/** Rank (1-based) del tool objetivo para una query y un índice de docs. */
function rankOf(query: number[], docs: number[][], target: number): number {
  let best = -Infinity;
  let better = 0;
  for (let i = 0; i < docs.length; i++) {
    const s = cosine(query, docs[i]);
    if (i === target) best = s;
  }
  for (let i = 0; i < docs.length; i++) {
    if (i === target) continue;
    if (cosine(query, docs[i]) > best) better++;
  }
  return better + 1;
}

function metrics(ranks: number[]): { top1: number; top5: number; mrr: number } {
  const top1 = ranks.filter((r) => r === 1).length / ranks.length;
  const top5 = ranks.filter((r) => r <= 5).length / ranks.length;
  const mrr = ranks.reduce((s, r) => s + 1 / r, 0) / ranks.length;
  return { top1, top5, mrr };
}

// ── Pruebas ──────────────────────────────────────────────────────────────────

test("Qwen3-Embedding-0.6B carga, dim=1024 y vector L2-normalizado", { skip: !available, timeout: 300_000 }, async () => {
  const mb = (statSync(join(DIR, ONNX)).size / 1024 / 1024).toFixed(0);
  const t0 = performance.now();
  const vec = await qwenEmbed("hola mundo", false);
  console.log(`[qwen3] ${ONNX}: ${mb} MB — primer embed ${(performance.now() - t0).toFixed(0)} ms`);
  assert.equal(vec.length, 1024, `esperaba dim 1024, obtuve ${vec.length}`);
  const n = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
  assert.ok(Math.abs(n - 1) < 1e-3, `norma=${n.toFixed(4)}, esperaba 1`);
  console.log(`[qwen3] dim=${vec.length} norma=${n.toFixed(4)}`);
});

test("Qwen3: latencia corto y largo (CPU)", { skip: !available, timeout: 300_000 }, async () => {
  const short: number[] = [];
  const long: number[] = [];
  await qwenEmbed(TEXT_SIM_A, true);
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await qwenEmbed(TEXT_SIM_A, true);
    short.push(performance.now() - t0);
  }
  await qwenEmbed(TEXT_LONG, false);
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await qwenEmbed(TEXT_LONG, false);
    long.push(performance.now() - t0);
  }
  const avgShort = short.reduce((s, t) => s + t, 0) / short.length;
  const avgLong = long.reduce((s, t) => s + t, 0) / long.length;
  console.log(`[qwen3] corto: avg ${avgShort.toFixed(0)} ms (${short.map((t) => t.toFixed(0)).join(", ")})`);
  console.log(`[qwen3] largo: avg ${avgLong.toFixed(0)} ms (${long.map((t) => t.toFixed(0)).join(", ")})`);
  assert.ok(avgLong < 20_000, `largo demasiado lento: ${avgLong.toFixed(0)}ms`);
});

test("separación semántica Qwen3 vs e5-large (mismo corpus)", { skip: !available, timeout: 300_000 }, async () => {
  const [a, b, c] = await Promise.all([
    qwenEmbed(TEXT_SIM_A, true),
    qwenEmbed(TEXT_SIM_B, true),
    qwenEmbed(TEXT_DIF, true),
  ]);
  const simQ = cosine(a, b);
  const difQ = cosine(a, c);
  const marginQ = simQ - difQ;
  console.log(`[qwen3] similar=${simQ.toFixed(3)} different=${difQ.toFixed(3)} margen=${marginQ.toFixed(3)}`);

  const engine = new EmbeddingEngineService(CFG);
  const [ea, eb, ec] = await Promise.all([
    engine.embedQuery(TEXT_SIM_A, "large").then((r) => r.embedding),
    engine.embedQuery(TEXT_SIM_B, "large").then((r) => r.embedding),
    engine.embedQuery(TEXT_DIF, "large").then((r) => r.embedding),
  ]);
  const simL = EmbeddingEngineService.cosine(ea, eb);
  const difL = EmbeddingEngineService.cosine(ea, ec);
  const marginL = simL - difL;
  console.log(`[qwen3] e5-large similar=${simL.toFixed(3)} different=${difL.toFixed(3)} margen=${marginL.toFixed(3)}`);

  assert.ok(simQ > difQ, `Qwen3: similar=${simQ.toFixed(3)} debe superar different=${difQ.toFixed(3)}`);
  assert.ok(marginQ >= marginL - 0.05, `margen Qwen3 (${marginQ.toFixed(3)}) por debajo de e5-large (${marginL.toFixed(3)})`);
});

test("comprensión de intención: recuperación top-K sobre el catálogo real (Qwen3 vs e5-large)", { skip: !available, timeout: 900_000 }, async () => {
  assert.ok(CATALOG.length >= 90, `catálogo inesperadamente corto: ${CATALOG.length}`);
  const targets = INTENTS.map((it) => {
    const idx = INDEX_OF.get(it.tool);
    assert.ok(idx !== undefined, `tool objetivo no está en el catálogo: ${it.tool}`);
    return idx as number;
  });

  const engine = new EmbeddingEngineService(CFG);

  // Docs e5-large (prefijo passage:) y Qwen3 (docs sin instrucción).
  const t0 = performance.now();
  const e5Docs: number[][] = [];
  for (const c of CATALOG) e5Docs.push(Array.from((await engine.embedPassage(c.text, "large")).embedding));
  const e5IndexMs = performance.now() - t0;

  const t1 = performance.now();
  const qDocs: number[][] = [];
  for (const c of CATALOG) qDocs.push(await qwenEmbed(c.text, false));
  const qIndexMs = performance.now() - t1;
  console.log(
    `[intent] indexado ${CATALOG.length} tools → e5-large ${(e5IndexMs / 1000).toFixed(1)}s, qwen3 ${(qIndexMs / 1000).toFixed(1)}s`,
  );

  const e5Ranks: number[] = [];
  const qRanks: number[] = [];
  const detail: string[] = [];
  for (let i = 0; i < INTENTS.length; i++) {
    const { q, tool } = INTENTS[i];
    const eq = (await engine.embedQuery(q, "large")).embedding;
    const qq = await qwenEmbed(q, true);
    const rE5 = rankOf(Array.from(eq), e5Docs, targets[i]);
    const rQ = rankOf(qq, qDocs, targets[i]);
    e5Ranks.push(rE5);
    qRanks.push(rQ);
    detail.push(`  ${rQ === 1 ? "✓" : rQ < rE5 ? "→" : rE5 === 1 ? "✗" : " "} e5=#${rE5} qwen3=#${rQ}  ${tool}`);
  }

  const e5m = metrics(e5Ranks);
  const qm = metrics(qRanks);
  console.log("[intent] detalle por intención (e5-large vs qwen3):");
  for (const d of detail) console.log(d);
  console.log(
    `[intent] e5-large  top1=${(e5m.top1 * 100).toFixed(0)}% top5=${(e5m.top5 * 100).toFixed(0)}% MRR=${e5m.mrr.toFixed(3)}`,
  );
  console.log(
    `[intent] qwen3     top1=${(qm.top1 * 100).toFixed(0)}% top5=${(qm.top5 * 100).toFixed(0)}% MRR=${qm.mrr.toFixed(3)}`,
  );

  const verdict =
    qm.mrr > e5m.mrr + 0.005 ? "Qwen3 MEJOR" : qm.mrr < e5m.mrr - 0.005 ? "Qwen3 PEOR" : "EMPATE";
  console.log(`[intent] veredicto (MRR): ${verdict} (Δ=${(qm.mrr - e5m.mrr).toFixed(3)})`);

  // Sanidad del cargador: ambos modelos deben recuperar el tool correcto en top-5.
  assert.ok(e5m.top5 >= 0.6, `e5-large top5=${e5m.top5.toFixed(2)} demasiado bajo (¿loader/engine roto?)`);
  assert.ok(qm.top5 >= 0.6, `qwen3 top5=${qm.top5.toFixed(2)} demasiado bajo (¿loader Qwen3 roto?)`);
});
