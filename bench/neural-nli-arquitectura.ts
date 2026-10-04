/**
 * Arquitectura v2 · 3 modelos SIN SEGMENTACIÓN (AGENT.md §1-§2).
 *
 *   [turno + historial]
 *         │
 *         ▼
 *   M1 · RECALL (bi-encoder gte, AGENT.md §2.A)
 *        pool = /predict(turno) ∪ top-K coseno(turno) ∪ top-K coseno(contexto)
 *        (el pool NO se recorta con el umbral adaptativo: multi-intención)
 *         │
 *         ▼
 *   M2 · JUEZ NLI MULTILINGÜE (mDeBERTa-v3-xnli, AGENT.md §2.B)
 *        Zero-shot por herramienta sobre el TURNO COMPLETO:
 *          premisa   = turno  (o contexto+turno si el turno no es autónomo)
 *          hipótesis = "El usuario quiere: {intentSummary}."
 *          score     = P(entailment) − P(contradiction)
 *        La negación, la ironía, la auto-corrección y la multi-intención se
 *        resuelven por la atención cruzada del modelo sobre la frase entera:
 *        no se parte el texto ni se listan palabras.
 *         │
 *         ▼
 *   M3 · DECISIÓN (umbral global único, sin reglas por caso)
 *        aceptadas = score ≥ τ  (orden por score) · ninguna ⇒ ∅ (abstención)
 *        Autonomía: si ninguna candidata supera τ con el turno solo, se
 *        re-juzga con el contexto (anáfora, confirmación, rechazo).
 *
 * Las hipótesis salen del CATÁLOGO (intentSummary), no de los casos: el mismo
 * código sirve para cualquier consulta y cualquier idioma del modelo.
 *
 *   pnpm tsx bench/neural-nli-arquitectura.ts             → mDeBERTa multilingüe
 *   pnpm tsx bench/neural-nli-arquitectura.ts --nli=en    → DeBERTa inglés (control)
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import type {
  OnnxModuleLike,
  OnnxSessionLike,
  TokenizerLike,
  TokenizerModuleLike,
  ToolDefinition,
} from "../src/shared/interfaces";
import { BENCH_TOOLS } from "./catalog";
import { BENCH_CASES, type BenchCase, type BenchTurn } from "./cases";
import { RemtkBenchClient } from "./remtk.client";

const nodeRequire = createRequire(__filename);
const MAX_SEQ_LEN = 384;

export const NLI_DIRS = {
  multi: resolve(process.cwd(), "models", "mdeberta-v3-base-xnli-int8-onnx"),
  en: resolve(process.cwd(), "models", "nli-deberta-v3-small-int8-onnx"),
};

/** Parámetros globales (uno por concepto, no por caso). */
export const PARAMS = {
  /** Umbral de aceptación sobre P(ent) − P(contra). */
  tau: 0.5,
  /** Candidatas por coseno que se añaden al pool (≈15% del catálogo, mínimo 5). */
  recallFrac: 0.15,
  recallMin: 5,
  /** Turnos de historial usados como contexto de la premisa. */
  contextTurns: 2,
};

// ── M2 · Modelo NLI (orden de etiquetas leído de config.json) ────────────────

export interface NliProbs {
  ent: number;
  neu: number;
  con: number;
}

export class NliModel {
  private session?: OnnxSessionLike;
  private tokenizer?: TokenizerLike;
  private onnx?: OnnxModuleLike;
  private idx = { ent: 0, neu: 1, con: 2 };
  private loading?: Promise<boolean>;
  private cache = new Map<string, NliProbs>();

  constructor(readonly dir: string) {}

  ready(): Promise<boolean> {
    this.loading ??= (async () => {
      const modelPath = join(this.dir, "model.onnx");
      if (!existsSync(modelPath)) return false;
      const cfg = JSON.parse(readFileSync(join(this.dir, "config.json"), "utf8")) as {
        id2label?: Record<string, string>;
      };
      for (const [k, v] of Object.entries(cfg.id2label ?? {})) {
        const l = v.toLowerCase();
        if (l.startsWith("entail")) this.idx.ent = Number(k);
        else if (l.startsWith("neutral")) this.idx.neu = Number(k);
        else if (l.startsWith("contra")) this.idx.con = Number(k);
      }
      this.onnx = nodeRequire("onnxruntime-node") as OnnxModuleLike;
      const tm = nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;
      const tj = JSON.parse(readFileSync(join(this.dir, "tokenizer.json"), "utf8"));
      const tc = JSON.parse(readFileSync(join(this.dir, "tokenizer_config.json"), "utf8"));
      this.session = await this.onnx.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
      this.tokenizer = new tm.Tokenizer(tj, tc);
      return true;
    })();
    return this.loading;
  }

  async probs(premise: string, hypothesis: string): Promise<NliProbs> {
    const key = `${premise}\u0000${hypothesis}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    if (!(await this.ready())) throw new Error(`modelo NLI no disponible en ${this.dir}`);
    const a = this.tokenizer!.encode(premise).ids;
    const b = this.tokenizer!.encode(hypothesis).ids.slice(1);
    // Se trunca la PREMISA (por la izquierda: lo reciente pesa más), nunca la hipótesis.
    const room = MAX_SEQ_LEN - b.length;
    const aT = a.length > room ? [a[0], ...a.slice(a.length - room + 1)] : a;
    const ids = [...aT, ...b];
    const types = [...aT.map(() => 0), ...b.map(() => 1)];
    const n = ids.length;
    const t = (arr: number[]) => new this.onnx!.Tensor("int64", BigInt64Array.from(arr.map(BigInt)), [1, n]);
    const feeds: Record<string, unknown> = { input_ids: t(ids), attention_mask: t(ids.map(() => 1)) };
    if (!this.session!.inputNames || this.session!.inputNames.includes("token_type_ids")) feeds.token_type_ids = t(types);
    const out = await this.session!.run(feeds);
    const l = (Object.values(out)[0] as { data: Float32Array }).data;
    const m = Math.max(l[0], l[1], l[2]);
    const e = [Math.exp(l[0] - m), Math.exp(l[1] - m), Math.exp(l[2] - m)];
    const s = e[0] + e[1] + e[2];
    const p = { ent: e[this.idx.ent] / s, neu: e[this.idx.neu] / s, con: e[this.idx.con] / s };
    this.cache.set(key, p);
    return p;
  }
}

// ── Utilidades ───────────────────────────────────────────────────────────────

function cos(a: Float32Array, b: Float32Array): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

/** Hipótesis derivada del catálogo (plantilla zero-shot estándar). */
export function hipotesis(t: ToolDefinition): string {
  const s = (t.intentSummary || t.description).trim();
  return `El usuario quiere: ${s.charAt(0).toLowerCase()}${s.slice(1)}.`;
}

function contexto(history: BenchTurn[], n: number): string {
  return history
    .slice(-n)
    .map((h) => `${h.role === "user" ? "Usuario" : "Asistente"}: ${h.content}`)
    .join("\n");
}

// ── Orquestador v2 ───────────────────────────────────────────────────────────

export interface Juicio {
  tool: string;
  score: number;
  p: NliProbs;
}

export interface ResultadoV2 {
  tools: string[];
  pool: string[];
  juicios: Juicio[];
  premisa: "turno" | "contexto";
  latencyMs: number;
}

export class ArquitecturaV2 {
  private firmas?: { t: ToolDefinition; v: Float32Array }[];
  private seq = 0;

  constructor(
    private readonly remtk: RemtkBenchClient,
    private readonly engine: EmbeddingEngineService,
    private readonly catalog: ToolDefinition[],
    readonly nli: NliModel,
  ) {}

  /** M1 · top-K por coseno contra la firma de cada herramienta. */
  private async topK(texto: string): Promise<string[]> {
    if (!this.firmas) {
      this.firmas = [];
      for (const t of this.catalog) {
        const firma = [t.intentSummary, t.description, (t.tags ?? []).join(" ")].filter(Boolean).join(". ");
        this.firmas.push({ t, v: (await this.engine.embedPassage(firma, undefined, { high: true })).embedding });
      }
    }
    const q = (await this.engine.embedQuery(texto, undefined, { high: true })).embedding;
    const k = Math.max(PARAMS.recallMin, Math.ceil(this.catalog.length * PARAMS.recallFrac));
    return this.firmas
      .map((f) => ({ n: f.t.name, s: cos(q, f.v) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map((x) => x.n);
  }

  private async juzgar(premisa: string, pool: string[]): Promise<Juicio[]> {
    const out: Juicio[] = [];
    for (const name of pool) {
      const t = this.catalog.find((x) => x.name === name);
      if (!t) continue;
      const p = await this.nli.probs(premisa, hipotesis(t));
      out.push({ tool: name, score: p.ent - p.con, p });
    }
    return out.sort((a, b) => b.score - a.score);
  }

  async predict(caseId: string, text: string, history: BenchTurn[]): Promise<ResultadoV2> {
    const t0 = Date.now();
    const ctx = contexto(history, PARAMS.contextTurns);

    // M1 · recall amplio
    const remtkTools = (await this.remtk.predict(`v2-${caseId}-${++this.seq}`, text, history)).tools;
    const pool = new Set<string>([...remtkTools, ...(await this.topK(text))]);
    if (ctx) for (const n of await this.topK(`${ctx}\n${text}`)) pool.add(n);
    const poolArr = [...pool];

    // M2 · juicio sobre el turno solo
    let premisa: "turno" | "contexto" = "turno";
    let juicios = await this.juzgar(text, poolArr);

    // Autonomía: si el turno solo no sostiene ninguna intención, decide el contexto.
    if (ctx && !juicios.some((j) => j.score >= PARAMS.tau)) {
      premisa = "contexto";
      juicios = await this.juzgar(`${ctx}\nUsuario: ${text}`, poolArr);
    }

    // M3 · decisión por umbral global
    const tools = juicios.filter((j) => j.score >= PARAMS.tau).map((j) => j.tool);
    return { tools, pool: poolArr, juicios, premisa, latencyMs: Date.now() - t0 };
  }
}

// ── Evaluación ───────────────────────────────────────────────────────────────

export function aprueba(c: BenchCase, tools: string[]): boolean {
  const picked = tools.slice(0, 3);
  if (c.expected.length === 0) return tools.length === 0;
  if (c.multi) return c.expected.every((e) => picked.includes(e));
  return tools[0] === c.expected[0] || (c.expected.includes(tools[0]) && c.expected.length > 1);
}

export async function crearV2(nliDir: string) {
  const remtk = new RemtkBenchClient();
  await remtk.start(BENCH_TOOLS, { qdrant: false });
  const engine = new EmbeddingEngineService(loadConfig());
  const nli = new NliModel(nliDir);
  if (!(await nli.ready())) throw new Error(`no se encontró el modelo NLI en ${nliDir}`);
  return { remtk, v2: new ArquitecturaV2(remtk, engine, BENCH_TOOLS, nli) };
}

async function main() {
  const variante = process.argv.includes("--nli=en") ? "en" : "multi";
  const { remtk, v2 } = await crearV2(NLI_DIRS[variante]);
  console.log(`[v2] NLI=${variante} (${path.basename(NLI_DIRS[variante])}) τ=${PARAMS.tau} remtk=${remtk.mode}\n`);

  const filas = [];
  let okB = 0, okV = 0, ms = 0;
  for (const c of BENCH_CASES) {
    const idx = c.turns.map((t) => t.role).lastIndexOf("user");
    const history = c.turns.slice(0, idx);
    const text = c.turns[idx].content;

    const h: BenchTurn[] = [];
    let base: string[] = [];
    for (const t of c.turns) {
      if (t.role === "user") base = (await remtk.predict(`base-${c.id}`, t.content, [...h])).tools;
      h.push(t);
    }
    const r = await v2.predict(c.id, text, history);
    const pb = aprueba(c, base), pv = aprueba(c, r.tools);
    okB += +pb; okV += +pv; ms += r.latencyMs;
    filas.push({ id: c.id, esperado: c.expected, base, v2: r.tools, pb, pv, premisa: r.premisa, juicios: r.juicios });

    const m = (p: boolean) => (p ? "✓" : "✗");
    console.log(
      `${m(pb)} base  ${m(pv)} v2  ${c.id.padEnd(28)} esperado=[${c.expected.join(",") || "∅"}] ` +
        `v2=[${r.tools.join(",")}] premisa=${r.premisa} ${r.latencyMs}ms`,
    );
    console.log(
      "      " +
        r.juicios
          .slice(0, 4)
          .map((j) => `${j.tool}=${j.score.toFixed(2)}(e${j.p.ent.toFixed(2)}/c${j.p.con.toFixed(2)})`)
          .join("  "),
    );
  }
  await remtk.close();

  console.log(`\n── resumen (NLI=${variante}) ──`);
  console.log(`baseline: ${okB}/${BENCH_CASES.length}   v2: ${okV}/${BENCH_CASES.length}   latencia v2 media: ${Math.round(ms / BENCH_CASES.length)}ms`);
  const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "results");
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `v2-${variante}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
  fs.writeFileSync(file, JSON.stringify({ variante, params: PARAMS, baseline: okB, v2: okV, casos: filas }, null, 2));
  console.log(`[v2] reporte → ${file}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error("[v2] fallo fatal:", e);
    process.exit(1);
  });
}
