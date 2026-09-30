/**
 * Test unitario de ARQUITECTURA: compara, sobre el benchmark real
 * (`bench/cases.ts`), la arquitectura actual contra la arquitectura alternativa
 * de arbitraje por doble juez.
 *
 *   [Input ruidoso]
 *            │
 *   ┌────────┴─────────┐
 *   │ ENCARGADO DE     │  gte-multilingual-base + graph-cache (bi-encoder)
 *   │ CACHÉ            │  → propone la hipótesis caliente C₀ (= top-1 del
 *   └────────┬─────────┘    pipeline actual, que ya corre sobre el grafo)
 *            │
 *   ┌────────┴─────────┐        ┌──────────────────┐
 *   │ JUEZ 1 SEMÁNTICO │        │ JUEZ 2 LÓGICO/NLI │
 *   │ reranker (CE)    │        │ NLI 3-vías        │
 *   │ ¿el texto soporta│        │ ¿contradice o     │
 *   │ la herramienta?  │        │ está en duda?     │
 *   └────────┬─────────┘        └─────────┬────────┘
 *            └────────────┬───────────────┘
 *                  [ARBITRAJE]
 *     J1 aprueba ∧ J2 aprueba  → Ratifica C₀
 *     J2 contradice            → Veta caché y abre Recall (candidatos sin C₀)
 *     discrepancia no resuelta → Abstención (__NOOP__ → ∅)
 *
 * El arnés vive DENTRO del test (no toca producción): levanta el listener real
 * de predicción, juega los 12 casos del benchmark con historial y, para cada
 * turno evaluado, aplica el arbitraje en memoria sobre la hipótesis caliente
 * que propone el pipeline. Ambos lados (actual vs alternativa) se puntúan con
 * la misma regla que `bench/run.ts` (`scoreRemtk`), para que la comparación sea
 * pareja.
 *
 * Se comparan DOS alternativas contra el pipeline actual:
 *
 *  - Variante B · 2 jueces: arbitra sobre la hipótesis caliente C₀ del pipeline
 *    (reranker ∧ NLI, AND estricto). Es quirúrgica pero hereda la poda del
 *    pipeline: si el pipeline no propone nada, no hay nada que arbitrar.
 *
 *  - Variante C · 3 discriminadores (autocontenida, MULTILINGÜE):
 *      D1 · SOPORTE   (bi-encoder gte)   → recall por UNIÓN (turno actual ∪ contexto),
 *                                          sin puerta λ dura: se pondera el turno
 *                                          actual, y cuando el turno es anafórico
 *                                          (no mapea a ninguna herramienta) el
 *                                          contexto domina.
 *      D2 · PRECISIÓN (MaxSim tardío gte)→ alineación token a token (multi-intención);
 *                                          sustituye al cross-encoder inglés, que se
 *                                          degradaba en español.
 *      D3 · ESTADO    (arquetipos gte)   → confirm / reject / neutral por coseno
 *                                          contra arquetipos; sustituye al NLI inglés.
 *                                          `reject` sobre un turno débil ⇒ ∅; en un
 *                                          turno con contenido, D3 es neutral y no veta.
 *    Arbitraje PONDERADO (no-AND) + umbral de abstención (__NOOP__ → ∅) + banda
 *    multi-intención. Objetivo: 12/12.
 *
 * Jueces: la Variante C es 100% `gte-multilingual-base` (embeddings + tokens).
 *   La Variante B conserva los jueces ingleses de contraste:
 *   - J1 = `ms-marco-minilm-l6` (cross-encoder) · J2 = `nli-deberta-v3-small`.
 * Se usa inferencia NLI propia (no `EstadoNliService`) porque el orden de
 * etiquetas del modelo instalado es {0:contradiction, 1:entailment, 2:neutral}.
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { CrossEncoderService } from "../src/juicio/services/cross-encoder.service";
import { onnxMaxTokens } from "../src/shared/constants/predict/embedding.constants";
import type {
  OnnxModuleLike,
  OnnxSessionLike,
  TokenizerLike,
  TokenizerModuleLike,
  ToolDefinition,
} from "../src/shared/interfaces";
import { BENCH_TOOLS } from "../bench/catalog";
import { BENCH_CASES, type BenchCase, type BenchTurn } from "../bench/cases";
import { RemtkBenchClient } from "../bench/remtk.client";
import { toolNamesInText } from "../src/predict/keywords";

const nodeRequire = createRequire(__filename);

/** Umbral de aprobación del Juez 1 (sigmoide del cross-encoder). */
const CE_APPROVE_TAU = 0.5;

/** Orden de etiquetas del NLI instalado (`nli-deberta-v3-small` id2label). */
const NLI_LABEL = { contradiction: 0, entailment: 1, neutral: 2 } as const;
type NliVerdict = "contradiction" | "entailment" | "neutral";

// ── Variante C v3: 3 discriminadores MULTILINGÜES (100% gte) ─────────────────
// D1 (SOPORTE, bi-encoder gte) segmenta el turno en UNIDADES DE INTENCIÓN y
// recupera candidatos por UNIÓN (top-K por unidad ∪ top-K del contexto). D2
// (PRECISIÓN, MaxSim tardío gte) alinea token a token y es el rankeador
// principal cuando el turno es autónomo. D3 (ESTADO/LÓGICA, arquetipos gte)
// detecta rechazo/confirmación, la MENCiÓN EXPLÍCITA de un nombre del catálogo
// y — clave — la AUTONOMÍA del turno (su margen coseno): si el turno no trae
// intención propia, decide el contexto; sin contexto (o rechazo) ⇒ ∅.
const C_RECALL_K = 8;
/** Margen (max−media del coseno del turno) que marca un turno AUTÓNOMO: su
 *  intención se resuelve con el propio turno. Por debajo, el turno es
 *  DEPENDIENTE (anafórico o no accionable): decide el contexto; sin contexto ⇒ ∅. */
const C_MARGIN = 0.13;
/** Fusión AUTÓNOMA: el turno se rankea por D2 (precisión MaxSim por cláusula);
 *  D1 (coseno del turno) desempata. */
const C_W1_AUTO = 0.25;
const C_W2_AUTO = 0.7;
/** Fusión DEPENDIENTE: manda D1 sobre el CONTEXTO (cc, coseno del historial);
 *  D2 (MaxSim del contexto) solo desempata. */
const C_W1_DEP = 0.85;
const C_W2_DEP = 0.1;
/** Peso del estado (D3). */
const C_W3 = 0.05;
/** Umbrales del detector de estado (D3) por coseno contra arquetipos. */
const C_REJ_TAU = 0.6;
const C_CONF_TAU = 0.6;
/** Umbral de abstención: si el mejor `fused` < τ ⇒ ∅ (__NOOP__). */
const C_TAU = 0.4;
/** Banda multi-intención: se aceptan los candidatos dentro de δ del primero. */
const C_DELTA = 0.12;
/** Unidades de intención del turno (conjunciones y puntuación). */
const C_CLAUSE_SPLIT = /[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b/gi;

/** Arquetipos de ESTADO (D3): prototipos de rechazo/confirmación del turno. */
const REJECT_ARCH = ["no, todavía no", "no gracias, mejor no", "no por ahora", "cancélalo, no quiero"];
const CONFIRM_ARCH = ["sí, hazlo, adelante", "dale, procede", "sí, por favor, confirma"];

const NLI_DIR = [
  resolve(process.cwd(), "models", "nli-deberta-v3-small-int8-onnx"),
  resolve(__dirname, "..", "models", "nli-deberta-v3-small-int8-onnx"),
].find((d) => existsSync(join(d, "model.onnx")));

const BY_NAME = new Map<string, ToolDefinition>(BENCH_TOOLS.map((t) => [t.name, t]));

// ─────────────────────────────────────────────────────────────────────────────
// Juez 2: NLI 3-vías (contradicción / entailment / neutral) sobre el par
// (propuesta de la herramienta → petición del usuario).
// ─────────────────────────────────────────────────────────────────────────────
class NliJudge {
  private session?: OnnxSessionLike;
  private tokenizer?: TokenizerLike;
  private onnx?: OnnxModuleLike;
  private tokenizerModule?: TokenizerModuleLike;
  private attempted = false;
  private loaded = false;

  constructor(private readonly dir: string | undefined) {}

  async ready(): Promise<boolean> {
    if (this.loaded) return true;
    if (this.attempted) return false;
    this.attempted = true;
    if (!this.dir) return false;
    const modelPath = join(this.dir, "model.onnx");
    if (!existsSync(modelPath)) return false;
    this.onnx = nodeRequire("onnxruntime-node") as OnnxModuleLike;
    this.tokenizerModule = nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;
    const tokenizerJson = JSON.parse(readFileSync(join(this.dir, "tokenizer.json"), "utf8")) as Record<string, unknown>;
    const tokenizerConfig = JSON.parse(readFileSync(join(this.dir, "tokenizer_config.json"), "utf8")) as Record<string, unknown>;
    this.session = await this.onnx.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
    this.tokenizer = new this.tokenizerModule.Tokenizer(tokenizerJson, tokenizerConfig);
    this.loaded = true;
    return true;
  }

  /** NLI(premisa → hipótesis) con el orden de etiquetas real del modelo. */
  async judge(
    premise: string,
    hypothesis: string,
  ): Promise<{ verdict: NliVerdict; p: [number, number, number] } | undefined> {
    if (!(await this.ready())) return undefined;
    try {
      const encA = this.tokenizer!.encode(premise);
      const encB = this.tokenizer!.encode(hypothesis);
      const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, onnxMaxTokens);
      const types = [...encA.ids.map(() => 0), ...encB.ids.slice(1).map(() => 1)].slice(0, onnxMaxTokens);
      const len = ids.length;
      const inputIds = new BigInt64Array(len);
      const attentionMask = new BigInt64Array(len);
      const tokenTypeIds = new BigInt64Array(len);
      for (let i = 0; i < len; i++) {
        inputIds[i] = BigInt(ids[i]);
        attentionMask[i] = 1n;
        tokenTypeIds[i] = BigInt(types[i]);
      }
      const feeds: Record<string, unknown> = {
        input_ids: new this.onnx!.Tensor("int64", inputIds, [1, len]),
        attention_mask: new this.onnx!.Tensor("int64", attentionMask, [1, len]),
      };
      if (!this.session!.inputNames || this.session!.inputNames.includes("token_type_ids")) {
        feeds.token_type_ids = new this.onnx!.Tensor("int64", tokenTypeIds, [1, len]);
      }
      const outputs = await this.session!.run(feeds);
      const logits = Object.values(outputs)[0] as { data: Float32Array };
      const p = softmax3(logits.data);
      let verdict: NliVerdict;
      if (p[NLI_LABEL.contradiction] >= p[NLI_LABEL.entailment] && p[NLI_LABEL.contradiction] >= p[NLI_LABEL.neutral]) {
        verdict = "contradiction";
      } else if (p[NLI_LABEL.entailment] >= p[NLI_LABEL.neutral]) {
        verdict = "entailment";
      } else {
        verdict = "neutral";
      }
      return { verdict, p };
    } catch {
      return undefined;
    }
  }
}

function softmax3(logits: Float32Array): [number, number, number] {
  const m = Math.max(logits[0] ?? 0, logits[1] ?? 0, logits[2] ?? 0);
  const e0 = Math.exp((logits[0] ?? 0) - m);
  const e1 = Math.exp((logits[1] ?? 0) - m);
  const e2 = Math.exp((logits[2] ?? 0) - m);
  const z = e0 + e1 + e2;
  return [e0 / z, e1 / z, e2 / z];
}

// ─────────────────────────────────────────────────────────────────────────────
// Arbitraje y puntuación
// ─────────────────────────────────────────────────────────────────────────────
type Decision = "ratify" | "veto-recall" | "abstain" | "no-candidate";

interface Arbitration {
  decision: Decision;
  tools: string[];
  j1?: number;
  j2?: NliVerdict;
}

function toolText(t: ToolDefinition): string {
  return `${t.name}: ${t.intentSummary}. ${t.description}`;
}

/** Análogo exacto de `scoreRemtk` de `bench/run.ts` (no se importa para no ejecutar su main). */
function scoreCase(c: BenchCase, tools: string[]): boolean {
  const picked = tools.slice(0, 3);
  if (c.expected.length === 0) return tools.length === 0;
  if (c.multi) return c.expected.every((e) => picked.includes(e));
  return tools[0] === c.expected[0] || (c.expected.includes(tools[0]) && c.expected.length > 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Estado del test
// ─────────────────────────────────────────────────────────────────────────────
const client = new RemtkBenchClient();
const cfg = loadConfig();
const reranker = new CrossEncoderService(cfg);
const nli = new NliJudge(NLI_DIR);

/**
 * Arbitraje de la arquitectura alternativa sobre la hipótesis caliente C₀
 * (`candidates[0]`, propuesta por el encargado de caché) y su recall
 * (`candidates[1..]`).
 */
async function arbitrate(text: string, candidates: string[]): Promise<Arbitration> {
  const top = candidates[0];
  const def = top ? BY_NAME.get(top) : undefined;
  if (!def) return { decision: "no-candidate", tools: [] };

  const j1 = await reranker.score(text, toolText(def));
  const nliOut = await nli.judge(
    `El asistente propone usar la herramienta ${def.name}: ${def.intentSummary}. ${def.description}`,
    `El usuario pide: ${text}`,
  );
  const j2 = nliOut?.verdict;

  // Juez 2 detecta contradicción → veta la caché y abre Recall (sin C₀).
  if (j2 === "contradiction") {
    return { decision: "veto-recall", tools: candidates.slice(1).slice(0, 3), j1, j2 };
  }
  // Juez 1 aprueba y Juez 2 aprueba → ratifica C₀.
  if (j1 !== undefined && j1 >= CE_APPROVE_TAU && j2 === "entailment") {
    return { decision: "ratify", tools: candidates.slice(0, 3), j1, j2 };
  }
  // Discrepancia no resuelta → abstención (__NOOP__ → ∅).
  return { decision: "abstain", tools: [], j1, j2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Variante C v2 · 3 discriminadores multilingües (autocontenida: recall propio)
// ─────────────────────────────────────────────────────────────────────────────
const engine = new EmbeddingEngineService(cfg);
/** D1 · firma de cada herramienta (passage) en un vector. */
let toolIndex: Array<{ name: string; vec: Float32Array }> = [];
/** D2 · firma de cada herramienta a NIVEL TOKEN (MaxSim tardío). */
let toolTokens = new Map<string, Float32Array[]>();
/** D3 · arquetipos de estado (passage) para el detector confirm/reject. */
let rejectArch: Float32Array[] = [];
let confirmArch: Float32Array[] = [];

/** Firma de la herramienta: SOLO la descripción (validado: incluir tags
 *  duplica términos y sesga el ranking — p.ej. `save_draft` gana a `send_email`
 *  en consultas con "correo"). */
function sig(t: ToolDefinition): string {
  return t.description;
}

/** Construye los índices de D1 (passage), D2 (tokens) y D3 (arquetipos). */
async function buildIndexes(): Promise<void> {
  toolIndex = [];
  toolTokens = new Map();
  for (const t of BENCH_TOOLS) {
    const { embedding } = await engine.embedPassage(sig(t), undefined, { high: true });
    toolIndex.push({ name: t.name, vec: embedding });
    toolTokens.set(t.name, await engine.embedTokens(sig(t), undefined, { high: true }));
  }
  const embedArches = async (texts: string[]): Promise<Float32Array[]> =>
    Promise.all(texts.map(async (s) => (await engine.embedPassage(s, undefined, { high: true })).embedding));
  rejectArch = await embedArches(REJECT_ARCH);
  confirmArch = await embedArches(CONFIRM_ARCH);
}

/** MaxSim tardío (estilo ColBERT): (1/|J|) Σ_j max_i (q_i · t_j). */
function maxsim(qTokens: Float32Array[], tTokens: Float32Array[]): number {
  if (qTokens.length === 0 || tTokens.length === 0) return 0;
  let acc = 0;
  for (const t of tTokens) {
    let best = 0;
    for (const q of qTokens) {
      const s = EmbeddingEngineService.cosine(q, t);
      if (s > best) best = s;
    }
    acc += best;
  }
  return acc / tTokens.length;
}

type StateVerdict = "reject" | "confirm" | "neutral";

const CATALOG_NAMES = BENCH_TOOLS.map((t) => t.name);

/** D3 · estado del turno por coseno contra arquetipos. Solo se consulta cuando
 *  el turno es DEPENDIENTE (sin intención propia); un turno AUTÓNOMO con
 *  contenido es neutral por defecto y D3 no veta. */
function detectState(
  nowVec: Float32Array,
  autonomous: boolean,
): { state: StateVerdict; reject: number; confirm: number } {
  let reject = 0;
  for (const a of rejectArch) {
    const s = EmbeddingEngineService.cosine(nowVec, a);
    if (s > reject) reject = s;
  }
  let confirm = 0;
  for (const a of confirmArch) {
    const s = EmbeddingEngineService.cosine(nowVec, a);
    if (s > confirm) confirm = s;
  }
  if (autonomous) return { state: "neutral", reject, confirm };
  if (reject >= C_REJ_TAU && reject >= confirm) return { state: "reject", reject, confirm };
  if (confirm >= C_CONF_TAU && confirm > reject) return { state: "confirm", reject, confirm };
  return { state: "neutral", reject, confirm };
}

/** Divide el turno en unidades de intención y devuelve sus tokens (D2 · msC). */
async function clauseTokenSets(text: string): Promise<Float32Array[][]> {
  const parts = text
    .split(C_CLAUSE_SPLIT)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const units = parts.length > 0 ? parts : [text];
  return Promise.all(units.map((u) => engine.embedTokens(u, undefined, { high: true })));
}

/** MaxSim por cláusula: el MÁXIMO (no la media) entre las cláusulas del turno. */
function msClause(clauses: Float32Array[][], toolTok: Float32Array[]): number {
  let best = 0;
  for (const c of clauses) {
    const s = maxsim(c, toolTok);
    if (s > best) best = s;
  }
  return best;
}

interface CandScore {
  name: string;
  s1: number; // D1 · soporte (autónomo: coseno del turno `cn`; dependiente: coseno del contexto `cc`)
  s2: number; // D2 · precisión (autónomo: msC del turno; dependiente: msC del contexto)
  s3: number; // D3 · estado (confirm=1 / neutral=0.5 / reject=0)
  cNow: number; // coseno del turno actual con la herramienta (diagnóstico)
  cCtx: number; // coseno del contexto con la herramienta (diagnóstico)
  fused: number;
}

interface VariantCResult {
  out: CandScore[];
  state: StateVerdict;
  maxNow: number;
  margin: number;
  autonomous: boolean;
  hasCtx: boolean;
  mentioned: string[];
}

/**
 * D1 · soporte: recall por UNIÓN (top-K del turno actual ∪ top-K del contexto).
 * El turno se clasifica por MARGEN (max−media del coseno contra el catálogo):
 *   · AUTÓNOMO (margen ≥ `C_MARGIN`): su intención se resuelve con el propio
 *     turno → rankea D2 (msC, MaxSim por cláusula); D1 (coseno del turno) desempata.
 *   · DEPENDIENTE: no trae intención propia (anafórico / confirmación / rechazo)
 *     → rankea D1 sobre el CONTEXTO (cc); sin contexto ⇒ ∅ (abstención).
 * D3 · estado por arquetipos; un rechazo sobre un turno dependiente ⇒ ∅.
 */
async function runVariantC(currentText: string, ctxText?: string): Promise<VariantCResult> {
  const nowVec = (await engine.embedQuery(currentText, undefined, { high: true })).embedding;
  const nowClauses = await clauseTokenSets(currentText);
  const ctxVec = ctxText ? (await engine.embedQuery(ctxText, undefined, { high: true })).embedding : undefined;
  const ctxClauses = ctxText ? await clauseTokenSets(ctxText) : [];

  const cNow = new Map<string, number>();
  const cCtx = new Map<string, number>();
  for (const t of toolIndex) {
    cNow.set(t.name, EmbeddingEngineService.cosine(nowVec, t.vec));
    cCtx.set(t.name, ctxVec ? EmbeddingEngineService.cosine(ctxVec, t.vec) : 0);
  }
  const nowVals = toolIndex.map((t) => cNow.get(t.name)!);
  const maxNow = Math.max(...nowVals);
  const meanNow = nowVals.reduce((a, b) => a + b, 0) / nowVals.length;
  const margin = maxNow - meanNow;
  const autonomous = margin >= C_MARGIN;
  const hasCtx = ctxVec !== undefined;

  const { state } = detectState(nowVec, autonomous);
  const s3 = state === "confirm" ? 1 : state === "neutral" ? 0.5 : 0;

  // Recall por unión: top-K del turno actual y top-K del contexto.
  const byName = (m: Map<string, number>) => [...m.entries()].sort((a, b) => b[1] - a[1]).map(([n]) => n);
  const topNow = byName(cNow).slice(0, C_RECALL_K);
  const topCtx = hasCtx ? byName(cCtx).slice(0, C_RECALL_K) : [];
  const names = [...new Set([...topNow, ...topCtx])];

  const mentioned = toolNamesInText(currentText, CATALOG_NAMES);

  const out: CandScore[] = names
    .map((name) => {
      const cn = cNow.get(name)!;
      const cc = cCtx.get(name)!;
      const toolTok = toolTokens.get(name) ?? [];
      let s1: number;
      let s2: number;
      if (autonomous) {
        s1 = cn;
        s2 = msClause(nowClauses, toolTok);
      } else {
        s1 = cc;
        s2 = ctxClauses.length ? msClause(ctxClauses, toolTok) : 0;
      }
      const w1 = autonomous ? C_W1_AUTO : C_W1_DEP;
      const w2 = autonomous ? C_W2_AUTO : C_W2_DEP;
      return { name, s1, s2, s3, cNow: cn, cCtx: cc, fused: w1 * s1 + w2 * s2 + C_W3 * s3 };
    })
    .sort((a, b) => b.fused - a.fused);

  return { out, state, maxNow, margin, autonomous, hasCtx, mentioned };
}

/**
 * Arbitraje ponderado: la MENCIÓN EXPLÍCITA del nombre de una herramienta la
 * ratifica (orden directa del usuario); un rechazo (D3) ⇒ ∅; un turno
 * DEPENDIENTE sin contexto ⇒ ∅ (no hay nada que resolver); si el mejor
 * `fused < τ` ⇒ ∅ (__NOOP__); si no, se aceptan los candidatos dentro de δ
 * (habilita multi-intención).
 */
function arbitrateC(res: VariantCResult): { decision: Decision; tools: string[]; top?: CandScore } {
  const top = res.out[0];
  if (res.mentioned.length > 0) {
    return { decision: "ratify", tools: res.mentioned, top };
  }
  if (res.state === "reject") return { decision: "abstain", tools: [], top };
  if (!res.autonomous && !res.hasCtx) return { decision: "abstain", tools: [], top };
  if (!top || top.fused < C_TAU) return { decision: "abstain", tools: [], top };
  const tools = res.out.filter((f) => f.fused >= top.fused - C_DELTA).map((f) => f.name);
  return { decision: "ratify", tools, top };
}

before(async () => {
  await client.start(BENCH_TOOLS, { qdrant: false });
  assert.ok(await reranker.ready(), "Juez 1 (cross-encoder ms-marco) debe cargar");
  assert.ok(await nli.ready(), `Juez 2 (NLI) debe cargar desde ${NLI_DIR ?? "(no encontrado)"}`);
  // Variante C (v3, 100% gte): calentar el bi-encoder y construir los 3 índices.
  await engine.embedQuery("warmup", undefined, { high: true });
  await buildIndexes();
  assert.equal(toolIndex.length, BENCH_TOOLS.length, "D1 debe indexar todos los tools del catálogo");
  assert.equal(toolTokens.size, BENCH_TOOLS.length, "D2 debe indexar los tokens de todos los tools");
  assert.ok(rejectArch.length > 0 && confirmArch.length > 0, "D3 debe embeber los arquetipos de estado");
});
after(async () => {
  await client.close();
});

interface Row {
  id: string;
  expected: string[];
  cur: string[];
  curPass: boolean;
  altB: string[];
  altBPass: boolean;
  decisionB: Decision;
  j1?: number;
  j2?: NliVerdict;
  altC: string[];
  altCPass: boolean;
  decisionC: Decision;
  cTop?: CandScore;
  cState: StateVerdict;
  cAutonomous: boolean;
  cMargin: number;
  cMaxNow: number;
  cMentioned: string[];
  cMs: number;
  curMs: number;
}

test("ARQUITECTURA: 2 jueces (B) vs 3 discriminadores (C) vs pipeline actual (benchmark)", async () => {
  const rows: Row[] = [];

  for (const c of BENCH_CASES) {
    const sessionId = `arq-${c.id}`;
    const history: BenchTurn[] = [];
    const userTexts: string[] = [];
    let last = { tools: [] as string[], latencyMs: 0 };
    for (const turn of c.turns) {
      if (turn.role === "user") {
        const res = await client.predict(sessionId, turn.content, [...history]);
        last = { tools: res.tools, latencyMs: res.latencyMs };
        userTexts.push(turn.content);
      }
      history.push(turn);
    }

    const currentText = userTexts[userTexts.length - 1];
    // Contexto: los 2 turnos previos INCLUYENDO el turno del assistant (la
    // propuesta "¿envío el correo...?" es la que resuelve una confirmación).
    const ctxText =
      c.turns
        .slice(0, -1)
        .slice(-2)
        .map((t) => (t.role === "assistant" ? "asistente: " : "") + t.content)
        .join(" ")
        .trim() || undefined;

    // Variante B (2 jueces): arbitra sobre la hipótesis caliente del pipeline actual.
    const arbB = await arbitrate(currentText, last.tools);

    // Variante C v2 (3 discriminadores multilingües): recall propio D1 → D2/D3 → arbitraje.
    const t0 = Date.now();
    const res = await runVariantC(currentText, ctxText);
    const arbC = arbitrateC(res);
    const cMs = Date.now() - t0;

    rows.push({
      id: c.id,
      expected: c.expected,
      cur: last.tools,
      curPass: scoreCase(c, last.tools),
      altB: arbB.tools,
      altBPass: scoreCase(c, arbB.tools),
      decisionB: arbB.decision,
      j1: arbB.j1,
      j2: arbB.j2,
      altC: arbC.tools,
      altCPass: scoreCase(c, arbC.tools),
      decisionC: arbC.decision,
      cTop: res.out[0],
      cState: res.state,
      cAutonomous: res.autonomous,
      cMargin: res.margin,
      cMaxNow: res.maxNow,
      cMentioned: res.mentioned,
      cMs,
      curMs: last.latencyMs,
    });
  }

  console.log("\n===== ARQUITECTURA: ACTUAL vs B(2 jueces) vs C(3 discriminadores v3) =====");
  console.log(`modelo pipeline=${client.mode}`);
  for (const r of rows) {
    const t = r.cTop;
    console.log(
      `${r.curPass ? "✓" : "✗"} actual  ${r.altBPass ? "✓" : "✗"} B  ${r.altCPass ? "✓" : "✗"} C  ` +
        `${r.id.padEnd(22)} esperado=[${r.expected.join(",") || "∅"}]  ` +
        `actual=[${r.cur.join(",") || "∅"}]  C=[${r.altC.join(",") || "∅"}]  ` +
        `| C.top=${t?.name ?? "-"} fused=${t?.fused.toFixed(3) ?? "?"} ` +
        `(s1=${t?.s1.toFixed(3) ?? "?"} s2=${t?.s2.toFixed(3) ?? "?"} s3=${t?.s3.toFixed(2) ?? "?"}) ` +
        `reg=${r.cAutonomous ? "auto" : "dep"} marg=${r.cMargin.toFixed(3)} ` +
        `st=${r.cState}${r.cMentioned.length ? ` men=[${r.cMentioned.join(",")}]` : ""} dec=${r.decisionC}`,
    );
  }
  const curPass = rows.filter((r) => r.curPass).length;
  const bPass = rows.filter((r) => r.altBPass).length;
  const cPass = rows.filter((r) => r.altCPass).length;
  const curAvg = Math.round(rows.reduce((a, r) => a + r.curMs, 0) / rows.length);
  const cAvg = Math.round(rows.reduce((a, r) => a + r.cMs, 0) / rows.length);
  console.log("──────────────────────────────────────────────────────────────────────────");
  console.log(`actual (pipeline):      ${curPass}/${rows.length} casos  (latencia media ${curAvg}ms)`);
  console.log(`B · 2 jueces:           ${bPass}/${rows.length} casos`);
  console.log(`C · 3 discriminadores:  ${cPass}/${rows.length} casos  (cómputo D1+D2+D3 medio ${cAvg}ms)`);
  const best = Math.max(curPass, bPass, cPass);
  const winner = cPass === best ? "C (3 discriminadores)" : bPass === best ? "B (2 jueces)" : "ACTUAL";
  console.log(`GANADOR:                ${winner}`);
  console.log(`OBJETIVO 12/12:         ${cPass === rows.length ? "ALCANZADO ✓" : `no (${cPass}/${rows.length})`}`);
  console.log("==========================================================================\n");

  assert.equal(rows.length, BENCH_CASES.length, "deben ejecutarse todos los casos del benchmark");

  // Criterio de adopción: la arquitectura de 3 discriminadores no debe degradar
  // el accuracy del pipeline actual. Si falla, el mensaje indica el delta.
  assert.ok(
    cPass >= curPass,
    `La arquitectura C de 3 discriminadores (${cPass}/${rows.length}) NO mejora a la actual ` +
      `(${curPass}/${rows.length}); delta=${cPass - curPass}.`,
  );
});
