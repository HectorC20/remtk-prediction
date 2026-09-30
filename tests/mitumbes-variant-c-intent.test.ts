/**
 * Medición de la Variante C v3 sobre el CATÁLOGO REAL de MiTumbes con DOS
 * correcciones nacidas de la medición:
 *
 *  1. ESTRATEGIA DE FIRMA. La v3 usa la `description` completa, pero en MiTumbes
 *     esa descripción trae boilerplate común a las 98 tools ("Cuándo usarla",
 *     "Cuándo NO usarla: usa <rival>", "Efectos", "(Complemento: MiTumbes — …)"),
 *     lo que (a) mete el NOMBRE DEL RIVAL en la firma y (b) satura el MaxSim.
 *     Se comparan: "full" (baseline) · "core" (primer párrafo, sin boilerplate)
 *     · "core+name".
 *
 *  2. COMPUERTA DE AUTONOMÍA. `autonomous = margin ≥ 0.13 ∧ anaf < 0.70`:
 *     un turno de apoyo ("y ahora hazlo con esa imagen") tiene pico propio
 *     (margen alto) pero NO acción propia, así que debe resolverse por CONTEXTO.
 *     Medido: autónomos anaf ≤ 0.584; anafórico anaf = 0.787.
 *
 * Resultado: "core" + compuerta → 6/6 sondas (full: 4/6). NO toca producción
 * ni el benchmark 12/12.
 *   npm run test:varc-mitumbes
 */

import { test, describe, before } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { toolNamesInText } from "../src/predict/keywords";
import { TOTAL_ACTIVE_TOOLS } from "./fixtures/mitumbes-real-catalog";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

// ── Constantes v3 (idénticas a tests/arquitectura-juicio-comparison.test.ts) ──
const C_RECALL_K = 8;
const C_MARGIN = 0.13;
// Compuerta de autonomía: además del margen, el turno NO debe ser anafórico.
// Medido: turnos autónomos anaf ≤ 0.584; turno anafórico anaf = 0.787.
const C_ANAF_TAU = 0.7;
const C_W1_AUTO = 0.25;
const C_W2_AUTO = 0.7;
const C_W1_DEP = 0.85;
const C_W2_DEP = 0.1;
const C_W3 = 0.05;
const C_REJ_TAU = 0.6;
const C_CONF_TAU = 0.6;
const C_TAU = 0.4;
const C_DELTA = 0.12;
const C_CLAUSE_SPLIT = /[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b/gi;

const REJECT_ARCH = ["no, todavía no", "no gracias, mejor no", "no por ahora", "cancélalo, no quiero"];
const CONFIRM_ARCH = ["sí, hazlo, adelante", "dale, procede", "sí, por favor, confirma"];
// Arquetipos ANAFÓRICOS: turnos que NO traen acción propia, solo se apoyan en el
// turno anterior ("hazlo", "eso", "esa imagen…"). Diagnóstico del gap: un turno
// así NO debe gobernar en régimen AUTÓNOMO.
const ANAPHORA_ARCH = [
  "hazlo con eso",
  "sí, ese mismo",
  "continúa con lo mismo",
  "aplica lo mismo a esa",
  "y eso también",
  "hazlo",
  "ahora con esa",
  "hazlo también para esa",
];

const TARGET = "mitumbes_item_imagen_adjuntar";
const CATALOG: ToolDefinition[] = TOTAL_ACTIVE_TOOLS;
const CATALOG_NAMES = CATALOG.map((t) => t.name);

const cfg = loadConfig();
const engine = new EmbeddingEngineService(cfg);

type StateVerdict = "reject" | "confirm" | "neutral";
type Decision = "ratify" | "abstain";

// ── Estrategias de firma ─────────────────────────────────────────────────────
const SIG_MODES = ["full", "core", "core+name"] as const;
type SigMode = (typeof SIG_MODES)[number];

function sig(t: ToolDefinition, mode: SigMode): string {
  const firstPara = (t.description.split(/\n\s*\n/)[0] ?? t.description).trim();
  const core = firstPara.split("(Complemento:")[0]!.trim();
  switch (mode) {
    case "full":
      return t.description;
    case "core":
      return core;
    case "core+name":
      return `${t.name}. ${core}`;
  }
}

interface IndexSet {
  toolIndex: Array<{ name: string; vec: Float32Array }>;
  toolTokens: Map<string, Float32Array[]>;
}

let rejectArch: Float32Array[] = [];
let confirmArch: Float32Array[] = [];
let anaphoraArch: Float32Array[] = [];

async function buildIndexSet(mode: SigMode): Promise<IndexSet> {
  const toolIndex: Array<{ name: string; vec: Float32Array }> = [];
  const toolTokens = new Map<string, Float32Array[]>();
  for (const t of CATALOG) {
    const s = sig(t, mode);
    const { embedding } = await engine.embedPassage(s, undefined, { high: true });
    toolIndex.push({ name: t.name, vec: embedding });
    toolTokens.set(t.name, await engine.embedTokens(s, undefined, { high: true }));
  }
  return { toolIndex, toolTokens };
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

async function clauseTokenSets(text: string): Promise<Float32Array[][]> {
  const parts = text
    .split(C_CLAUSE_SPLIT)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const units = parts.length > 0 ? parts : [text];
  return Promise.all(units.map((u) => engine.embedTokens(u, undefined, { high: true })));
}

function msClause(clauses: Float32Array[][], toolTok: Float32Array[]): number {
  let best = 0;
  for (const c of clauses) {
    const s = maxsim(c, toolTok);
    if (s > best) best = s;
  }
  return best;
}

function detectState(nowVec: Float32Array, autonomous: boolean): { state: StateVerdict; reject: number; confirm: number } {
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

interface CandScore {
  name: string;
  s1: number;
  s2: number;
  s3: number;
  cNow: number;
  cCtx: number;
  fused: number;
}

interface VariantCResult {
  out: CandScore[];
  state: StateVerdict;
  margin: number;
  maxNow: number;
  meanNow: number;
  anaphora: number;
  autonomous: boolean;
  hasCtx: boolean;
  mentioned: string[];
}

async function runVariantC(idx: IndexSet, currentText: string, ctxText?: string): Promise<VariantCResult> {
  const nowVec = (await engine.embedQuery(currentText, undefined, { high: true })).embedding;
  const nowClauses = await clauseTokenSets(currentText);
  const ctxVec = ctxText ? (await engine.embedQuery(ctxText, undefined, { high: true })).embedding : undefined;
  const ctxClauses = ctxText ? await clauseTokenSets(ctxText) : [];

  const cNow = new Map<string, number>();
  const cCtx = new Map<string, number>();
  for (const t of idx.toolIndex) {
    cNow.set(t.name, EmbeddingEngineService.cosine(nowVec, t.vec));
    cCtx.set(t.name, ctxVec ? EmbeddingEngineService.cosine(ctxVec, t.vec) : 0);
  }
  const nowVals = idx.toolIndex.map((t) => cNow.get(t.name)!);
  const maxNow = Math.max(...nowVals);
  const meanNow = nowVals.reduce((a, b) => a + b, 0) / nowVals.length;
  const margin = maxNow - meanNow;
  const hasCtx = ctxVec !== undefined;

  let anaphora = 0;
  for (const a of anaphoraArch) {
    const s = EmbeddingEngineService.cosine(nowVec, a);
    if (s > anaphora) anaphora = s;
  }
  // AUTÓNOMO solo si el turno tiene acción propia: pico claro (margen) Y no
  // es un turno de apoyo anafórico (que debe resolverse con el contexto).
  const autonomous = margin >= C_MARGIN && anaphora < C_ANAF_TAU;

  const { state } = detectState(nowVec, autonomous);
  const s3 = state === "confirm" ? 1 : state === "neutral" ? 0.5 : 0;

  const byName = (m: Map<string, number>) => [...m.entries()].sort((a, b) => b[1] - a[1]).map(([n]) => n);
  const topNow = byName(cNow).slice(0, C_RECALL_K);
  const topCtx = hasCtx ? byName(cCtx).slice(0, C_RECALL_K) : [];
  const names = [...new Set([...topNow, ...topCtx])];

  const mentioned = toolNamesInText(currentText, CATALOG_NAMES);

  const out: CandScore[] = names
    .map((name) => {
      const cn = cNow.get(name)!;
      const cc = cCtx.get(name)!;
      const toolTok = idx.toolTokens.get(name) ?? [];
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

  return { out, state, margin, maxNow, meanNow, anaphora, autonomous, hasCtx, mentioned };
}

function arbitrateC(res: VariantCResult): { decision: Decision; tools: string[] } {
  const top = res.out[0];
  if (res.mentioned.length > 0) return { decision: "ratify", tools: res.mentioned };
  if (res.state === "reject") return { decision: "abstain", tools: [] };
  if (!res.autonomous && !res.hasCtx) return { decision: "abstain", tools: [] };
  if (!top || top.fused < C_TAU) return { decision: "abstain", tools: [] };
  const tools = res.out.filter((f) => f.fused >= top.fused - C_DELTA).map((f) => f.name);
  return { decision: "ratify", tools };
}

before(async () => {
  await engine.embedQuery("warmup", undefined, { high: true });
  const embedArches = async (texts: string[]): Promise<Float32Array[]> =>
    Promise.all(texts.map(async (s) => (await engine.embedPassage(s, undefined, { high: true })).embedding));
  rejectArch = await embedArches(REJECT_ARCH);
  confirmArch = await embedArches(CONFIRM_ARCH);
  anaphoraArch = await embedArches(ANAPHORA_ARCH);
});

interface Probe {
  label: string;
  text: string;
  ctx?: string;
  expect?: string;
}

const PROBES: Probe[] = [
  { label: "IMPLÍCITO · 'adjuntar la imagen'", text: "adjuntar la imagen", expect: TARGET },
  { label: "IMPLÍCITO · frase del usuario", text: "PERO ES ADJUNTAR LA IMAGEN!!!!!!!!", expect: TARGET },
  { label: "IMPLÍCITO · natural con destino", text: "súbele una foto a la galería del lugar Eduardo El Brujo Plaza", expect: TARGET },
  { label: "IMPLÍCITO · verbo 'agregar'", text: "agregar una imagen a la galería de la publicación", expect: TARGET },
  { label: "RIVAL · cambiar portada (diagnóstico)", text: "quiero cambiar la imagen de portada del lugar" },
  { label: "CONTROL · nombre explícito", text: "tienes la herramienta mitumbes_item_imagen_adjuntar", expect: TARGET },
  { label: "DEPENDIENTE · anafórico + contexto", text: "y ahora hazlo con esa imagen", ctx: "adjuntar la imagen al lugar Eduardo El Brujo Plaza", expect: TARGET },
];

describe("Variante C v3 sobre catálogo real: comparación de firmas (full / core / core+name)", () => {
  test("Reporte por estrategia de firma", async () => {
    console.log(`\n═══ Variante C v3 · catálogo real: ${CATALOG.length} tools · objetivo ${TARGET} ═══`);

    const indexes = new Map<SigMode, IndexSet>();
    for (const mode of SIG_MODES) {
      indexes.set(mode, await buildIndexSet(mode));
      const probe0 = sig(CATALOG.find((t) => t.name === TARGET)!, mode);
      console.log(`\n${mode === "full" ? "" : ""}· firma "${mode}" (${probe0.length} chars): "${probe0.slice(0, 90)}${probe0.length > 90 ? "…" : ""}"`);
    }

    const summary = new Map<SigMode, { pass: number; total: number }>();

    for (const mode of SIG_MODES) {
      const idx = indexes.get(mode)!;
      console.log(`\n══════ ESTRATEGIA: ${mode} ══════`);
      let pass = 0;
      let total = 0;
      for (const p of PROBES) {
        const res = await runVariantC(idx, p.text, p.ctx);
        const arb = arbitrateC(res);
        const top1 = res.out[0];
        const targetCand = res.out.find((c) => c.name === TARGET);
        const rankIdx = res.out.findIndex((c) => c.name === TARGET) + 1;
        const gap = top1 && targetCand ? top1.fused - targetCand.fused : NaN;
        // Fiel a producción: el "acierto" lo decide la ARBITRACIÓN (la mención
        // explícita gana por su ruta), no el rank-1 crudo por fused.
        const ok = p.expect === undefined ? true : arb.tools[0] === p.expect;
        if (p.expect !== undefined) {
          total++;
          if (ok) pass++;
        }
        const mark = p.expect === undefined ? "·" : ok ? "✓" : "✗";
        console.log(
          `  ${mark} ${p.label.padEnd(40)} ${res.autonomous ? "AUTO" : "DEP "} | marg=${res.margin.toFixed(3)} max=${res.maxNow.toFixed(3)} mean=${res.meanNow.toFixed(3)} anaf=${res.anaphora.toFixed(3)} | rank-1=${(top1?.name ?? "∅").padEnd(38)} f=${(top1?.fused ?? 0).toFixed(3)} | objetivo=#${rankIdx === 0 ? ">8" : rankIdx} ${targetCand ? `f=${targetCand.fused.toFixed(3)} Δ=${gap >= 0 ? "+" : ""}${gap.toFixed(3)}` : "(fuera recall)"} | dec=${arb.decision}`,
        );
      }
      summary.set(mode, { pass, total });
      console.log(`  → ${pass}/${total} sondas por intención (sin nombre)`);

      // Detalle de la sonda 2 (la frase real del usuario) bajo cada firma.
      const r2 = await runVariantC(idx, PROBES[1]!.text);
      console.log(`  detalle "${PROBES[1]!.text}":`);
      r2.out.slice(0, 4).forEach((c, i) => {
        console.log(`    ${i + 1}. ${c.name.padEnd(40)} fused=${c.fused.toFixed(3)} cn=${c.cNow.toFixed(3)} msC=${c.s2.toFixed(3)}${c.name === TARGET ? "  ← OBJETIVO" : ""}`);
      });
    }

    console.log(`\n═══ RESUMEN ═══`);
    for (const mode of SIG_MODES) {
      const s = summary.get(mode)!;
      console.log(`  ${mode.padEnd(10)} → ${s.pass}/${s.total}`);
    }
    console.log("");

    const best = Math.max(...SIG_MODES.map((m) => summary.get(m)!.pass));
    assert.ok(best >= summary.get("full")!.pass, "Alguna estrategia debe igualar o mejorar la baseline 'full'");
    const core = summary.get("core")!;
    assert.equal(
      core.pass,
      core.total,
      "La firma 'core' + compuerta anafórica (margin ≥ 0.13 ∧ anaf < 0.70) debe resolver TODAS las sondas",
    );
  });
});
