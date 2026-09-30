/**
 * VarianteCService: núcleo de 3 discriminadores multilingües (100%
 * `gte-multilingual-base`), portado del arnés ganador
 * (`tests/arquitectura-juicio-comparison.test.ts`, 12/12) con la firma `core`
 * y la compuerta anafórica de `tests/mitumbes-variant-c-intent.test.ts` (6/6).
 *
 *   D1 · SOPORTE   (bi-encoder gte)     → recall por UNIÓN (top-K del turno ∪
 *                                          top-K del contexto).
 *   D2 · PRECISIÓN (MaxSim tardío gte)  → alineación token a token por cláusula
 *                                          (`msC`); es el rankeador principal del
 *                                          turno autónomo.
 *   D3 · ESTADO    (arquetipos gte)     → confirm / reject / neutral por coseno,
 *                                          más la MENCIÓN EXPLÍCITA del nombre de
 *                                          una herramienta y la AUTONOMÍA del
 *                                          turno (margen coseno).
 *
 * Régimen por MARGEN (`max−media` del coseno del turno contra el catálogo):
 *   · AUTÓNOMO   (margen ≥ C_MARGIN ∧ no anafórico): decide el propio turno
 *     (fusión `0.25·cn + 0.7·msC + 0.05·s3`).
 *   · DEPENDIENTE: turno de apoyo/anafórico → decide el CONTEXTO
 *     (fusión `0.85·cc + 0.1·msC_ctx + 0.05·s3`); sin contexto ⇒ ∅.
 *
 * Arbitraje: mención explícita → ratifica; `reject` → ∅; dependiente sin
 * contexto → ∅; `fused < C_TAU` → ∅; si no, banda `C_DELTA`.
 *
 * El servicio es autocontenido: indexa el catálogo por scope (firma `core`) y
 * expone `rank()` + `arbitrate()`. NO toca producción por sí solo: el orquestador
 * lo activa con el flag `VARIANTE_C_ENABLED` (default OFF).
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { log, warn } from "src/logger";
import { toolNamesInText } from "src/predict/keywords";
import type { ToolDefinition } from "src/shared/interfaces/domain.interface";
import {
  C_ANAF_TAU,
  C_ANAPHORA_ARCH,
  C_CLAUSE_SPLIT,
  C_CONF_TAU,
  C_CONFIRM_ARCH,
  C_DELTA,
  C_MARGIN,
  C_RECALL_K,
  C_REJECT_ARCH,
  C_REJ_TAU,
  C_TAU,
  C_W1_AUTO,
  C_W1_DEP,
  C_W2_AUTO,
  C_W2_DEP,
  C_W3,
} from "src/shared/constants/predict";

export type StateVerdict = "reject" | "confirm" | "neutral";
export type CDecision = "ratify" | "abstain";

/** Candidata puntuada por el núcleo C. */
export interface CCandidate {
  name: string;
  /** D1 · soporte (autónomo: coseno del turno `cn`; dependiente: coseno del contexto `cc`). */
  s1: number;
  /** D2 · precisión (autónomo: msC del turno; dependiente: msC del contexto). */
  s2: number;
  /** D3 · estado (confirm=1 / neutral=0.5 / reject=0). */
  s3: number;
  cNow: number;
  cCtx: number;
  fused: number;
}

/** Resultado completo de `rank()`: candidatas + diagnóstico del régimen. */
export interface CRanking {
  out: CCandidate[];
  state: StateVerdict;
  margin: number;
  maxNow: number;
  meanNow: number;
  anaphora: number;
  autonomous: boolean;
  hasCtx: boolean;
  mentioned: string[];
}

interface IndexSet {
  toolIndex: Array<{ name: string; vec: Float32Array }>;
  toolTokens: Map<string, Float32Array[]>;
  names: string[];
}

interface Archetypes {
  reject: Float32Array[];
  confirm: Float32Array[];
  anaphora: Float32Array[];
}

/**
 * Firma `core` de la herramienta: primer párrafo de `description` sin el
 * boilerplate común a las tools del catálogo (`(Complemento: …)`), que metía el
 * nombre del RIVAL en la firma y saturaba el MaxSim. Medido: `core` + compuerta
 * resuelve todas las sondas; `full` falla 2 de 6.
 */
export function coreSignature(t: ToolDefinition): string {
  const firstPara = (t.description.split(/\n\s*\n/)[0] ?? t.description).trim();
  return firstPara.split("(Complemento:")[0]!.trim();
}

export class VarianteCService {
  /** Índices D1/D2 por scopeKey (se reconstruyen al re-registrar el catálogo). */
  private readonly indexes = new Map<string, IndexSet>();
  /** Arquetipos D3 (una sola carga; son independientes del scope). */
  private archetypes?: Archetypes;

  constructor(private readonly engine: EmbeddingEngineService) {}

  /** True si el scope tiene índice C listo (si no, el orquestador degrada). */
  has(scopeKey: string): boolean {
    return this.indexes.has(scopeKey);
  }

  /** Invalida el índice del scope (re-registro del catálogo). */
  clearScope(scopeKey: string): void {
    this.indexes.delete(scopeKey);
  }

  /**
   * Indexa el catálogo del scope: firma `core` por herramienta en un vector
   * (D1) y a nivel token (D2), más los arquetipos D3. Recompute masivo de
   * registro → prioridad baja (`high:false`).
   */
  async indexScope(scopeKey: string, tools: ToolDefinition[]): Promise<void> {
    await this.ensureArchetypes();
    const toolIndex: Array<{ name: string; vec: Float32Array }> = [];
    const toolTokens = new Map<string, Float32Array[]>();
    for (const t of tools) {
      const s = coreSignature(t);
      const { embedding } = await this.engine.embedPassage(s, undefined, { high: false });
      toolIndex.push({ name: t.name, vec: embedding });
      toolTokens.set(t.name, await this.engine.embedTokens(s, undefined, { high: false }));
    }
    this.indexes.set(scopeKey, { toolIndex, toolTokens, names: tools.map((t) => t.name) });
    log(`[varianteC] scope=${scopeKey} indexado D1/D2=${toolIndex.length} tools`);
  }

  /**
   * Corre los 3 discriminadores sobre un turno (y su contexto) y devuelve las
   * candidatas fusionadas y ordenadas. `undefined` si el scope no está indexado
   * (el llamador degrada al ranking del pipeline).
   */
  async rank(scopeKey: string, currentText: string, ctxText?: string): Promise<CRanking | undefined> {
    const idx = this.indexes.get(scopeKey);
    if (!idx || idx.toolIndex.length === 0) return undefined;

    const arch = await this.ensureArchetypes();
    const nowVec = (await this.engine.embedQuery(currentText, undefined, { high: true })).embedding;
    const nowClauses = await this.clauseTokenSets(currentText);
    const ctxVec = ctxText
      ? (await this.engine.embedQuery(ctxText, undefined, { high: true })).embedding
      : undefined;
    const ctxClauses = ctxText ? await this.clauseTokenSets(ctxText) : [];

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

    // Compuerta de autonomía: pico claro (margen) Y no ser un turno de apoyo
    // anafórico (que debe resolverse con el contexto).
    let anaphora = 0;
    for (const a of arch.anaphora) {
      const s = EmbeddingEngineService.cosine(nowVec, a);
      if (s > anaphora) anaphora = s;
    }
    const autonomous = margin >= C_MARGIN && anaphora < C_ANAF_TAU;

    const { state } = this.detectState(nowVec, arch, autonomous);
    const s3 = state === "confirm" ? 1 : state === "neutral" ? 0.5 : 0;

    // Recall por unión: top-K del turno actual ∪ top-K del contexto.
    const byName = (m: Map<string, number>) =>
      [...m.entries()].sort((a, b) => b[1] - a[1]).map(([n]) => n);
    const topNow = byName(cNow).slice(0, C_RECALL_K);
    const topCtx = hasCtx ? byName(cCtx).slice(0, C_RECALL_K) : [];
    const names = [...new Set([...topNow, ...topCtx])];

    const mentioned = toolNamesInText(currentText, idx.names);

    const out: CCandidate[] = names
      .map((name) => {
        const cn = cNow.get(name)!;
        const cc = cCtx.get(name)!;
        const toolTok = idx.toolTokens.get(name) ?? [];
        let s1: number;
        let s2: number;
        if (autonomous) {
          s1 = cn;
          s2 = this.msClause(nowClauses, toolTok);
        } else {
          s1 = cc;
          s2 = ctxClauses.length ? this.msClause(ctxClauses, toolTok) : 0;
        }
        const w1 = autonomous ? C_W1_AUTO : C_W1_DEP;
        const w2 = autonomous ? C_W2_AUTO : C_W2_DEP;
        return { name, s1, s2, s3, cNow: cn, cCtx: cc, fused: w1 * s1 + w2 * s2 + C_W3 * s3 };
      })
      .sort((a, b) => b.fused - a.fused);

    return { out, state, margin, maxNow, meanNow, anaphora, autonomous, hasCtx, mentioned };
  }

  /**
   * Arbitraje ponderado: la MENCIÓN EXPLÍCITA ratifica (orden directa); un
   * rechazo (D3) ⇒ ∅; un turno DEPENDIENTE sin contexto ⇒ ∅; el mejor
   * `fused < C_TAU` ⇒ ∅; si no, se aceptan los candidatos dentro de δ.
   */
  arbitrate(res: CRanking): { decision: CDecision; tools: string[] } {
    const top = res.out[0];
    if (res.mentioned.length > 0) return { decision: "ratify", tools: res.mentioned };
    if (res.state === "reject") return { decision: "abstain", tools: [] };
    if (!res.autonomous && !res.hasCtx) return { decision: "abstain", tools: [] };
    if (!top || top.fused < C_TAU) return { decision: "abstain", tools: [] };
    const tools = res.out.filter((f) => f.fused >= top.fused - C_DELTA).map((f) => f.name);
    return { decision: "ratify", tools };
  }

  // ── D3: estado por arquetipos ────────────────────────────────────────────
  private detectState(
    nowVec: Float32Array,
    arch: Archetypes,
    autonomous: boolean,
  ): { state: StateVerdict; reject: number; confirm: number } {
    let reject = 0;
    for (const a of arch.reject) {
      const s = EmbeddingEngineService.cosine(nowVec, a);
      if (s > reject) reject = s;
    }
    let confirm = 0;
    for (const a of arch.confirm) {
      const s = EmbeddingEngineService.cosine(nowVec, a);
      if (s > confirm) confirm = s;
    }
    // Un turno AUTÓNOMO con contenido es neutral por defecto: D3 no veta.
    if (autonomous) return { state: "neutral", reject, confirm };
    if (reject >= C_REJ_TAU && reject >= confirm) return { state: "reject", reject, confirm };
    if (confirm >= C_CONF_TAU && confirm > reject) return { state: "confirm", reject, confirm };
    return { state: "neutral", reject, confirm };
  }

  // ── D2: MaxSim por cláusula ──────────────────────────────────────────────
  /** Divide el turno en unidades de intención y devuelve sus tokens. */
  private async clauseTokenSets(text: string): Promise<Float32Array[][]> {
    const parts = text
      .split(C_CLAUSE_SPLIT)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const units = parts.length > 0 ? parts : [text];
    return Promise.all(units.map((u) => this.engine.embedTokens(u, undefined, { high: true })));
  }

  /** MaxSim por cláusula: el MÁXIMO (no la media) entre las cláusulas del turno. */
  private msClause(clauses: Float32Array[][], toolTok: Float32Array[]): number {
    let best = 0;
    for (const c of clauses) {
      const s = maxsim(c, toolTok);
      if (s > best) best = s;
    }
    return best;
  }

  private async ensureArchetypes(): Promise<Archetypes> {
    if (this.archetypes) return this.archetypes;
    const embed = (texts: string[]): Promise<Float32Array[]> =>
      Promise.all(
        texts.map(async (s) => (await this.engine.embedPassage(s, undefined, { high: true })).embedding),
      );
    try {
      this.archetypes = {
        reject: await embed(C_REJECT_ARCH),
        confirm: await embed(C_CONFIRM_ARCH),
        anaphora: await embed(C_ANAPHORA_ARCH),
      };
    } catch (err) {
      warn(`[varianteC] arquetipos D3 degradados: ${String((err as Error)?.message ?? err)}`);
      this.archetypes = { reject: [], confirm: [], anaphora: [] };
    }
    return this.archetypes;
  }
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
