/**
 * VarianteCService: núcleo de discriminadores continuos multilingües (100%
 * `gte-multilingual-base`), conforme estrictamente a AGENT.md:
 *
 *   D1 · SOPORTE   (bi-encoder gte)     → recall por UNIÓN (top-K del turno ∪ top-K del contexto).
 *   D2 · PRECISIÓN (MaxSim ColBERT gte) → alineación token a token combinada con coseno denso por cláusula.
 *   D3 · ESTADO Y MODALIDAD (arquetipos gte):
 *        - Modalidad: Atenúa cláusulas contrafácticas / hipotéticas irreales.
 *        - Estructura: Aísla citas entre comillas como argumentos reportados sin fuerza de comando.
 *        - Polaridad: Veto continuo de herramientas asociadas a cláusulas de prohibición directa.
 *        - Accionabilidad: Compuerta continua contra small-talk / charlas informales o quejas casuales.
 *        - Régimen: Conmutación dinámica entre autónomo y dependiente (contexto).
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { log, warn } from "src/logger";
import { toolNamesInText } from "src/predict/keywords";
import type { ToolDefinition } from "src/shared/interfaces/domain.interface";
import {
  C_ANAF_TAU,
  C_ANAPHORA_ARCH,
  C_ARCH_ACTIONABLE,
  C_ARCH_CHATTER,
  C_ARCH_DIRECTIVE,
  C_ARCH_HYPOTHETICAL,
  C_ARCH_PROHIBITION,
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

export interface CCandidate {
  name: string;
  s1: number;
  s2: number;
  s3: number;
  cNow: number;
  cCtx: number;
  fused: number;
  affirmativeHit?: number;
}

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
  actionabilityDelta: number;
  vetoedTools: string[];
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
  directive: Float32Array[];
  hypothetical: Float32Array[];
  prohibition: Float32Array[];
  chatter: Float32Array[];
  actionable: Float32Array[];
}

export function coreSignature(t: ToolDefinition): string {
  const firstPara = (t.description.split(/\n\s*\n/)[0] ?? t.description).trim();
  return firstPara.split("(Complemento:")[0]!.trim();
}

export class VarianteCService {
  private readonly indexes = new Map<string, IndexSet>();
  private archetypes?: Archetypes;

  constructor(private readonly engine: EmbeddingEngineService) {}

  has(scopeKey: string): boolean {
    return this.indexes.has(scopeKey);
  }

  clearScope(scopeKey: string): void {
    this.indexes.delete(scopeKey);
  }

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

  async rank(scopeKey: string, currentText: string, ctxText?: string): Promise<CRanking | undefined> {
    const idx = this.indexes.get(scopeKey);
    if (!idx || idx.toolIndex.length === 0) return undefined;

    const arch = await this.ensureArchetypes();
    const nowVec = (await this.engine.embedQuery(currentText, undefined, { high: true })).embedding;

    // 1. COMPUERTA CONTINUA DE ACCIONABILIDAD (Detección de Small-Talk / Queja sin orden)
    const simChatter = this.maxCosine(nowVec, arch.chatter);
    const simActionable = this.maxCosine(nowVec, arch.actionable);
    const actionabilityDelta = simActionable - simChatter;

    // 2. SEGMENTACIÓN ESTRUCTURAL DE CLÁUSULAS
    const nowClauses = this.segmentClauses(currentText);
    const nowClauseEmbeddings = await Promise.all(
      nowClauses.map(async (c) => (await this.engine.embedQuery(c.text, undefined, { high: true })).embedding)
    );
    const nowClauseTokens = await Promise.all(
      nowClauses.map(async (c) => this.engine.embedTokens(c.text, undefined, { high: true }))
    );

    // Contexto (si existe)
    const ctxVec = ctxText
      ? (await this.engine.embedQuery(ctxText, undefined, { high: true })).embedding
      : undefined;
    const ctxClauses = ctxText ? this.segmentClauses(ctxText) : [];
    const ctxClauseTokens = await Promise.all(
      ctxClauses.map(async (c) => this.engine.embedTokens(c.text, undefined, { high: true }))
    );

    // 3. ANÁLISIS DE MODALIDAD Y POLARIDAD POR CLÁUSULA
    const clauseModalityWeights: number[] = [];
    const clauseProhibitionFlags: boolean[] = [];

    for (let i = 0; i < nowClauses.length; i++) {
      const c = nowClauses[i];
      if (c.isQuoted) {
        clauseModalityWeights.push(0.15); // Cita entre comillas atenuada (reported speech)
        clauseProhibitionFlags.push(false);
        continue;
      }

      const cVec = nowClauseEmbeddings[i];
      const simDir = this.maxCosine(cVec, arch.directive);
      const simHyp = this.maxCosine(cVec, arch.hypothetical);
      const simProh = this.maxCosine(cVec, arch.prohibition);

      const isHypothetical = (simHyp - simDir) > 0.02;
      clauseModalityWeights.push(isHypothetical ? 0.10 : 1.0);

      const isProhibition = simProh >= 0.68 && simProh > simDir;
      clauseProhibitionFlags.push(isProhibition);
    }

    // 4. D1 & D2: RECALL Y ALINEACIÓN COMBINADA POR CLÁUSULA
    const cNow = new Map<string, number>();
    const cCtx = new Map<string, number>();
    const msNowMap = new Map<string, number>();
    const toolAffirmativeHits = new Map<string, number>();
    const vetoedTools = new Set<string>();

    let maxMsNow = 0;

    for (const t of idx.toolIndex) {
      cNow.set(t.name, EmbeddingEngineService.cosine(nowVec, t.vec));
      cCtx.set(t.name, ctxVec ? EmbeddingEngineService.cosine(ctxVec, t.vec) : 0);

      const tTokens = idx.toolTokens.get(t.name) ?? [];
      let maxAffirmativeScore = 0;
      let maxProhibitiveScore = 0;

      for (let i = 0; i < nowClauses.length; i++) {
        const ms = this.maxsim(nowClauseTokens[i], tTokens);
        const clauseDense = EmbeddingEngineService.cosine(nowClauseEmbeddings[i], t.vec);
        const clauseCombined = 0.40 * clauseDense + 0.60 * ms;
        const weight = clauseModalityWeights[i];
        const isProhibition = clauseProhibitionFlags[i];

        if (isProhibition) {
          if (clauseCombined > maxProhibitiveScore) maxProhibitiveScore = clauseCombined;
        } else {
          const effective = clauseCombined * weight;
          if (effective > maxAffirmativeScore) maxAffirmativeScore = effective;
        }
      }

      toolAffirmativeHits.set(t.name, maxAffirmativeScore);
      msNowMap.set(t.name, maxAffirmativeScore);
      if (maxAffirmativeScore > maxMsNow) maxMsNow = maxAffirmativeScore;

      // Veto continuo si la prohibición supera la orden afirmativa
      if (maxProhibitiveScore >= 0.70 && maxProhibitiveScore > maxAffirmativeScore) {
        vetoedTools.add(t.name);
      }
    }

    const nowVals = idx.toolIndex.map((t) => cNow.get(t.name)!);
    const maxNow = Math.max(...nowVals);
    const meanNow = nowVals.reduce((a, b) => a + b, 0) / nowVals.length;
    const margin = maxNow - meanNow;
    const hasCtx = ctxVec !== undefined;

    // Compuerta de autonomía
    let anaphora = 0;
    for (const a of arch.anaphora) {
      const s = EmbeddingEngineService.cosine(nowVec, a);
      if (s > anaphora) anaphora = s;
    }
    const autonomous = (margin >= C_MARGIN || maxMsNow >= 0.70) && anaphora < C_ANAF_TAU;

    const { state } = this.detectState(nowVec, arch, autonomous);
    const s3 = state === "confirm" ? 1 : state === "neutral" ? 0.5 : 0;

    // Recall por unión
    const byName = (m: Map<string, number>) =>
      [...m.entries()].sort((a, b) => b[1] - a[1]).map(([n]) => n);
    const topNow = [...new Set([...byName(cNow).slice(0, C_RECALL_K), ...byName(msNowMap).slice(0, C_RECALL_K)])];
    const topCtx = hasCtx ? byName(cCtx).slice(0, C_RECALL_K) : [];
    const names = [...new Set([...topNow, ...topCtx])];

    const mentioned = toolNamesInText(currentText, idx.names);

    const out: CCandidate[] = names
      .filter((name) => !vetoedTools.has(name))
      .map((name) => {
        const cn = cNow.get(name)!;
        const cc = cCtx.get(name)!;
        const toolTok = idx.toolTokens.get(name) ?? [];
        let s1: number;
        let s2: number;
        if (autonomous) {
          s1 = cn;
          s2 = msNowMap.get(name) ?? 0;
        } else {
          s1 = cc;
          s2 = ctxClauseTokens.length ? this.msClause(ctxClauseTokens, toolTok) : 0;
        }
        const w1 = autonomous ? C_W1_AUTO : C_W1_DEP;
        const w2 = autonomous ? C_W2_AUTO : C_W2_DEP;
        const affirmativeHit = toolAffirmativeHits.get(name) ?? 0;
        return {
          name,
          s1,
          s2,
          s3,
          cNow: cn,
          cCtx: cc,
          fused: w1 * s1 + w2 * s2 + C_W3 * s3,
          affirmativeHit,
        };
      })
      .filter((c) => !autonomous || (c.affirmativeHit !== undefined && c.affirmativeHit >= 0.45) || mentioned.includes(c.name))
      .sort((a, b) => b.fused - a.fused);

    return {
      out,
      state,
      margin,
      maxNow,
      meanNow,
      anaphora,
      autonomous,
      hasCtx,
      mentioned,
      actionabilityDelta,
      vetoedTools: [...vetoedTools],
    };
  }

  arbitrate(res: CRanking): { decision: CDecision; tools: string[] } {
    const top = res.out[0];
    if (res.mentioned.length > 0) return { decision: "ratify", tools: res.mentioned };
    if (res.state === "reject") return { decision: "abstain", tools: [] };
    if (!res.autonomous && !res.hasCtx) return { decision: "abstain", tools: [] };

    // Filtro de Small-talk / Charla casual sin orden ejecutiva
    if (res.actionabilityDelta < -0.05 && (!top || top.fused < 0.85)) {
      return { decision: "abstain", tools: [] };
    }

    if (!top || top.fused < C_TAU) return { decision: "abstain", tools: [] };

    // Banda multi-intención asistida por alineación afirmativa
    const bestAffirmative = Math.max(...res.out.map((f) => f.affirmativeHit ?? f.s2));
    const tools = res.out
      .filter((f) => {
        const aff = f.affirmativeHit ?? f.s2;
        return f.fused >= top.fused - C_DELTA && aff >= bestAffirmative - 0.08;
      })
      .map((f) => f.name);

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
    if (autonomous) return { state: "neutral", reject, confirm };
    if (reject >= C_REJ_TAU && reject >= confirm) return { state: "reject", reject, confirm };
    if (confirm >= C_CONF_TAU && confirm > reject) return { state: "confirm", reject, confirm };
    return { state: "neutral", reject, confirm };
  }

  // ── D2: MaxSim y segmentación sintáctica ──────────────────────────────────
  private segmentClauses(text: string): Array<{ text: string; isQuoted: boolean }> {
    const segments: Array<{ text: string; isQuoted: boolean }> = [];
    const quotePattern = /["'«“]([^"'»”]+)["'»”]/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = quotePattern.exec(text)) !== null) {
      const before = text.substring(lastIndex, match.index).trim();
      if (before) {
        const sub = before.split(/[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b|\bpero\b|\bbut\b/gi);
        for (const s of sub) {
          const clean = s.trim();
          if (clean.length > 0) segments.push({ text: clean, isQuoted: false });
        }
      }
      const quotedContent = match[1].trim();
      if (quotedContent.length > 0) {
        segments.push({ text: quotedContent, isQuoted: true });
      }
      lastIndex = match.index + match[0].length;
    }

    const rest = text.substring(lastIndex).trim();
    if (rest) {
      const sub = rest.split(/[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b|\bpero\b|\bbut\b/gi);
      for (const s of sub) {
        const clean = s.trim();
        if (clean.length > 0) segments.push({ text: clean, isQuoted: false });
      }
    }

    return segments.length > 0 ? segments : [{ text, isQuoted: false }];
  }

  private maxCosine(vec: Float32Array, archetypes: Float32Array[]): number {
    let max = -1;
    for (const a of archetypes) {
      const sim = EmbeddingEngineService.cosine(vec, a);
      if (sim > max) max = sim;
    }
    return max;
  }

  private msClause(clauses: Float32Array[][], toolTok: Float32Array[]): number {
    let best = 0;
    for (const c of clauses) {
      const s = this.maxsim(c, toolTok);
      if (s > best) best = s;
    }
    return best;
  }

  private maxsim(qTokens: Float32Array[], tTokens: Float32Array[]): number {
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
        directive: await embed(C_ARCH_DIRECTIVE),
        hypothetical: await embed(C_ARCH_HYPOTHETICAL),
        prohibition: await embed(C_ARCH_PROHIBITION),
        chatter: await embed(C_ARCH_CHATTER),
        actionable: await embed(C_ARCH_ACTIONABLE),
      };
    } catch (err) {
      warn(`[varianteC] arquetipos D3 degradados: ${String((err as Error)?.message ?? err)}`);
      this.archetypes = {
        reject: [],
        confirm: [],
        anaphora: [],
        directive: [],
        hypothetical: [],
        prohibition: [],
        chatter: [],
        actionable: [],
      };
    }
    return this.archetypes;
  }
}
