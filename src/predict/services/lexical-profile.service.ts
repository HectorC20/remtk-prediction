/**
 * LexicalProfileService: cuarta señal del ranking — perfil léxico aprendido por
 * canal (ver docs/entrenamiento-prediccion.md).
 *
 * No usa ONNX, no usa batch y no reentrena nada: es un refuerzo de pesos
 * término → herramienta con IDF del propio canal y decaimiento exponencial.
 * Toda la estructura vive en `Map<scopeKey, LexicalProfile>`, así que el
 * aprendizaje de un canal es físicamente incapaz de alterar la predicción de
 * otro (§9.1).
 *
 *   POST /tools            → seedScope   (semilla del catálogo, §7)
 *   POST /predict/feedback → observe     (refuerzo/penalización por uso, §6.3)
 *   POST /predict          → score       (lectura normalizada, §6.2)
 */
import { createHash } from "node:crypto";
import type { AppConfig } from "src/config";
import { log } from "src/logger";
import type { ToolDefinition } from "src/shared/interfaces/domain.interface";
import type {
  LearnedScores,
  LexicalProfile,
  LexicalProfileSnapshot,
  LexicalSeedResult,
  LexicalSource,
  ToolLexicon,
} from "src/shared/interfaces/lexical.interface";
import { LEARN_DAY_MS } from "src/shared/constants/predict/index";
import { extractQueryKeywords, normalizeToken, toolKeywords } from "../keywords";

/** Modo de escritura de un peso: `set` (semilla) o `add` (refuerzo acumulativo). */
type GrantMode = "set" | "add";

export class LexicalProfileService {
  /** Un perfil por canal. Nunca se comparte entre canales. */
  private readonly profiles = new Map<string, LexicalProfile>();

  constructor(private readonly config: AppConfig) {}

  // ─────────────────────────────────────────────────────────────────────
  // Escritura: semilla del catálogo (disparador = alta de herramientas, §7)
  // ─────────────────────────────────────────────────────────────────────

  /**
   * Aplica la semilla del catálogo al perfil del canal. Idempotente por
   * `seededHash`: re-registrar el mismo catálogo no rehace trabajo. No pisa los
   * términos de origen `learned` y poda las herramientas dadas de baja.
   */
  seedScope(scopeKey: string, tools: ToolDefinition[]): LexicalSeedResult {
    const result: LexicalSeedResult = { seeded: 0, cached: 0, pruned: 0 };
    if (!this.config.learnEnabled) return result;

    const profile = this.profileFor(scopeKey);
    const now = Date.now();

    const names = new Set<string>();
    for (const tool of tools) {
      const name = (tool?.name ?? "").trim();
      if (name) names.add(name);
    }

    // 5. Podado de bajas: tools que estaban en el perfil y ya no están en el
    // catálogo dejan de puntuar por términos huérfanos.
    for (const name of [...profile.tools.keys()]) {
      if (!names.has(name)) this.removeTool(profile, name);
    }

    // 1-4. Semilla por tool, respetando lo aprendido e idempotente por hash.
    for (const tool of tools) {
      const name = (tool?.name ?? "").trim();
      if (!name) continue;
      const terms = this.seedTerms(tool);
      const hash = createHash("sha1").update(terms.join("|")).digest("hex");
      const prev = profile.tools.get(name);
      if (prev && prev.seededHash === hash) {
        result.cached++;
        continue;
      }

      const lexicon = prev ?? {
        terms: new Map<string, number>(),
        sources: new Map<string, LexicalSource>(),
        seededHash: "",
        lastDecay: now,
      };
      this.decayTool(lexicon, now);
      // Solo se re-siembran los términos de origen `seed`: el uso real manda.
      for (const [term, source] of [...lexicon.sources.entries()]) {
        if (source === "seed") this.revoke(profile, lexicon, name, term);
      }
      for (const term of terms) {
        if (lexicon.sources.get(term) === "learned") continue;
        this.grant(profile, lexicon, name, term, this.config.learnSeedWeight, "seed", "set");
      }
      lexicon.seededHash = hash;
      profile.tools.set(name, lexicon);
      result.seeded++;
    }

    if (tools.length > 0) profile.version++;
    profile.updatedAt = now;
    result.pruned = this.prune(scopeKey);
    log(
      `[lexical] seed scope=${scopeKey} seeded=${result.seeded} cached=${result.cached} ` +
        `pruned=${result.pruned} tools=${profile.tools.size} v=${profile.version}`,
    );
    return result;
  }

  // ─────────────────────────────────────────────────────────────────────
  // Escritura: señal de refuerzo (§6.3)
  // ─────────────────────────────────────────────────────────────────────

  /**
   * Aprende de un evento de uso: refuerza los términos de la consulta en las
   * herramientas que funcionaron y los penaliza en las que no. Devuelve el
   * total de señales acumuladas del canal.
   */
  observe(scopeKey: string, prompt: string, used: string[], rejected: string[] = []): number {
    const existing = this.profiles.get(scopeKey);
    if (!this.config.learnEnabled) return existing?.events ?? 0;

    const usedNames = sanitizeNames(used);
    const rejectedNames = sanitizeNames(rejected);
    if (usedNames.length === 0 && rejectedNames.length === 0) return existing?.events ?? 0;

    const terms = extractQueryKeywords(prompt);
    if (terms.length === 0) return existing?.events ?? 0;

    const profile = this.profileFor(scopeKey);
    const now = Date.now();

    for (const name of usedNames) {
      const lexicon = this.lexiconFor(profile, name, now);
      this.decayTool(lexicon, now);
      for (const term of terms) {
        this.grant(
          profile,
          lexicon,
          name,
          term,
          this.config.learnEta * this.idf(profile, term),
          "learned",
          "add",
        );
      }
    }

    for (const name of rejectedNames) {
      const lexicon = profile.tools.get(name);
      if (!lexicon) continue;
      this.decayTool(lexicon, now);
      for (const term of terms) {
        const current = lexicon.terms.get(term);
        if (current === undefined) continue;
        const next = Math.max(0, current - this.config.learnNegativeGamma * this.idf(profile, term));
        if (next <= 0) this.revoke(profile, lexicon, name, term);
        else lexicon.terms.set(term, next);
      }
    }

    profile.events += usedNames.length + rejectedNames.length;
    profile.updatedAt = now;
    this.prune(scopeKey);
    log(
      `[lexical] observe scope=${scopeKey} used=${usedNames.length} rejected=${rejectedNames.length} ` +
        `terms=${terms.length} events=${profile.events}`,
    );
    return profile.events;
  }

  // ─────────────────────────────────────────────────────────────────────
  // Lectura: score aprendido de una consulta (§6.2)
  // ─────────────────────────────────────────────────────────────────────

  /**
   * Puntúa la consulta contra el perfil del canal. `weight` es el
   * `learnWeightEfectivo`: 0 si la capa está desactivada, el canal no tiene
   * perfil o aún no alcanzó `LEARN_MIN_EVENTS` (§6.5) — con él en 0 la
   * predicción es idéntica a la de hoy.
   */
  score(scopeKey: string, prompt: string): LearnedScores {
    const terms = extractQueryKeywords(prompt);
    const inactive: LearnedScores = { scores: new Map(), weight: 0, terms };
    if (!this.config.learnEnabled) return inactive;

    const profile = this.profiles.get(scopeKey);
    if (!profile || profile.tools.size === 0 || terms.length === 0) return inactive;

    const weight = profile.events >= this.config.learnMinEvents ? this.config.learnWeight : 0;
    if (weight <= 0) return { scores: new Map(), weight: 0, terms };

    const now = Date.now();
    const accum = new Map<string, number>();
    let budget = Math.max(0, this.config.learnMaxPostings);

    outer: for (const term of terms) {
      const posting = profile.postings.get(term);
      if (!posting) continue;
      const idf = this.idf(profile, term);
      for (const name of posting) {
        if (budget-- <= 0) break outer;
        const lexicon = profile.tools.get(name);
        if (!lexicon) continue;
        this.decayTool(lexicon, now);
        const w = lexicon.terms.get(term) ?? 0;
        if (w <= 0) continue;
        accum.set(name, (accum.get(name) ?? 0) + idf * w);
      }
    }
    if (accum.size === 0) return { scores: new Map(), weight: 0, terms };

    // Normalización a [0,1] por el máximo de la tanda, igual que el BM25 (§6.2).
    const max = Math.max(...accum.values());
    const scores = new Map<string, number>();
    for (const [name, value] of accum) scores.set(name, max > 0 ? value / max : 0);
    return { scores, weight, terms };
  }

  // ─────────────────────────────────────────────────────────────────────
  // Poda, decaimiento y utilidades internas
  // ─────────────────────────────────────────────────────────────────────

  /**
   * Poda el perfil del canal: elimina términos por debajo de
   * `LEARN_TERM_MIN_WEIGHT` y recorta cada tool a `LEARN_MAX_TERMS_PER_TOOL`
   * conservando los pesos más altos. Devuelve los términos eliminados.
   */
  prune(scopeKey: string): number {
    const profile = this.profiles.get(scopeKey);
    if (!profile) return 0;
    const minWeight = this.config.learnTermMinWeight;
    const maxTerms = Math.max(0, this.config.learnMaxTermsPerTool);
    let pruned = 0;

    for (const [name, lexicon] of profile.tools) {
      for (const [term, weight] of [...lexicon.terms.entries()]) {
        if (weight < minWeight) {
          this.revoke(profile, lexicon, name, term);
          pruned++;
        }
      }
      if (lexicon.terms.size > maxTerms) {
        const excess = [...lexicon.terms.entries()]
          .sort((a, b) => a[1] - b[1])
          .slice(0, lexicon.terms.size - maxTerms);
        for (const [term] of excess) {
          this.revoke(profile, lexicon, name, term);
          pruned++;
        }
      }
    }
    return pruned;
  }

  /** Decaimiento perezoso por tool: `w ← w · exp(−λ·Δt)` (§6.4). */
  private decayTool(lexicon: ToolLexicon, now: number): void {
    const days = (now - lexicon.lastDecay) / LEARN_DAY_MS;
    lexicon.lastDecay = now;
    if (days <= 0) return;
    const factor = Math.exp(-this.config.learnDecayLambda * days);
    if (factor === 1) return;
    for (const [term, weight] of lexicon.terms) lexicon.terms.set(term, weight * factor);
  }

  /** IDF del propio canal: un término presente en todas las tools no discrimina. */
  private idf(profile: LexicalProfile, term: string): number {
    const n = profile.tools.size;
    const df = profile.df.get(term) ?? 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  /** Añade peso a un término y mantiene df + índice invertido coherentes. */
  private grant(
    profile: LexicalProfile,
    lexicon: ToolLexicon,
    toolName: string,
    term: string,
    weight: number,
    source: LexicalSource,
    mode: GrantMode,
  ): void {
    const had = lexicon.terms.has(term);
    const next = (mode === "add" ? (lexicon.terms.get(term) ?? 0) : 0) + weight;
    lexicon.terms.set(term, next);
    lexicon.sources.set(term, source);
    if (had) return;
    profile.df.set(term, (profile.df.get(term) ?? 0) + 1);
    let posting = profile.postings.get(term);
    if (!posting) {
      posting = new Set<string>();
      profile.postings.set(term, posting);
    }
    posting.add(toolName);
  }

  /** Elimina un término de una tool y mantiene df + índice invertido coherentes. */
  private revoke(
    profile: LexicalProfile,
    lexicon: ToolLexicon,
    toolName: string,
    term: string,
  ): void {
    if (!lexicon.terms.has(term)) return;
    lexicon.terms.delete(term);
    lexicon.sources.delete(term);
    const df = profile.df.get(term) ?? 0;
    if (df <= 1) profile.df.delete(term);
    else profile.df.set(term, df - 1);
    const posting = profile.postings.get(term);
    if (!posting) return;
    posting.delete(toolName);
    if (posting.size === 0) profile.postings.delete(term);
  }

  private removeTool(profile: LexicalProfile, toolName: string): void {
    const lexicon = profile.tools.get(toolName);
    if (!lexicon) return;
    for (const term of [...lexicon.terms.keys()]) this.revoke(profile, lexicon, toolName, term);
    profile.tools.delete(toolName);
  }

  /**
   * Semilla de términos de una tool (§7.1), reusando lo que ya existe:
   * `toolKeywords` (tags + intentSummary) ∪ nombre ∪ grupo ∪ categoría ∪ claves
   * del `inputSchema`, todos normalizados como los términos de la consulta.
   */
  private seedTerms(tool: ToolDefinition): string[] {
    const raw: string[] = [...toolKeywords(tool)];
    raw.push(tool.name ?? "", tool.group ?? "", tool.category ?? "");
    for (const key of Object.keys(tool.inputSchema ?? {})) raw.push(key);

    const seen = new Set<string>();
    const out: string[] = [];
    for (const value of raw) {
      for (const part of String(value).split(/[^\p{L}\p{N}_-]+/u)) {
        const token = normalizeToken(part);
        // El mismo corte (> 2) que `extractQueryKeywords`, o la semilla nunca
        // coincidiría con los términos de la consulta.
        if (token.length <= 2 || seen.has(token)) continue;
        seen.add(token);
        out.push(token);
        if (out.length >= this.config.learnMaxTermsPerTool) return out;
      }
    }
    return out;
  }

  private profileFor(scopeKey: string): LexicalProfile {
    let profile = this.profiles.get(scopeKey);
    if (!profile) {
      profile = {
        scopeKey,
        version: 0,
        tools: new Map(),
        df: new Map(),
        postings: new Map(),
        events: 0,
        updatedAt: Date.now(),
      };
      this.profiles.set(scopeKey, profile);
    }
    return profile;
  }

  private lexiconFor(profile: LexicalProfile, name: string, now: number): ToolLexicon {
    let lexicon = profile.tools.get(name);
    if (!lexicon) {
      lexicon = {
        terms: new Map(),
        sources: new Map(),
        seededHash: "",
        lastDecay: now,
      };
      profile.tools.set(name, lexicon);
    }
    return lexicon;
  }

  // ─────────────────────────────────────────────────────────────────────
  // Observabilidad
  // ─────────────────────────────────────────────────────────────────────

  /** Resumen del perfil del canal (para `/debug`); `undefined` si no existe. */
  snapshot(scopeKey: string): LexicalProfileSnapshot | undefined {
    const profile = this.profiles.get(scopeKey);
    if (!profile) return undefined;
    let terms = 0;
    let learnedTools = 0;
    for (const lexicon of profile.tools.values()) {
      terms += lexicon.terms.size;
      for (const source of lexicon.sources.values()) {
        if (source === "learned") {
          learnedTools++;
          break;
        }
      }
    }
    return {
      scopeKey: profile.scopeKey,
      version: profile.version,
      events: profile.events,
      tools: profile.tools.size,
      terms,
      postings: profile.postings.size,
      updatedAt: new Date(profile.updatedAt).toISOString(),
      learnedTools,
    };
  }

  /** Herramientas con perfil en el canal (0 si el canal no tiene perfil). */
  count(scopeKey: string): number {
    return this.profiles.get(scopeKey)?.tools.size ?? 0;
  }
}

/** Normaliza una lista de nombres de herramienta (trim, sin vacíos, sin duplicados). */
function sanitizeNames(names: string[] | undefined): string[] {
  if (!Array.isArray(names)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of names) {
    if (typeof raw !== "string") continue;
    const name = raw.trim();
    if (name === "" || seen.has(name)) continue;
    seen.add(name);
    out.push(name);
  }
  return out;
}
