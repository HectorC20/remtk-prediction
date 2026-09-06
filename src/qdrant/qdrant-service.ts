/**
 * QdrantService: arquitectura Qdrant migrada de src (QdrantMemoryService +
 * ToolRetrievalService + KeywordingService) y del RecallService Go.
 * Orquesta: expansión por sinónimos → keywords (RAKE) → BM25 en paralelo
 * (mcp_tools + tool_keywords) → blend 60/40 → cruce con el catálogo del tenant.
 */
import type { AppConfig } from "../config";
import { log, warn } from "../logger";
import type { MemoryCandidate, MemoryDefinition, ScoredTool, ToolDefinition } from "../types";
import { QdrantClient, type SearchResult, uuidv5 } from "./qdrant-client";

const MAX_KEYWORDS = 8;
/** Tope de resultados por búsqueda BM25 (150 era excesivo para payloads grandes). */
const MAX_SEARCH_LIMIT = 50;
/** Tope de sinónimos añadidos a la query expandida. */
const MAX_SYNONYM_EXPANSIONS = 8;

/** Catálogo de herramientas por tenant (nombre → definición completa). */
export interface ToolCatalog {
  get(name: string): ToolDefinition | undefined;
  all(): ToolDefinition[];
}

export class QdrantService {
  readonly client: QdrantClient;
  private readonly catalogs = new Map<string, Map<string, ToolDefinition>>();
  private readonly synonymCache = new Map<string, string[]>();

  constructor(private readonly config: AppConfig) {
    this.client = new QdrantClient(config);
  }

  /** Registra/actualiza el catálogo del tenant (fuente de definiciones completas). */
  setCatalog(tenant: string, tools: ToolDefinition[]): void {
    const map = new Map<string, ToolDefinition>();
    for (const t of tools) map.set(t.name.trim(), t);
    this.catalogs.set(tenant, map);
  }

  catalog(tenant: string): ToolCatalog {
    const map = this.catalogs.get(tenant) ?? new Map<string, ToolDefinition>();
    return {
      get: (name: string) => map.get(name.trim()),
      all: () => [...map.values()],
    };
  }

  countTools(tenant: string): number {
    return this.catalogs.get(tenant)?.size ?? 0;
  }

  /** Crea las colecciones si no existen (no destructivo). */
  async ensureCollections(): Promise<void> {
    if (!this.client.enabled) return;
    for (const coll of [
      this.config.toolsCollection,
      this.config.keywordsCollection,
      this.config.synonymsCollection,
      this.config.memoriesCollection,
    ]) {
      try {
        await this.client.ensureSparseCollection(coll);
      } catch (err) {
        warn(`[qdrant] ensure ${coll} falló: ${String((err as Error)?.message ?? err)}`);
      }
    }
  }

  /**
   * Indexa las herramientas del tenant en Qdrant (colecciones mcp_tools y
   * tool_keywords) con el mismo payload que ToolRetrievalService.syncTools.
   */
  async indexTools(tenant: string, tools: ToolDefinition[]): Promise<void> {
    this.setCatalog(tenant, tools);
    if (!this.client.enabled) return;

    // IDs determinísticos (mismo esquema que el monolith): uuidv5 DNS.
    const toolPoints = tools.map((t) => ({
      id: uuidv5(`mcp-tool:${t.group || "general"}:${t.name}`),
      payload: {
        tenant,
        toolName: t.name,
        group: t.group,
        category: t.category,
        description: t.description,
        tags: t.tags ?? [],
        intentSummary: t.intentSummary,
        toolDocument: buildToolDocument(t),
      },
      vector: { bm25_text: this.client.textToSparseVector(buildToolDocument(t)) },
    }));

    // Una keyword-point por tool (mismo esquema tool-kw del monolith).
    const kwPoints: {
      id: string;
      payload: Record<string, unknown>;
      vector: Record<string, { indices: number[]; values: number[] }>;
    }[] = [];
    for (const t of tools) {
      const keywords = extractToolKeywords(t);
      if (keywords.length === 0) continue;
      kwPoints.push({
        id: uuidv5(`tool-kw:${t.group || "general"}:${t.name}`),
        payload: { tenant, toolName: t.name, keywords },
        vector: {
          bm25_text: this.client.textToSparseVector(keywords.join(" ")),
        },
      });
    }

    try {
      await this.client.upsertPoints(this.config.toolsCollection, toolPoints);
      if (kwPoints.length > 0) {
        await this.client.upsertPoints(this.config.keywordsCollection, kwPoints);
      }
      log(`[qdrant] indexados ${toolPoints.length} tools + ${kwPoints.length} keywords (tenant=${tenant})`);
    } catch (err) {
      warn(`[qdrant] indexado falló: ${String((err as Error)?.message ?? err)}`);
    }
  }

  /**
   * Capa 1 del pipeline: recall BM25. Replica RecallService.RetrieveTools del Go.
   * Devuelve candidatas con score crudo de Qdrant (0-∞); vacío/error → el
   * orquestador degrada al catálogo completo.
   */
  async retrieveTools(tenant: string, prompt: string, limit: number): Promise<ScoredTool[]> {
    const raw = prompt.trim();
    const catalog = this.catalog(tenant);
    if (raw === "" || catalog.all().length === 0 || !this.client.enabled) return [];

    // 1. Expansión por sinónimos (cold start → query intacta).
    const expanded = await this.expandSynonyms(raw);

    // 2. Keywords RAKE.
    const keywords = extractKeywordsRake(raw, MAX_KEYWORDS);

    log(`[recall] query=${JSON.stringify(raw)} expanded=${JSON.stringify(expanded)} keywords=${JSON.stringify(keywords)}`);

    // 3. BM25 en paralelo (mcp_tools + tool_keywords).
    const promptMatches = await this.search(
      this.config.toolsCollection,
      expanded,
      Math.min(Math.max(limit * 3, 10), MAX_SEARCH_LIMIT),
      undefined,
      ["toolName"],
    );
    const keywordMatches =
      keywords.length > 0
        ? await this.search(
            this.config.keywordsCollection,
            keywords.join(" "),
            Math.min(Math.max(limit * 2, 10), MAX_SEARCH_LIMIT),
            undefined,
            ["toolName"],
          )
        : [];

    // Score de keywords por toolName.
    const keywordScore = new Map<string, number>();
    for (const m of keywordMatches) {
      const name = payloadString(m, "toolName");
      if (!name) continue;
      const cur = keywordScore.get(name);
      if (cur === undefined || m.score > cur) keywordScore.set(name, m.score);
    }
    const hasKeywords = keywordMatches.length > 0;

    log(`[recall] prompt_matches=${promptMatches.length} keyword_matches=${keywordMatches.length}`);

    // 4. Merge + blend 60/40 y cruce con el catálogo.
    const merged = new Map<string, { tool: ToolDefinition; blended: number }>();
    for (const m of promptMatches) {
      const name = payloadString(m, "toolName");
      if (!name) continue;
      const tool = catalog.get(name);
      if (!tool) continue;
      const blended = hasKeywords ? m.score * 0.6 + (keywordScore.get(name) ?? 0) * 0.4 : m.score;
      merged.set(name, { tool, blended });
    }

    const entries = [...merged.values()].sort((a, b) => b.blended - a.blended);
    const out = entries.slice(0, limit).map((e) => ({ name: e.tool.name, score: e.blended }));
    log(`[recall] candidates=${out.length}`);
    return out;
  }

  /** Recall de memorias contextuales (BM25 filtrado por userId). Replica MemoryRecallService. */
  async searchMemories(userId: string, text: string, limit: number): Promise<MemoryCandidate[]> {
    const raw = text.trim();
    if (raw === "" || userId === "" || !this.client.enabled) return [];
    const filter = { must: [{ key: "userId", match: { value: userId } }] };
    const results = await this.search(
      this.config.memoriesCollection,
      raw,
      Math.min(Math.max(limit * 2, 10), MAX_SEARCH_LIMIT),
      filter,
      ["memoryId", "content", "chatId", "metadata", "createdAt"],
    );
    const out: MemoryCandidate[] = [];
    for (const r of results) {
      const id = payloadString(r, "memoryId") || r.id;
      const content = payloadString(r, "content");
      if (!id || !content.trim()) continue;
      out.push({
        memory: {
          id,
          content,
          chatId: payloadString(r, "chatId"),
          metadata: payloadObject(r, "metadata"),
          createdAt: payloadString(r, "createdAt"),
          score: 0,
        },
        retrievalScore: r.score,
      });
    }
    return out;
  }

  /** Expansión de sinónimos (colección query_synonyms). Replica SynonymExpander. */
  async expandSynonyms(query: string, toolNames: Set<string> = new Set()): Promise<string> {
    const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return query;
    const seen = new Set(tokens);
    const expansions: string[] = [];
    // Lookups en paralelo para no amplificar latencias de Qdrant.
    const perToken = await Promise.all(
      tokens.map(async (token) => {
        const clean = cleanToken(token);
        if (clean.length < 2) return [];
        return this.lookupSynonyms(clean);
      }),
    );
    for (const syns of perToken) {
      for (const syn of syns) {
        if (seen.size >= MAX_SYNONYM_EXPANSIONS) break;
        // Filtro de contaminación: un sinónimo no debe ser un nombre de tool
        // del catálogo ni una frase larga inyectada desde una descripción.
        const synTokens = syn.toLowerCase().split(/\s+/).filter(Boolean);
        if (synTokens.length > 5 || toolNames.has(syn.toLowerCase())) continue;
        if (!seen.has(syn)) {
          seen.add(syn);
          expansions.push(syn);
        }
      }
    }
    if (expansions.length === 0) return query;
    return `${query} ${expansions.join(" ")}`;
  }

  private async lookupSynonyms(token: string): Promise<string[]> {
    if (!this.client.enabled) return [];
    const cached = this.synonymCache.get(token);
    if (cached) return cached;
    try {
      const results = await this.search(
        this.config.synonymsCollection,
        token,
        3,
        undefined,
        ["synonyms"],
      );
      const seen = new Set<string>();
      const out: string[] = [];
      for (const r of results) {
        const syns = r.payload["synonyms"];
        if (Array.isArray(syns)) {
          for (const syn of syns) {
            if (typeof syn === "string" && syn !== "" && !seen.has(syn)) {
              seen.add(syn);
              out.push(syn);
            }
          }
        }
      }
      this.synonymCache.set(token, out);
      return out;
    } catch {
      return [];
    }
  }

  private async search(
    collection: string,
    query: string,
    limit: number,
    filter?: unknown,
    include?: string[],
  ): Promise<SearchResult[]> {
    try {
      return await this.client.searchBm25(collection, query, limit, filter, include);
    } catch (err) {
      warn(`[qdrant] search ${collection} falló: ${String((err as Error)?.message ?? err)}`);
      return [];
    }
  }

  async status(): Promise<{ enabled: boolean; qdrant: { ok: boolean; collections: string[] } }> {
    const q = await this.client.getStatus();
    return { enabled: this.client.enabled, qdrant: q };
  }
}

function payloadString(r: SearchResult, key: string): string {
  const v = r.payload?.[key];
  return typeof v === "string" ? v : "";
}

function payloadObject(r: SearchResult, key: string): Record<string, unknown> | undefined {
  const v = r.payload?.[key];
  return v && typeof v === "object" ? (v as Record<string, unknown>) : undefined;
}

function buildToolDocument(t: ToolDefinition): string {
  return [
    t.name,
    t.group,
    t.category,
    t.description,
    (t.tags ?? []).join(" "),
    t.intentSummary,
    JSON.stringify(t.inputSchema ?? {}),
  ]
    .filter(Boolean)
    .join(" ");
}

function extractToolKeywords(t: ToolDefinition): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const kw of [...(t.tags ?? []), ...(t.intentSummary ?? "").toLowerCase().split(/\s+/)]) {
    const clean = kw.toLowerCase().trim();
    if (clean.length > 1 && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out.slice(0, 12);
}

/** RAKE ligero (replica KeywordExtractor del Go). */
function extractKeywordsRake(prompt: string, limit: number): string[] {
  const normalized = prompt.toLowerCase().trim();
  if (normalized === "" || [...normalized].length < 3) return [];

  const phrases = normalized.split(/[^\p{L}\p{N}_\-\s]+/u).filter(Boolean);
  const wordFreq = new Map<string, number>();
  const wordDeg = new Map<string, number>();

  for (const phrase of phrases) {
    const words: string[] = [];
    for (const w of phrase.split(/\s+/)) {
      const clean = cleanToken(w);
      if ([...clean].length <= 2) continue;
      words.push(clean);
    }
    if (words.length === 0) continue;
    const unique = [...new Set(words)];
    for (const w of words) wordFreq.set(w, (wordFreq.get(w) ?? 0) + 1);
    for (const w of unique) wordDeg.set(w, (wordDeg.get(w) ?? 0) + (unique.length - 1));
  }

  if (wordFreq.size === 0) return [];
  const list = [...wordFreq.entries()]
    .map(([w, freq]) => ({ word: w, score: freq > 0 ? (wordDeg.get(w) ?? 0) / freq : 0 }))
    .sort((a, b) => b.score - a.score);
  return list.slice(0, limit).map((s) => s.word);
}

function cleanToken(w: string): string {
  return w.replace(/^[^\p{L}\p{N}_-]+/u, "").replace(/[^\p{L}\p{N}_-]+$/u, "");
}
