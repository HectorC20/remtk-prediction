/**
 * KeywordService: capas 1 y 2 del pipeline de predicción.
 *
 * Capa 1 — Extracción de keywords salientes del prompt (frecuencia + stopwords).
 * Capa 2 — Match cross-idioma por embeddings: embebe las keywords de la query
 *          (en su idioma) y las keywords canónicas internas de cada tool (inglés),
 *          y puntúa por cosine. El modelo multilingual-e5-small mapea términos
 *          equivalentes a vectores cercanos, reemplazando una traducción explícita.
 *          Resultado: reduce drásticamente el catálogo al top-K más relevante.
 */
import { createHash } from "node:crypto";
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { log } from "src/logger";
import type { ScoredTool, ToolDefinition } from "src/shared/interfaces/domain.interface";
import { extractQueryKeywords, toolKeywords } from "../keywords";

export interface KeywordReduceResult {
  candidates: ScoredTool[];
  modelSize: string;
  recomputed: number;
  cached: number;
}

interface Entry {
  embedding: Float32Array;
  hash: string;
}

export class KeywordService {
  private readonly perTenant = new Map<string, Map<string, Entry>>();

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly topK: number,
  ) {}

  private mapFor(tenant: string): Map<string, Entry> {
    let m = this.perTenant.get(tenant);
    if (!m) {
      m = new Map();
      this.perTenant.set(tenant, m);
    }
    return m;
  }

  /** Registro/precalentamiento de keywords de las tools (prioridad baja). */
  async upsertTools(
    tenant: string,
    tools: ToolDefinition[],
  ): Promise<{ recomputed: number; cached: number }> {
    const map = this.mapFor(tenant);
    let recomputed = 0;
    let cached = 0;
    for (const tool of tools) {
      const doc = toolKeywords(tool).join(" ");
      const hash = createHash("sha1").update(doc).digest("hex");
      const prev = map.get(tool.name);
      if (prev && prev.hash === hash) {
        cached++;
        continue;
      }
      const res = await this.engine.embedPassage(doc || tool.name, "small", { high: false });
      map.set(tool.name, { embedding: res.embedding, hash });
      recomputed++;
    }
    return { recomputed, cached };
  }

  private async keywordEmbedding(
    tenant: string,
    tool: ToolDefinition,
  ): Promise<{ embedding: Float32Array; recomputed: boolean }> {
    const map = this.mapFor(tenant);
    const doc = toolKeywords(tool).join(" ");
    const hash = createHash("sha1").update(doc).digest("hex");
    const prev = map.get(tool.name);
    if (prev && prev.hash === hash) return { embedding: prev.embedding, recomputed: false };
    // En plena predicción: prioridad alta.
    const res = await this.engine.embedPassage(doc || tool.name, "small", { high: true });
    map.set(tool.name, { embedding: res.embedding, hash });
    return { embedding: res.embedding, recomputed: true };
  }

  /** Capa 2: reduce el catálogo al top-K por cosine cross-idioma de keywords. */
  async reduce(
    tenant: string,
    prompt: string,
    catalog: { all(): ToolDefinition[] },
  ): Promise<KeywordReduceResult> {
    const kws = extractQueryKeywords(prompt);
    const qText = kws.join(" ") || prompt;
    const q = await this.engine.embedQuery(qText, "small", { high: true });

    let recomputed = 0;
    let cached = 0;
    const scored: ScoredTool[] = [];
    for (const t of catalog.all()) {
      const r = await this.keywordEmbedding(tenant, t);
      if (r.recomputed) recomputed++;
      else cached++;
      scored.push({ name: t.name, score: EmbeddingEngineService.cosine(q.embedding, r.embedding) });
    }
    scored.sort((a, b) => b.score - a.score);
    const reduced = scored.slice(0, Math.max(1, this.topK));

    log(
      `[keywords] query=${JSON.stringify(kws)} reduced=${reduced.length}/${scored.length} ` +
        `model=${q.model} top1=${reduced[0]?.name}(${reduced[0]?.score.toFixed(3)})`,
    );
    return { candidates: reduced, modelSize: q.model, recomputed, cached };
  }

  count(tenant: string): number {
    return this.mapFor(tenant).size;
  }
}
