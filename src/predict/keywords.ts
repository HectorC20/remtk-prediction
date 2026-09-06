/**
 * Extracción de keywords compartida por las 3 capas del pipeline.
 * - `extractQueryKeywords`: keywords salientes del prompt (frecuencia + stopwords).
 * - `toolKeywords`: keywords canónicas internas de la tool (tags + intentSummary).
 * El match cross-idioma NO es léxico aquí: lo hace KeywordService por embeddings.
 */
import { STOPWORDS } from "src/shared/dictionary/stopwords.dictionary";
import type { ToolDefinition } from "../shared/interfaces/domain.interface";

/** Stopwords ES/EN comunes para descartar tokens no informativos. */

export function normalizeToken(raw: string): string {
  return raw
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N}_-]+/gu, "")
    .trim();
}

/**
 * Extrae keywords salientes del prompt: descarta stopwords y tokens cortos,
 * ordena por frecuencia (palabras repetidas primero) y luego por longitud.
 */
export function extractQueryKeywords(prompt: string): string[] {
  const freq = new Map<string, number>();
  for (const raw of prompt.split(/[^\p{L}\p{N}_-]+/u)) {
    const w = normalizeToken(raw);
    if (w.length > 2 && !STOPWORDS.has(w)) {
      freq.set(w, (freq.get(w) ?? 0) + 1);
    }
  }
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
    .map(([w]) => w);
}

/** Keywords canónicas internas de la tool (tags + intentSummary), deduplicadas. */
export function toolKeywords(tool: ToolDefinition): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const sources = [...(tool.tags ?? [])];
  for (const w of (tool.intentSummary ?? "").split(/\s+/)) sources.push(w);
  for (const raw of sources) {
    const w = normalizeToken(raw);
    if (w.length > 2 && !STOPWORDS.has(w) && !seen.has(w)) {
      seen.add(w);
      out.push(w);
    }
  }
  return out;
}
