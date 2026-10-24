import type { ToolDefinition } from "./domain.interface";

export interface SparseVector {
  indices: number[];
  values: number[];
}

export interface QdrantPoint {
  id: string | number;
  payload: Record<string, unknown>;
  /** Mapa nombre → vector (p. ej. `{ bm25_text: { indices, values } }`). */
  vector: Record<string, SparseVector>;
}

export interface SearchResult {
  id: string;
  score: number;
  payload: Record<string, unknown>;
}

export interface ToolCatalog {
  get(name: string): ToolDefinition | undefined;
  all(): ToolDefinition[];
  has?(name: string): boolean;
  size?: number;
}
