import type { ScoredTool } from "./domain.interface";

export interface KeywordReduceResult {
  candidates: ScoredTool[];
  modelSize: string;
  recomputed: number;
  cached: number;
}

export interface Entry {
  embedding: Float32Array;
  hash: string;
}

export type KeywordCacheEntry = Entry;
