import type { ModelSize } from "../constants/predict/version.constants";

export interface EmbedResult {
  embedding: Float32Array;
  model: ModelSize | "hash";
  dim: number;
  modelPath: string;
}

export interface OnnxTensorLike {
  dims: number[];
}

export interface OnnxSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export interface OnnxModuleLike {
  InferenceSession: {
    create(path: string, opts: Record<string, unknown>): Promise<OnnxSessionLike>;
  };
  Tensor: new (type: string, data: unknown, dims: number[]) => OnnxTensorLike;
}

export interface TokenizerLike {
  encode(text: string): { ids: number[]; attention_mask: number[] };
}

export interface TokenizerModuleLike {
  Tokenizer: new (
    json: Record<string, unknown>,
    config: Record<string, unknown>,
  ) => TokenizerLike;
}

export interface DebugStats {
  tenants: number;
  confirmGets: number;
  confirmHits: number;
  confirmSets: number;
  embeddingsRecomputed: number;
  embeddingsCached: number;
}