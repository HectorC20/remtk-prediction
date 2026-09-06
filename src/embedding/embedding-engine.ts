/**
 * EmbeddingEngine: empaqueta los dos modelos de embeddings (multilingual-e5-large
 * y multilingual-e5-small) en un solo proceso. Carga sesiones ONNX perezosamente
 * con onnxruntime-node + tokenizer @huggingface/tokenizers, mean-pooling y
 * normalización L2 (equivalente a OnnxNeuralFilterService del monolith NestJS).
 *
 * Si ONNX no está disponible o el modelo está incompleto, degrada a un embedding
 * determinístico por hashing-trick (igual que el fallback del monolith).
 */
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { AppConfig } from "../config";
import { log, warn } from "../logger";
import type { ModelSize } from "../types";

const nodeRequire = createRequire(__filename);

export const EMBED_DIMS: Record<ModelSize, number> = { small: 384 };
const MAX_TOKENS = 512;

export interface EmbedResult {
  embedding: Float32Array;
  model: ModelSize | "hash";
  dim: number;
  modelPath: string;
}

interface OnnxTensorLike {
  dims: number[];
}

interface OnnxSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
}

interface OnnxModuleLike {
  InferenceSession: {
    create(path: string, opts: Record<string, unknown>): Promise<OnnxSessionLike>;
  };
  Tensor: new (type: string, data: unknown, dims: number[]) => OnnxTensorLike;
}

interface TokenizerLike {
  encode(text: string): { ids: number[]; attention_mask: number[] };
}

interface TokenizerModuleLike {
  Tokenizer: new (
    json: Record<string, unknown>,
    config: Record<string, unknown>,
  ) => TokenizerLike;
}

function hashDjb2(token: string): number {
  let h = 0;
  for (let i = 0; i < token.length; i++) {
    h = ((h << 5) - h + token.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export class EmbeddingEngine {
  private sessions = new Map<ModelSize, OnnxSessionLike>();
  private tokenizers = new Map<ModelSize, TokenizerLike>();
  private attempts = new Set<ModelSize>();
  private loading = new Map<ModelSize, Promise<boolean>>();
  private onnx?: OnnxModuleLike;
  private tokenizerModule?: TokenizerModuleLike;

  /** Cola de inferencia con prioridad: los embeds de predicción (high) saltan
   *  por delante del recompute masivo de registro (low). */
  private queue: Array<{ fn: () => Promise<void>; high: boolean }> = [];
  private draining = false;
  /** Dedupe: un mismo texto+size en curso se reutiliza (evita cómputos dobles). */
  private inflight = new Map<string, Promise<EmbedResult>>();

  readonly defaultSize: ModelSize;

  constructor(private readonly config: AppConfig) {
    this.defaultSize = config.onnxModelSize;
  }

  get enabled(): boolean {
    return this.config.onnxEnabled;
  }

  modelDirName(size: ModelSize): string {
    return `multilingual-e5-${size}-onnx`;
  }

  modelOnnxPath(size: ModelSize): string {
    return join(this.config.onnxModelsPath, this.modelDirName(size), "model.onnx");
  }

  isReady(size: ModelSize): boolean {
    return this.sessions.has(size) && this.tokenizers.has(size);
  }

  /** Intenta cargar ambos modelos en background (warm-up no bloqueante). */
  warmup(): void {
    if (!this.config.onnxEnabled) {
      log("[embed] ONNX deshabilitado — se usará fallback hash");
      return;
    }
    for (const size of Object.keys(EMBED_DIMS) as ModelSize[]) {
      this.ensure(size)
        .then((ok) =>
          log(
            `[embed] warmup ${size} → ${ok ? "ONNX cargado" : "fallback hash"} (dim=${EMBED_DIMS[size]})`,
          ),
        )
        .catch((err) =>
          warn(`[embed] warmup ${size} falló: ${String(err?.message ?? err)}`),
        );
    }
  }

  async getInfo(size: ModelSize): Promise<{ model: string; dim: number; modelPath: string }> {
    const ready = this.isReady(size);
    return {
      model: ready ? size : "hash",
      dim: EMBED_DIMS[size],
      modelPath: ready ? join(this.config.onnxModelsPath, this.modelDirName(size)) : "",
    };
  }

  /**
   * Embedding con prioridad. La inferencia ONNX es CPU-bound y síncrona, así que
   * se encola: los embeds de predicción (high) saltan por delante del recompute
   * masivo de registro (low). Un mismo texto+size en curso se reutiliza (dedupe).
   */
  async embed(text: string, size?: ModelSize, opts?: { high?: boolean }): Promise<EmbedResult> {
    const s = size ?? this.defaultSize;
    if (this.config.onnxEnabled) {
      const key = `${s}\u0000${text}`;
      try {
        const ok = await this.ensure(s);
        if (ok) {
          const pending = this.inflight.get(key);
          if (pending) return pending;
          const p = this.enqueue(opts?.high ?? false, () => this.runModel(s, text))
            .then(
              (embedding) => ({
                embedding,
                model: s,
                dim: EMBED_DIMS[s],
                modelPath: join(this.config.onnxModelsPath, this.modelDirName(s)),
              }),
            )
            .catch((err) => {
              warn(`[embed] inferencia ${s} falló (degradando a hash): ${String((err as Error)?.message ?? err)}`);
              return {
                embedding: this.hashEmbedding(text, EMBED_DIMS[s]),
                model: "hash" as const,
                dim: EMBED_DIMS[s],
                modelPath: "",
              };
            });
          this.inflight.set(key, p);
          p.finally(() => {
            if (this.inflight.get(key) === p) this.inflight.delete(key);
          }).catch(() => undefined);
          return p;
        }
      } catch (err) {
        warn(`[embed] inferencia ${s} falló (degradando a hash): ${String((err as Error)?.message ?? err)}`);
      }
    }
    return {
      embedding: this.hashEmbedding(text, EMBED_DIMS[s]),
      model: "hash",
      dim: EMBED_DIMS[s],
      modelPath: "",
    };
  }

  /** Embedding con prefijo query: (para prompts de predicción). */
  async embedQuery(text: string, size?: ModelSize, opts?: { high?: boolean }): Promise<EmbedResult> {
    return this.embed(`query: ${text}`, size, opts);
  }

  /** Embedding con prefijo passage: (para contenido de tools/memorias). */
  async embedPassage(text: string, size?: ModelSize, opts?: { high?: boolean }): Promise<EmbedResult> {
    return this.embed(`passage: ${text}`, size, opts);
  }

  /** Encola una tarea de inferencia y resuelve cuando se ejecuta. */
  private enqueue<T>(high: boolean, fn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({
        high,
        fn: () => fn().then(resolve, reject),
      });
      this.drain();
    });
  }

  /** Procesa la cola en serie, tomando primero los items high (predicción). */
  private drain(): void {
    if (this.draining) return;
    this.draining = true;
    const next = (): void => {
      let idx = this.queue.findIndex((item) => item.high);
      if (idx < 0) idx = 0;
      const item = this.queue[idx];
      if (!item) {
        this.draining = false;
        return;
      }
      this.queue.splice(idx, 1);
      item.fn().then(next, next);
    };
    next();
  }

  private ensure(size: ModelSize): Promise<boolean> {
    if (this.isReady(size)) return Promise.resolve(true);
    if (this.attempts.has(size)) {
      // Carga ya iniciada: esperar la misma promesa (evita cargas duplicadas).
      return this.loading.get(size) ?? Promise.resolve(false);
    }
    this.attempts.add(size);
    const promise = this.doLoad(size);
    this.loading.set(size, promise);
    return promise;
  }

  private async doLoad(size: ModelSize): Promise<boolean> {
    const modelPath = this.modelOnnxPath(size);
    if (!existsSync(modelPath)) {
      warn(`[embed] modelo no encontrado: ${modelPath}`);
      return false;
    }

    this.onnx ??= nodeRequire("onnxruntime-node") as OnnxModuleLike;
    this.tokenizerModule ??= nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;

    const dir = join(this.config.onnxModelsPath, this.modelDirName(size));
    const tokenizerJson = JSON.parse(
      readFileSync(join(dir, "tokenizer.json"), "utf8"),
    ) as Record<string, unknown>;
    const tokenizerConfig = JSON.parse(
      readFileSync(join(dir, "tokenizer_config.json"), "utf8"),
    ) as Record<string, unknown>;

    const session = await this.onnx.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
    });
    const tokenizer = new this.tokenizerModule.Tokenizer(tokenizerJson, tokenizerConfig);

    this.sessions.set(size, session);
    this.tokenizers.set(size, tokenizer);
    log(`[embed] sesión ONNX cargada: ${size} (${modelPath})`);
    return true;
  }

  private async runModel(size: ModelSize, text: string): Promise<Float32Array> {
    const session = this.sessions.get(size)!;
    const tokenizer = this.tokenizers.get(size)!;

    const encoded = tokenizer.encode(text);
    const ids = encoded.ids.slice(0, MAX_TOKENS);
    const mask = (encoded.attention_mask ?? ids.map(() => 1)).slice(0, MAX_TOKENS);
    const len = ids.length;

    const inputIds = new BigInt64Array(len);
    const attentionMask = new BigInt64Array(len);
    const tokenTypeIds = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(ids[i]);
      attentionMask[i] = BigInt(mask[i]);
      tokenTypeIds[i] = 0n;
    }

    const feeds: Record<string, unknown> = {
      input_ids: new this.onnx!.Tensor("int64", inputIds, [1, len]),
      attention_mask: new this.onnx!.Tensor("int64", attentionMask, [1, len]),
      token_type_ids: new this.onnx!.Tensor("int64", tokenTypeIds, [1, len]),
    };
    const outputs = await session.run(feeds);

    const result = outputs["sentence_embedding"] ?? outputs["last_hidden_state"];
    if (!result) throw new Error(`sin salida de embedding (${Object.keys(outputs).join(",")})`);

    const data = (result as { data: Float32Array }).data;
    const dims = (result as OnnxTensorLike).dims;
    // [1, seq, dim] → mean-pooling por máscara; [1, dim] → directo.
    const vec = dims.length === 3 ? meanPool(data, dims[1], dims[2], mask) : data;
    return normalizeL2(vec);
  }

  private hashEmbedding(text: string, dim: number): Float32Array {
    const vec = new Float32Array(dim);
    const lower = text.toLowerCase();
    const tokens = lower
      .split(/[^a-záéíóúüñ0-9]+/i)
      .filter((t) => t.length > 1);
    for (const tok of tokens) {
      vec[hashDjb2(tok) % dim] += 1;
    }
    return normalizeL2(vec);
  }
}

function meanPool(data: Float32Array, seq: number, dim: number, mask: number[]): Float32Array {
  const out = new Float32Array(dim);
  let valid = 0;
  for (let s = 0; s < seq; s++) {
    if (mask[s] === 0) continue;
    valid++;
    const off = s * dim;
    for (let d = 0; d < dim; d++) {
      out[d] += data[off + d];
    }
  }
  if (valid > 1) {
    for (let d = 0; d < dim; d++) out[d] /= valid;
  }
  return out;
}

export function normalizeL2(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-9) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }
  return vec;
}

/** Producto punto con clamp 0..1 (equivalente a shared.Cosine del Go). */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  return Math.min(1, Math.max(0, dot));
}
