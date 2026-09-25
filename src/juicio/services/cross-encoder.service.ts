/**
 * Cross-Encoder de juicio sintáctico (juicio §4, provider ONNX opcional):
 * negación sin máscaras léxicas.
 *
 * A diferencia del bi-encoder (pooling independiente), aquí consulta y
 * descripción de la herramienta se concatenan como dos segmentos:
 *
 *   [CLS] consulta [SEP] descripción_tool [SEP]
 *
 * La autoatención cruzada token a token deja que los tokens de polaridad
 * negativa anulen la activación de los verbos siguientes ("NO envíes el correo"
 * → el logit de send_email cae, el de save_draft se mantiene).
 *
 * Se activa con `JUICIO_RERANKER_MODEL_PATH` apuntando a una carpeta con
 * `model.onnx` + tokenizer (p. ej. un reranker multilingual tipo bge-reranker).
 * Sin modelo, el servicio queda inactivo y la capa de juicio sigue con las
 * demás señales (MaxSim + especificidad + abstención).
 */
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { AppConfig } from "src/config";
import { log, warn } from "src/logger";
import type { OnnxModuleLike, OnnxSessionLike, TokenizerLike, TokenizerModuleLike } from "src/shared/interfaces";
import { onnxMaxTokens } from "src/shared/constants/predict/embedding.constants";

const nodeRequire = createRequire(__filename);

export class CrossEncoderService {
  private session?: OnnxSessionLike;
  private tokenizer?: TokenizerLike;
  private loadAttempted = false;
  private onnx?: OnnxModuleLike;
  private tokenizerModule?: TokenizerModuleLike;

  constructor(private readonly config: AppConfig) {}

  /** Carga perezosa; true si el reranker está operativo. */
  async ready(): Promise<boolean> {
    if (this.session && this.tokenizer) return true;
    if (this.loadAttempted) return false;
    this.loadAttempted = true;
    const dir = this.config.juicioRerankerModelPath;
    if (!dir) return false;
    try {
      const modelPath = join(dir, "model.onnx");
      if (!existsSync(modelPath)) {
        warn(`[juicio:reranker] modelo no encontrado: ${modelPath} (inactivo)`);
        return false;
      }
      this.onnx ??= nodeRequire("onnxruntime-node") as OnnxModuleLike;
      this.tokenizerModule ??= nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;
      const tokenizerJson = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")) as Record<string, unknown>;
      const tokenizerConfig = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8")) as Record<string, unknown>;
      this.session = await this.onnx.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
      this.tokenizer = new this.tokenizerModule.Tokenizer(tokenizerJson, tokenizerConfig);
      log(`[juicio:reranker] cross-encoder cargado (${modelPath})`);
      return true;
    } catch (err) {
      warn(`[juicio:reranker] carga falló (inactivo): ${String((err as Error)?.message ?? err)}`);
      return false;
    }
  }

  /**
   * Logit sigmoideo del par (consulta, descripción de herramienta) como dos
   * segmentos. Devuelve undefined si el par no se pudo puntuar.
   */
  async score(query: string, passage: string): Promise<number | undefined> {
    if (!(await this.ready())) return undefined;
    try {
      const encA = this.tokenizer!.encode(query);
      const encB = this.tokenizer!.encode(passage);
      const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, onnxMaxTokens);
      const types = [...encA.ids.map(() => 0), ...encB.ids.slice(1).map(() => 1)].slice(0, onnxMaxTokens);
      const len = ids.length;
      const inputIds = new BigInt64Array(len);
      const attentionMask = new BigInt64Array(len);
      const tokenTypeIds = new BigInt64Array(len);
      for (let i = 0; i < len; i++) {
        inputIds[i] = BigInt(ids[i]);
        attentionMask[i] = 1n;
        tokenTypeIds[i] = BigInt(types[i]);
      }
      const feeds: Record<string, unknown> = {
        input_ids: new this.onnx!.Tensor("int64", inputIds, [1, len]),
        attention_mask: new this.onnx!.Tensor("int64", attentionMask, [1, len]),
      };
      if (!this.session!.inputNames || this.session!.inputNames.includes("token_type_ids")) {
        feeds.token_type_ids = new this.onnx!.Tensor("int64", tokenTypeIds, [1, len]);
      }
      const outputs = await this.session!.run(feeds);
      const out = Object.values(outputs)[0] as { data: Float32Array };
      const logit = out.data[0] ?? 0;
      return 1 / (1 + Math.exp(-logit));
    } catch (err) {
      warn(`[juicio:reranker] inferencia falló: ${String((err as Error)?.message ?? err)}`);
      return undefined;
    }
  }
}
