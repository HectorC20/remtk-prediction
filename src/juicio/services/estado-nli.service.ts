/**
 * NLI de Estado Condicionado (juicio §1): rechazo y confirmación de la
 * propuesta pendiente SIN regex, modelado como inferencia de lenguaje natural
 * dependiente del estado:
 *
 *   premisa   = "El asistente propone: <acción pendiente de la sesión>"
 *   hipótesis = "El usuario aprueba y autoriza proceder con la acción propuesta."
 *   NLI(premisa + entrada, hipótesis) → entailment | contradiction | neutral
 *
 * Dos motores, mismo contrato:
 *  - `JUICIO_NLI_MODEL_PATH` apunta a un cross-encoder NLI ONNX (p. ej.
 *    mDeBERTa XNLI, logits [contradiction, neutral, entailment]): veredicto
 *    por softmax de los logits del par premisa/hipótesis.
 *  - Fallback por arquetipos embebidos (coseno, multi-idioma): pools
 *    REJECT vs CONFIRM con margen. Más débil que el cross-encoder pero sin
 *    dependencias nuevas; el acondicionamiento al estado (solo se evalúa si
 *    hay propuesta pendiente) es lo que evita el falso `new_query`.
 */
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import type { AppConfig } from "src/config";
import { log, warn } from "src/logger";
import type { JuicioEstado } from "src/shared/interfaces/juicio.interface";
import type { OnnxModuleLike, OnnxSessionLike, TokenizerLike, TokenizerModuleLike } from "src/shared/interfaces";
import { NLI_HYPOTHESIS, REJECT_ARCHETYPES } from "src/shared/constants/juicio";
import { CONFIRM_ARCHETYPES } from "src/shared/constants/messages";
import { onnxMaxTokens } from "src/shared/constants/predict/embedding.constants";

const nodeRequire = createRequire(__filename);

/** Orden de logits del modelo XNLI estándar (mDeBERTa-mnli-xnli). */
const LABEL = { contradiction: 0, neutral: 1, entailment: 2 } as const;

export class EstadoNliService {
  private session?: OnnxSessionLike;
  private tokenizer?: TokenizerLike;
  private loadAttempted = false;
  private onnx?: OnnxModuleLike;
  private tokenizerModule?: TokenizerModuleLike;

  private rejectEmbs?: Float32Array[];
  private confirmEmbs?: Float32Array[];

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly config: AppConfig,
  ) {}

  /**
   * Veredicto sobre la propuesta pendiente. `premise` es la descripción de la
   * acción pendiente (propuesta del asistente o resumen del plan cacheado).
   */
  async verdict(entryText: string, premise: string): Promise<JuicioEstado> {
    if (await this.nliReady()) {
      return this.verdictOnnx(premise, entryText);
    }
    return this.verdictArchetypes(entryText);
  }

  // ── Motor cross-encoder NLI (opcional) ─────────────────────────────────────

  private async nliReady(): Promise<boolean> {
    if (this.session && this.tokenizer) return true;
    if (this.loadAttempted) return false;
    this.loadAttempted = true;
    const dir = this.config.juicioNliModelPath;
    if (!dir) return false;
    try {
      const modelPath = join(dir, "model.onnx");
      if (!existsSync(modelPath)) {
        warn(`[juicio:nli] modelo no encontrado: ${modelPath} (fallback arquetipos)`);
        return false;
      }
      this.onnx ??= nodeRequire("onnxruntime-node") as OnnxModuleLike;
      this.tokenizerModule ??= nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;
      const tokenizerJson = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")) as Record<string, unknown>;
      const tokenizerConfig = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8")) as Record<string, unknown>;
      this.session = await this.onnx.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
      this.tokenizer = new this.tokenizerModule.Tokenizer(tokenizerJson, tokenizerConfig);
      log(`[juicio:nli] cross-encoder NLI cargado (${modelPath})`);
      return true;
    } catch (err) {
      warn(`[juicio:nli] carga falló (fallback arquetipos): ${String((err as Error)?.message ?? err)}`);
      return false;
    }
  }

  /** Softmax de los logits del par (premisa+entrada, hipótesis) como dos segmentos. */
  private async verdictOnnx(premise: string, entryText: string): Promise<JuicioEstado> {
    try {
      const encA = this.tokenizer!.encode(`${premise} El usuario responde: "${entryText}"`);
      const encB = this.tokenizer!.encode(NLI_HYPOTHESIS);
      // [CLS] A [SEP] + cuerpo de B [SEP] (sin el CLS de B), tipos 0/1 por segmento.
      const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, onnxMaxTokens);
      const types = [
        ...encA.ids.map(() => 0),
        ...encB.ids.slice(1).map(() => 1),
      ].slice(0, onnxMaxTokens);
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
      const logits = Object.values(outputs)[0] as { data: Float32Array };
      const p = softmax3(logits.data);
      log(
        `[juicio:nli] entail=${p[LABEL.entailment].toFixed(3)} neutral=${p[LABEL.neutral].toFixed(3)} contra=${p[LABEL.contradiction].toFixed(3)}`,
      );
      if (p[LABEL.contradiction] >= p[LABEL.entailment] && p[LABEL.contradiction] > p[LABEL.neutral]) return "reject";
      if (p[LABEL.entailment] > p[LABEL.neutral]) return "confirm";
      return "neutral";
    } catch (err) {
      warn(`[juicio:nli] inferencia falló: ${String((err as Error)?.message ?? err)}`);
      return "neutral";
    }
  }

  // ── Fallback por arquetipos embebidos (coseno, sin estado externo) ────────

  private async verdictArchetypes(entryText: string): Promise<JuicioEstado> {
    await this.ensurePools();
    const input = await this.engine.embedQuery(entryText, "small", { high: true });
    if (input.model === "hash") return "neutral";

    const maxVs = (pool: Float32Array[]): number => {
      let m = 0;
      for (const emb of pool) {
        const s = EmbeddingEngineService.cosine(input.embedding, emb);
        if (s > m) m = s;
      }
      return m;
    };
    const sReject = maxVs(this.rejectEmbs!);
    const sConfirm = maxVs(this.confirmEmbs!);
    const margin = this.config.juicioNliMargin;
    log(`[juicio:nli] arquetipos reject=${sReject.toFixed(3)} confirm=${sConfirm.toFixed(3)}`);
    if (sReject >= sConfirm + margin) return "reject";
    if (sConfirm >= sReject + margin) return "confirm";
    return "neutral";
  }

  private async ensurePools(): Promise<void> {
    if (this.rejectEmbs && this.confirmEmbs) return;
    const [rejects, confirms] = await Promise.all([
      Promise.all(REJECT_ARCHETYPES.map((a) => this.engine.embedQuery(a, "small").then((r) => r.embedding))),
      Promise.all(CONFIRM_ARCHETYPES.map((a) => this.engine.embedQuery(a, "small").then((r) => r.embedding))),
    ]);
    this.rejectEmbs = rejects;
    this.confirmEmbs = confirms;
  }
}

function softmax3(logits: Float32Array): [number, number, number] {
  const m = Math.max(logits[0] ?? 0, logits[1] ?? 0, logits[2] ?? 0);
  const e0 = Math.exp((logits[0] ?? 0) - m);
  const e1 = Math.exp((logits[1] ?? 0) - m);
  const e2 = Math.exp((logits[2] ?? 0) - m);
  const z = e0 + e1 + e2;
  return [e0 / z, e1 / z, e2 / z];
}
