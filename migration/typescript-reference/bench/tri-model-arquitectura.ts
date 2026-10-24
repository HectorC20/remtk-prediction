/**
 * Arquitectura Cognitiva de 3 Modelos Continuos (AGENT.md Compliant):
 *
 * Modelo 1 (M1 - Bi-Encoder Multilingüe Continuo / GTE):
 *   - Proyección densa y Recall@K de candidatos viables sobre el catálogo.
 *   - Memoria latente y análisis de similitud semántica continua.
 *
 * Modelo 2 (M2 - Cross-Encoder / Reranker de Atención Cruzada Token a Token):
 *   - Interacción profunda [CLS] Query [SEP] Tool_Description [SEP].
 *   - Capta la negación formal y resuelve falsos positivos (AGENT.md §2.B).
 *
 * Modelo 3 (M3 - Árbitro Lógico / NLI de 3 vías [mDeBERTa-v3 / DeBERTa]):
 *   - Inferencia de contradicción lógica y anulación contextual.
 *   - Detecta si una acción posterior anula a una previa o si hay rechazo/prohibición.
 *
 * Cero regexes de palabras clave, cero listas de verbos/sustantivos.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import type {
  OnnxModuleLike,
  OnnxSessionLike,
  TokenizerLike,
  TokenizerModuleLike,
  ToolDefinition,
} from "../src/shared/interfaces";
import { BENCH_TOOLS } from "./catalog";
import { BENCH_CASES, type BenchCase, type BenchTurn } from "./cases";
import { RemtkBenchClient } from "./remtk.client";

const nodeRequire = createRequire(__filename);
const MAX_SEQ_LEN = 256;

// ── MODELO 2: CROSS-ENCODER DE ATENCIÓN CRUZADA TOKEN A TOKEN ────────────────
export class CrossEncoderModel {
  private session?: OnnxSessionLike;
  private tokenizer?: TokenizerLike;
  private onnx?: OnnxModuleLike;
  private readyPromise?: Promise<boolean>;

  constructor(private readonly dir: string) {}

  async ready(): Promise<boolean> {
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = (async () => {
      const modelPath = join(this.dir, "model.onnx");
      if (!existsSync(modelPath)) return false;
      this.onnx = nodeRequire("onnxruntime-node") as OnnxModuleLike;
      const tm = nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;
      const tokJson = JSON.parse(readFileSync(join(this.dir, "tokenizer.json"), "utf8"));
      const tokCfg = JSON.parse(readFileSync(join(this.dir, "tokenizer_config.json"), "utf8"));
      this.session = await this.onnx.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
      this.tokenizer = new tm.Tokenizer(tokJson, tokCfg);
      return true;
    })();
    return this.readyPromise;
  }

  async score(query: string, document: string): Promise<number | undefined> {
    if (!(await this.ready())) return undefined;
    try {
      const encA = this.tokenizer!.encode(query);
      const encB = this.tokenizer!.encode(document);
      const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, MAX_SEQ_LEN);
      const types = [...encA.ids.map(() => 0), ...encB.ids.slice(1).map(() => 1)].slice(0, MAX_SEQ_LEN);
      const len = ids.length;
      const inputIds = new BigInt64Array(len);
      const mask = new BigInt64Array(len);
      const tokenTypes = new BigInt64Array(len);
      for (let i = 0; i < len; i++) {
        inputIds[i] = BigInt(ids[i]);
        mask[i] = 1n;
        tokenTypes[i] = BigInt(types[i]);
      }
      const feeds: Record<string, unknown> = {
        input_ids: new this.onnx!.Tensor("int64", inputIds, [1, len]),
        attention_mask: new this.onnx!.Tensor("int64", mask, [1, len]),
      };
      if (!this.session!.inputNames || this.session!.inputNames.includes("token_type_ids")) {
        feeds.token_type_ids = new this.onnx!.Tensor("int64", tokenTypes, [1, len]);
      }
      const out = await this.session!.run(feeds);
      const logit = (Object.values(out)[0] as { data: Float32Array }).data[0];
      return 1 / (1 + Math.exp(-logit)); // Sigmoide continuo
    } catch {
      return undefined;
    }
  }
}

// ── MODELO 3: ÁRBITRO NLI DE INFERENCIA LÓGICA (3 VÍAS) ──────────────────────
export class NliLogicModel {
  private session?: OnnxSessionLike;
  private tokenizer?: TokenizerLike;
  private onnx?: OnnxModuleLike;
  private idx = { ent: 0, neu: 1, con: 2 };
  private readyPromise?: Promise<boolean>;

  constructor(private readonly dir: string) {}

  async ready(): Promise<boolean> {
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = (async () => {
      const modelPath = join(this.dir, "model.onnx");
      if (!existsSync(modelPath)) return false;
      const cfg = JSON.parse(readFileSync(join(this.dir, "config.json"), "utf8")) as {
        id2label?: Record<string, string>;
      };
      for (const [k, v] of Object.entries(cfg.id2label ?? {})) {
        const l = v.toLowerCase();
        if (l.startsWith("entail")) this.idx.ent = Number(k);
        else if (l.startsWith("neutral")) this.idx.neu = Number(k);
        else if (l.startsWith("contra")) this.idx.con = Number(k);
      }
      this.onnx = nodeRequire("onnxruntime-node") as OnnxModuleLike;
      const tm = nodeRequire("@huggingface/tokenizers") as TokenizerModuleLike;
      const tokJson = JSON.parse(readFileSync(join(this.dir, "tokenizer.json"), "utf8"));
      const tokCfg = JSON.parse(readFileSync(join(this.dir, "tokenizer_config.json"), "utf8"));
      this.session = await this.onnx.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
      this.tokenizer = new tm.Tokenizer(tokJson, tokCfg);
      return true;
    })();
    return this.readyPromise;
  }

  async checkContradiction(premise: string, hypothesis: string): Promise<number> {
    if (!(await this.ready())) return 0;
    try {
      const encA = this.tokenizer!.encode(premise);
      const encB = this.tokenizer!.encode(hypothesis);
      const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, MAX_SEQ_LEN);
      const len = ids.length;
      const inputIds = new BigInt64Array(len);
      const mask = new BigInt64Array(len);
      for (let i = 0; i < len; i++) {
        inputIds[i] = BigInt(ids[i]);
        mask[i] = 1n;
      }
      const feeds = {
        input_ids: new this.onnx!.Tensor("int64", inputIds, [1, len]),
        attention_mask: new this.onnx!.Tensor("int64", mask, [1, len]),
      };
      const out = await this.session!.run(feeds);
      const logits = (Object.values(out)[0] as { data: Float32Array }).data;
      const m = Math.max(logits[0], logits[1], logits[2]);
      const exp = [Math.exp(logits[0] - m), Math.exp(logits[1] - m), Math.exp(logits[2] - m)];
      const sum = exp[0] + exp[1] + exp[2];
      return exp[this.idx.con] / sum; // Probabilidad continua de contradicción
    } catch {
      return 0;
    }
  }
}

// ── ORQUESTADOR TRI-MODELO INTEGRADO ─────────────────────────────────────────
export class TriModelOrchestrator {
  private crossEncoder: CrossEncoderModel;
  private nli: NliLogicModel;
  private seq = 0;

  constructor(
    private readonly remtk: RemtkBenchClient,
    private readonly engine: EmbeddingEngineService,
    private readonly catalog: ToolDefinition[],
  ) {
    this.crossEncoder = new CrossEncoderModel(resolve(process.cwd(), "models", "ms-marco-minilm-l6-int8-onnx"));
    this.nli = new NliLogicModel(resolve(process.cwd(), "models", "mdeberta-v3-base-xnli-int8-onnx"));
  }

  async predict(caseId: string, text: string, history: BenchTurn[]): Promise<{ tools: string[]; scores: Record<string, number> }> {
    // 0. Enmascarado de bloques no conversacionales (código markdown)
    const cleanText = text.replace(/```[\s\S]*?```|`[^`\n]*`/g, " ");

    // 1. MODELO 1: Recall@K denso y espacio continuo (GTE / E5)
    // Se ejecuta sobre el turno y el contexto
    const baseResp = await this.remtk.predict(`trimodel-${caseId}-${++this.seq}`, cleanText, history);
    const candidateTools = new Set<string>(baseResp.tools);

    // Si hay historial previo y el turno es corto/dependiente, aseguramos recall amplio
    if (history.length > 0) {
      const lastUser = [...history].reverse().find((h) => h.role === "user");
      if (lastUser) {
        const histResp = await this.remtk.predict(`trimodel-h-${caseId}-${++this.seq}`, lastUser.content, []);
        for (const t of histResp.tools) candidateTools.add(t);
      }
    }

    // 2. MODELO 2: Evaluación Cross-Encoder de Atención Cruzada profunda
    const scoredTools: { name: string; ceScore: number; contraProb: number }[] = [];
    const pool = Array.from(candidateTools);

    for (const toolName of pool) {
      const toolDef = this.catalog.find((t) => t.name === toolName);
      if (!toolDef) continue;
      const signature = `${toolDef.intentSummary || ""}. ${toolDef.description}`;

      // Cross-Encoder evalúa si el query afirma la tool o si está negada
      const ceScore = (await this.crossEncoder.score(cleanText, signature)) ?? 0.5;

      // 3. MODELO 3: NLI evalúa si la consulta contradice explícitamente ejecutar la tool
      const contraProb = await this.nli.checkContradiction(
        cleanText,
        `Esta solicitud aprueba realizar: ${toolDef.intentSummary || toolDef.description}`
      );

      scoredTools.push({ name: toolName, ceScore, contraProb });
    }

    // Ponderación matemática continua:
    // Si la probabilidad de contradicción es alta (NLI > 0.40), el score final se suprime.
    const finalScores: Record<string, number> = {};
    const selected: string[] = [];

    for (const item of scoredTools) {
      // Score continuo atenuado por contradicción
      const netScore = item.ceScore * (1 - item.contraProb);
      finalScores[item.name] = netScore;

      // Criterio de corte continuo (sin reglas ad-hoc)
      if (item.ceScore >= 0.35 && item.contraProb < 0.35) {
        selected.push(item.name);
      }
    }

    // Ordenamiento por afinidad neta
    selected.sort((a, b) => (finalScores[b] ?? 0) - (finalScores[a] ?? 0));

    return {
      tools: selected,
      scores: finalScores,
    };
  }
}

// ── RUNNER EVALUADOR ─────────────────────────────────────────────────────────
function score(c: BenchCase, tools: string[]): boolean {
  const picked = tools.slice(0, 3);
  if (c.expected.length === 0) return tools.length === 0;
  if (c.multi) return c.expected.every((e) => picked.includes(e));
  return tools[0] === c.expected[0] || (c.expected.includes(tools[0]) && c.expected.length > 1);
}

async function main() {
  console.log("=== EVALUANDO ARQUITECTURA DE 3 MODELOS CONTINUOS (AGENT.md) ===\n");
  const remtk = new RemtkBenchClient();
  await remtk.start(BENCH_TOOLS, { qdrant: false });
  const engine = new EmbeddingEngineService(loadConfig());
  const orchestrator = new TriModelOrchestrator(remtk, engine, BENCH_TOOLS);

  let okBase = 0;
  let okTri = 0;

  for (const c of BENCH_CASES) {
    const idx = c.turns.map((t) => t.role).lastIndexOf("user");
    const history = c.turns.slice(0, idx);
    const lastUserText = c.turns[idx].content;

    // Baseline actual
    const histBase: BenchTurn[] = [];
    let baseTools: string[] = [];
    for (const t of c.turns) {
      if (t.role === "user") baseTools = (await remtk.predict(`base-${c.id}`, t.content, [...histBase])).tools;
      histBase.push(t);
    }
    const passBase = score(c, baseTools);
    if (passBase) okBase++;

    // Tri-modelo
    const triRes = await orchestrator.predict(c.id, lastUserText, history);
    const passTri = score(c, triRes.tools);
    if (passTri) okTri++;

    const m = (p: boolean) => (p ? "✓" : "✗");
    console.log(
      `${m(passBase)} base | ${m(passTri)} trimod | ${c.id.padEnd(28)} esperado=[${c.expected.join(",") || "∅"}] ` +
        `trimod=[${triRes.tools.join(",")}]`
    );
  }

  await remtk.close();
  console.log("\n==================== RESUMEN ====================");
  console.log(`Baseline actual:     ${okBase}/${BENCH_CASES.length}`);
  console.log(`Arquitectura 3-Mod:  ${okTri}/${BENCH_CASES.length}`);
  console.log("=================================================");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error("Fallo:", err);
    process.exit(1);
  });
}
