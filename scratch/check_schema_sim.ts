import { Tokenizer } from "@huggingface/tokenizers";
import { InferenceSession, Tensor } from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { MITUMBES_VERBATIM_TOOLS } from "../tests/mitumbes-item-creation-comparison.test";

async function main() {
  const e5Dir = "models/multilingual-e5-small-onnx";
  const e5TJson = JSON.parse(readFileSync(join(e5Dir, "tokenizer.json"), "utf8"));
  const e5TCfg = JSON.parse(readFileSync(join(e5Dir, "tokenizer_config.json"), "utf8"));
  const tok = new Tokenizer(e5TJson, e5TCfg);
  const session = await InferenceSession.create(join(e5Dir, "model.onnx"));

  async function embed(text: string): Promise<Float32Array> {
    const enc = tok.encode(text);
    const len = enc.ids.length;
    const inputIds = new BigInt64Array(len);
    const mask = new BigInt64Array(len);
    const types = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(enc.ids[i]);
      mask[i] = BigInt(enc.attention_mask[i]);
      types[i] = BigInt(enc.type_ids[i]);
    }
    const out = await session.run({
      input_ids: new Tensor("int64", inputIds, [1, len]),
      attention_mask: new Tensor("int64", mask, [1, len]),
      token_type_ids: new Tensor("int64", types, [1, len]),
    });
    const data = (out.last_hidden_state ?? Object.values(out)[0]).data as Float32Array;
    const dim = 384;
    const vec = new Float32Array(dim);
    let sumW = 0;
    for (let i = 0; i < len; i++) {
      const w = Number(mask[i]);
      sumW += w;
      for (let d = 0; d < dim; d++) vec[d] += data[i * dim + d] * w;
    }
    if (sumW > 0) for (let d = 0; d < dim; d++) vec[d] /= sumW;
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let d = 0; d < dim; d++) vec[d] /= norm;
    return vec;
  }

  function cosine(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }

  const catTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_categoria_listar")!;
  const zoneTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_zona_listar")!;
  const itemTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_item_crear")!;

  const catEmb = await embed(`passage: ${catTool.name}: ${catTool.intentSummary}. ${catTool.description}`);
  const zoneEmb = await embed(`passage: ${zoneTool.name}: ${zoneTool.intentSummary}. ${zoneTool.description}`);
  const itemEmb = await embed(`passage: ${itemTool.name}: ${itemTool.intentSummary}. ${itemTool.description}`);

  const qCat = await embed("query: categoryId categoria");
  const qZone = await embed("query: zoneId zona");

  console.log("Sim categoryId with mitumbes_categoria_listar:", cosine(qCat, catEmb).toFixed(4));
  console.log("Sim categoryId with mitumbes_item_crear:", cosine(qCat, itemEmb).toFixed(4));
  console.log("Sim zoneId with mitumbes_zona_listar:", cosine(qZone, zoneEmb).toFixed(4));
  console.log("Sim zoneId with mitumbes_item_crear:", cosine(qZone, itemEmb).toFixed(4));
}

main().catch(console.error);
