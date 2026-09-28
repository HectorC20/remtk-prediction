import { Tokenizer } from "@huggingface/tokenizers";
import { InferenceSession, Tensor } from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-production-tools";

async function main() {
  const ceDir = "models/ms-marco-minilm-l6-int8-onnx";
  const ceTJson = JSON.parse(readFileSync(join(ceDir, "tokenizer.json"), "utf8"));
  const ceTCfg = JSON.parse(readFileSync(join(ceDir, "tokenizer_config.json"), "utf8"));
  const tok = new Tokenizer(ceTJson, ceTCfg);
  const session = await InferenceSession.create(join(ceDir, "model.onnx"));

  async function score(query: string, tool: any): Promise<number> {
    const summary = tool.intentSummary ? `${tool.intentSummary}. ` : "";
    const passage = `${tool.name}: ${summary}${tool.description}`;
    const encA = tok.encode(query);
    const encB = tok.encode(passage);
    const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, 128);
    const types = [...encA.ids.map(() => 0), ...encB.ids.slice(1).map(() => 1)].slice(0, 128);
    const len = ids.length;
    const inputIds = new BigInt64Array(len);
    const attentionMask = new BigInt64Array(len);
    const tokenTypeIds = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(ids[i]);
      attentionMask[i] = 1n;
      tokenTypeIds[i] = BigInt(types[i]);
    }
    const out = await session.run({
      input_ids: new Tensor("int64", inputIds, [1, len]),
      attention_mask: new Tensor("int64", attentionMask, [1, len]),
      token_type_ids: new Tensor("int64", tokenTypeIds, [1, len]),
    });
    const logit = (out.logits ?? Object.values(out)[0]).data[0] as number;
    return 1 / (1 + Math.exp(-logit));
  }

  const query = "Verificar si existe una herramienta activa para crear ítems en el catálogo de mitumbes.com";
  const tools = [
    TOTAL_ACTIVE_TOOLS.find(t => t.name === "mitumbes_item_crear"),
    TOTAL_ACTIVE_TOOLS.find(t => t.name === "mitumbes_ayuda"),
    TOTAL_ACTIVE_TOOLS.find(t => t.name === "mitumbes_item_listar"),
    TOTAL_ACTIVE_TOOLS.find(t => t.name === "mitumbes_publicidad_click"),
  ];

  console.log(`Query: "${query}"\n`);
  for (const t of tools) {
    if (!t) continue;
    const s = await score(query, t);
    console.log(`  ${t.name.padEnd(28)} -> CrossScore: ${s.toFixed(4)}`);
  }
}

main().catch(console.error);
