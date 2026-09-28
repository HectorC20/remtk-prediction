import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function main() {
  const e5Dir = "models/multilingual-e5-small-onnx";

  const e5Tokenizer = new Tokenizer(
    JSON.parse(readFileSync(join(e5Dir, "tokenizer.json"), "utf8")),
    JSON.parse(readFileSync(join(e5Dir, "tokenizer_config.json"), "utf8"))
  );
  const e5Session = await InferenceSession.create(join(e5Dir, "model.onnx"), { executionProviders: ["cpu"] });

  async function embed(text: string) {
    const enc = e5Tokenizer.encode(text);
    const len = enc.ids.length;
    const inputIds = new BigInt64Array(len);
    const mask = new BigInt64Array(len);
    const types = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(enc.ids[i]);
      mask[i] = 1n;
      types[i] = 0n;
    }
    const out = await e5Session.run({
      input_ids: new Tensor("int64", inputIds, [1, len]),
      attention_mask: new Tensor("int64", mask, [1, len]),
      token_type_ids: new Tensor("int64", types, [1, len]),
    });
    const data = (out.last_hidden_state ?? Object.values(out)[0]).data as Float32Array;
    const vec = new Float32Array(384);
    for (let i = 0; i < len; i++) {
      for (let d = 0; d < 384; d++) vec[d] += data[i * 384 + d];
    }
    let norm = 0;
    for (let d = 0; d < 384; d++) {
      vec[d] /= len;
      norm += vec[d] * vec[d];
    }
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < 384; d++) vec[d] /= norm;
    return vec;
  }

  function dot(a: Float32Array, b: Float32Array) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  const pCat = await embed("query: categoryId");
  const pZone = await embed("query: zoneId");

  const tCatList = await embed("passage: mitumbes_categoria_listar: Lista categorías del sistema.");
  const tZoneList = await embed("passage: mitumbes_zona_listar: Lista zonas del sistema.");
  const tItemList = await embed("passage: mitumbes_item_listar: Lista ítems del catálogo.");

  console.log("=== Afinidad Vectorial en Espacio Latente (Sin Keywords) ===");
  console.log("categoryId -> mitumbes_categoria_listar :", dot(pCat, tCatList).toFixed(4));
  console.log("categoryId -> mitumbes_zona_listar      :", dot(pCat, tZoneList).toFixed(4));
  console.log("categoryId -> mitumbes_item_listar      :", dot(pCat, tItemList).toFixed(4));
  console.log("");
  console.log("zoneId     -> mitumbes_zona_listar       :", dot(pZone, tZoneList).toFixed(4));
  console.log("zoneId     -> mitumbes_categoria_listar :", dot(pZone, tCatList).toFixed(4));
  console.log("zoneId     -> mitumbes_item_listar      :", dot(pZone, tItemList).toFixed(4));
}

main();
