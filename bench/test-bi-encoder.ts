import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function main() {
  console.log("=== Benchmarking Bi-Encoder (multilingual-e5-small) ===");
  const dir = "models/multilingual-e5-small-onnx";
  const tokenizerJson = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8"));
  const tokenizerConfig = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8"));
  const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);

  const session = await InferenceSession.create(join(dir, "model.onnx"), {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });

  const query = "query: No envíes el correo todavía, mejor guárdalo como borrador";
  const enc = tokenizer.encode(query);
  const len = enc.ids.length;
  const inputIds = new BigInt64Array(len);
  const mask = new BigInt64Array(len);
  const tokenTypeIds = new BigInt64Array(len);
  for (let i = 0; i < len; i++) {
    inputIds[i] = BigInt(enc.ids[i]);
    mask[i] = 1n;
    tokenTypeIds[i] = 0n;
  }
  const feeds: Record<string, any> = {
    input_ids: new Tensor("int64", inputIds, [1, len]),
    attention_mask: new Tensor("int64", mask, [1, len]),
  };
  if (session.inputNames.includes("token_type_ids")) {
    feeds.token_type_ids = new Tensor("int64", tokenTypeIds, [1, len]);
  }

  // Warmup
  for (let i = 0; i < 5; i++) await session.run(feeds);

  // Measure
  const runs = 30;
  let totalMs = 0;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await session.run(feeds);
    totalMs += performance.now() - t0;
  }
  const avg = totalMs / runs;
  console.log(`Tiempo promedio de embedding de consulta: ${avg.toFixed(2)} ms`);
}

main().catch(console.error);
