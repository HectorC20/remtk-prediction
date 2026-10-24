import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function benchmark(dir: string, label: string) {
  console.log(`\n=== Benchmarking ${label} ===`);
  const tokenizerJson = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8"));
  const tokenizerConfig = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8"));
  const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);

  const session = await InferenceSession.create(join(dir, "model.onnx"), {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });

  const query = "No envíes el correo todavía, mejor guárdalo como borrador";
  const tools = [
    { name: "save_draft", text: "guardar borrador de correo o documento sin enviar" },
    { name: "send_email", text: "enviar correo electrónico a destinatarios de forma inmediata" },
    { name: "database_query", text: "consultar base de datos relacional SQL" },
  ];

  // Batch encode
  const pairs = tools.map((t) => {
    const encA = tokenizer.encode(query);
    const encB = tokenizer.encode(t.text);
    return [...encA.ids, ...encB.ids.slice(1)].slice(0, 64);
  });

  const maxLen = Math.max(...pairs.map((p) => p.length));
  const batchSize = pairs.length;
  const inputIds = new BigInt64Array(batchSize * maxLen);
  const mask = new BigInt64Array(batchSize * maxLen);
  const tokenTypeIds = new BigInt64Array(batchSize * maxLen);

  for (let b = 0; b < batchSize; b++) {
    const p = pairs[b];
    for (let i = 0; i < maxLen; i++) {
      const idx = b * maxLen + i;
      if (i < p.length) {
        inputIds[idx] = BigInt(p[i]);
        mask[idx] = 1n;
      } else {
        inputIds[idx] = 0n;
        mask[idx] = 0n;
      }
      tokenTypeIds[idx] = 0n;
    }
  }

  const feeds: Record<string, any> = {
    input_ids: new Tensor("int64", inputIds, [batchSize, maxLen]),
    attention_mask: new Tensor("int64", mask, [batchSize, maxLen]),
  };
  if (session.inputNames.includes("token_type_ids")) {
    feeds.token_type_ids = new Tensor("int64", tokenTypeIds, [batchSize, maxLen]);
  }

  // Warmup batch
  for (let i = 0; i < 5; i++) {
    await session.run(feeds);
  }

  // Measure batched execution
  const runs = 20;
  let totalMs = 0;
  for (let r = 0; r < runs; r++) {
    const t0 = performance.now();
    await session.run(feeds);
    totalMs += performance.now() - t0;
  }
  const avgBatched = totalMs / runs;
  console.log(`Inferencia en BATCH (3 pares paralelos en CPU):`);
  console.log(`Tiempo promedio total para 3 candidatos: ${avgBatched.toFixed(2)} ms`);
  console.log(`Tiempo amortizado por candidato: ${(avgBatched / 3).toFixed(2)} ms`);
}

async function main() {
  await benchmark("models/ms-marco-minilm-l6-int8-onnx", "MiniLM-L6 INT8 (23 MB)");
  await benchmark("models/nli-deberta-v3-small-int8-onnx", "DeBERTa-v3-small INT8 (160 MB)");
}

main().catch(console.error);
