import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function main() {
  console.log("=== Test de Todas las Consultas Críticas con MiniLM-L6 Reranker INT8 ===");

  const miniDir = "models/ms-marco-minilm-l6-int8-onnx";
  const miniTJson = JSON.parse(readFileSync(join(miniDir, "tokenizer.json"), "utf8"));
  const miniTCfg = JSON.parse(readFileSync(join(miniDir, "tokenizer_config.json"), "utf8"));
  const miniTokenizer = new Tokenizer(miniTJson, miniTCfg);
  const miniSession = await InferenceSession.create(join(miniDir, "model.onnx"), { executionProviders: ["cpu"] });

  const tools = [
    {
      name: "mitumbes_lugar_crear",
      text: "mitumbes_lugar_crear: Crea un lugar turístico o establecimiento en el catálogo (hoteles, hospedajes, restaurantes). Parámetros requeridos: slug, title, description, address, categoryId, zoneId.",
    },
    {
      name: "mitumbes_evento_crear",
      text: "mitumbes_evento_crear: Crea un evento en el catálogo de MiTumbes. Parámetros requeridos: slug, title, dates, hours, price, zoneId.",
    },
    {
      name: "mitumbes_categoria_crear",
      text: "mitumbes_categoria_crear: Crea una nueva categoría de contenido en MiTumbes.",
    },
    {
      name: "mitumbes_item_actualizar",
      text: "mitumbes_item_actualizar: Actualiza los campos de un ítem existente en el catálogo unificado mediante su id o enlace. No crea ítems nuevos.",
    },
    {
      name: "mitumbes_item_obtener",
      text: "mitumbes_item_obtener: Obtiene los detalles de un ítem existente en el catálogo por su uuid o slug.",
    },
    {
      name: "mitumbes_item_listar",
      text: "mitumbes_item_listar: Lista ítems del catálogo unificado con filtros por tipo, categoría o zona.",
    },
    {
      name: "mitumbes_zona_listar",
      text: "mitumbes_zona_listar: Lista las zonas geográficas registradas (Punta Sal, Zorritos, Tumbes) para obtener el zoneId.",
    },
  ];

  const queries = [
    "CUALES SON LOS PARÁMETROS PARA CREAR EL ITEM",
    "NO EXISTE UNO PARA CREAR ITEM ?",
    "QUÉ DICES DE mitumbes_item_crear",
    "como creas los items",
    "como se crea el evento ?",
    "crea el item Casas de Punta Sal Hotel Karibian, es un hotel, ES DE PUNTA SAL, la descripcion te lo inventas, acá está los servicios",
  ];

  for (const query of queries) {
    console.log(`\n================================================================`);
    console.log(`Consulta: "${query}"`);
    console.log(`----------------------------------------------------------------`);
    const results = [];
    const t0 = performance.now();

    for (const tool of tools) {
      const encA = miniTokenizer.encode(query);
      const encB = miniTokenizer.encode(tool.text);
      const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, 128);
      const len = ids.length;
      const inputIds = new BigInt64Array(len);
      const mask = new BigInt64Array(len);
      const types = new BigInt64Array(len);
      for (let i = 0; i < len; i++) {
        inputIds[i] = BigInt(ids[i]);
        mask[i] = 1n;
        types[i] = BigInt(i < encA.ids.length ? 0 : 1);
      }
      const out = await miniSession.run({
        input_ids: new Tensor("int64", inputIds, [1, len]),
        attention_mask: new Tensor("int64", mask, [1, len]),
        token_type_ids: new Tensor("int64", types, [1, len]),
      });
      const logit = (out.logits ?? Object.values(out)[0]).data[0] as number;
      const score = 1 / (1 + Math.exp(-logit));
      results.push({ name: tool.name, logit, score });
    }
    const elapsed = performance.now() - t0;
    results.sort((a, b) => b.score - a.score);

    for (const r of results) {
      const bar = "█".repeat(Math.round(r.score * 20)).padEnd(20);
      console.log(`  ${r.name.padEnd(26)} | Logit: ${r.logit.toFixed(3).padStart(6)} | Score: ${r.score.toFixed(4)} [${bar}]`);
    }
    console.log(`Tiempo total para 7 tools evaluadas: ${elapsed.toFixed(1)} ms (${(elapsed / 7).toFixed(1)} ms / tool)`);
  }
}

main().catch(console.error);
