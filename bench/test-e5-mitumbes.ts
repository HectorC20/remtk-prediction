import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function main() {
  console.log("=== Cosine Similarity con Multilingual E5-Small (Dense Latent Space) ===");

  const dir = "models/multilingual-e5-small-onnx";
  const tJson = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8"));
  const tCfg = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8"));
  const tokenizer = new Tokenizer(tJson, tCfg);
  const session = await InferenceSession.create(join(dir, "model.onnx"), { executionProviders: ["cpu"] });

  async function embed(text: string): Promise<Float32Array> {
    const enc = tokenizer.encode(text);
    const len = enc.ids.length;
    const inputIds = new BigInt64Array(len);
    const mask = new BigInt64Array(len);
    const types = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(enc.ids[i]);
      mask[i] = 1n;
      types[i] = 0n;
    }
    const out = await session.run({
      input_ids: new Tensor("int64", inputIds, [1, len]),
      attention_mask: new Tensor("int64", mask, [1, len]),
      token_type_ids: new Tensor("int64", types, [1, len]),
    });
    const data = (out.last_hidden_state ?? Object.values(out)[0]).data as Float32Array;
    const dim = 384;
    const vec = new Float32Array(dim);
    for (let i = 0; i < len; i++) {
      for (let d = 0; d < dim; d++) vec[d] += data[i * dim + d];
    }
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      vec[d] /= len;
      norm += vec[d] * vec[d];
    }
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) vec[d] /= norm;
    return vec;
  }

  function dot(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  const tools = [
    {
      name: "mitumbes_lugar_crear",
      text: "passage: mitumbes_lugar_crear: Crea un nuevo lugar turístico, hotel, hospedaje o establecimiento en el catálogo de MiTumbes. Parámetros requeridos y opcionales: slug, title, description, address, categoryId, zoneId, services, image.",
    },
    {
      name: "mitumbes_evento_crear",
      text: "passage: mitumbes_evento_crear: Crea un nuevo evento en el catálogo de MiTumbes. Parámetros: slug, title, dates, hours, price, zoneId.",
    },
    {
      name: "mitumbes_categoria_crear",
      text: "passage: mitumbes_categoria_crear: Crea una nueva categoría o subcategoría de contenido en MiTumbes.",
    },
    {
      name: "mitumbes_item_actualizar",
      text: "passage: mitumbes_item_actualizar: Actualiza los campos de un ítem existente en el catálogo unificado mediante su id o enlace. No crea ítems nuevos.",
    },
    {
      name: "mitumbes_item_obtener",
      text: "passage: mitumbes_item_obtener: Obtiene los detalles de un ítem existente en el catálogo por su uuid o slug.",
    },
    {
      name: "mitumbes_item_listar",
      text: "passage: mitumbes_item_listar: Lista ítems del catálogo unificado con filtros por tipo, categoría o zona.",
    },
    {
      name: "mitumbes_zona_listar",
      text: "passage: mitumbes_zona_listar: Lista las zonas geográficas registradas (Punta Sal, Zorritos, Tumbes) para obtener el zoneId.",
    },
  ];

  const toolEmbeddings = new Map<string, Float32Array>();
  for (const t of tools) {
    toolEmbeddings.set(t.name, await embed(t.text));
  }

  const queries = [
    "CUALES SON LOS PARÁMETROS PARA CREAR EL ITEM",
    "NO EXISTE UNO PARA CREAR ITEM ?",
    "QUÉ DICES DE mitumbes_item_crear",
    "como creas los items",
    "como se crea el evento ?",
    "crea el item Casas de Punta Sal Hotel Karibian, es un hotel, ES DE PUNTA SAL, la descripcion te lo inventas, acá está los servicios",
  ];

  for (const q of queries) {
    const qVec = await embed(`query: ${q}`);
    const scores = tools.map((t) => ({ name: t.name, sim: dot(qVec, toolEmbeddings.get(t.name)!) }));
    scores.sort((a, b) => b.sim - a.sim);
    console.log(`\nConsulta: "${q}"`);
    for (const s of scores.slice(0, 3)) {
      console.log(`  ${s.name.padEnd(26)} | Sim: ${s.sim.toFixed(4)}`);
    }
  }
}

main().catch(console.error);
