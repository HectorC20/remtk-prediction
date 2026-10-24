import { InferenceSession, Tensor } from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function main() {
  const e5Dir = "models/multilingual-e5-small-onnx";
  const ceDir = "models/ms-marco-minilm-l6-int8-onnx";

  const e5Tokenizer = new Tokenizer(
    JSON.parse(readFileSync(join(e5Dir, "tokenizer.json"), "utf8")),
    JSON.parse(readFileSync(join(e5Dir, "tokenizer_config.json"), "utf8"))
  );
  const e5Session = await InferenceSession.create(join(e5Dir, "model.onnx"), { executionProviders: ["cpu"] });

  const ceTokenizer = new Tokenizer(
    JSON.parse(readFileSync(join(ceDir, "tokenizer.json"), "utf8")),
    JSON.parse(readFileSync(join(ceDir, "tokenizer_config.json"), "utf8"))
  );
  const ceSession = await InferenceSession.create(join(ceDir, "model.onnx"), { executionProviders: ["cpu"] });

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

  async function scoreCE(query: string, toolText: string) {
    const encA = ceTokenizer.encode(query);
    const encB = ceTokenizer.encode(toolText);
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
    const out = await ceSession.run({
      input_ids: new Tensor("int64", inputIds, [1, len]),
      attention_mask: new Tensor("int64", mask, [1, len]),
      token_type_ids: new Tensor("int64", types, [1, len]),
    });
    const logit = (out.logits ?? Object.values(out)[0]).data[0] as number;
    return { logit, score: 1 / (1 + Math.exp(-logit)) };
  }

  const query = "crea el item Casas de Punta Sal Hotel Karibian, es un hotel, ES DE PUNTA SAL, la descripcion te lo inventas, acá está los servicios";

  const tItemCrear = "mitumbes_item_crear: Crea un ítem nuevo. Contenido del catálogo unificado: lugares, actividades, eventos, rutas y servicios (type: place|activity|event|route|service). Cada ítem tiene title/description/body por idioma { es, en, pt, qu }, clasificación por categoryId (nivel 1), subcategoryId (hoja) y zoneId, coordenadas, enlaces (links) e imagen de portada.";
  const tLugarCrear = "mitumbes_lugar_crear: Crea un lugar nuevo. Campos: title, description, excerpt, address, hours, price, services, howToGet, activities, source y body en es/en/pt/qu; slug legible para la URL; image o imageUrl para la portada.";

  const qVec = await embed("query: " + query);
  const vItem = await embed("passage: " + tItemCrear);
  const vLugar = await embed("passage: " + tLugarCrear);

  console.log("=== Diagnóstico de Caso 3 ===");
  console.log("Query:", query);
  console.log("\nFase 1 (Bi-Encoder Cosine Sim):");
  console.log("  mitumbes_item_crear :", dot(qVec, vItem).toFixed(4));
  console.log("  mitumbes_lugar_crear:", dot(qVec, vLugar).toFixed(4));

  console.log("\nFase 2 (Cross-Encoder con Query Completa):");
  const ceItem = await scoreCE(query, tItemCrear);
  const ceLugar = await scoreCE(query, tLugarCrear);
  console.log("  mitumbes_item_crear : Logit:", ceItem.logit.toFixed(3), "Score:", ceItem.score.toFixed(4));
  console.log("  mitumbes_lugar_crear: Logit:", ceLugar.logit.toFixed(3), "Score:", ceLugar.score.toFixed(4));

  // ¿Qué pasa si extraemos la acción nuclear vs los parámetros complementarios?
  // Acción nuclear: "crea el item Casas de Punta Sal Hotel Karibian"
  // Parámetros: "es un hotel, ES DE PUNTA SAL, la descripcion te lo inventas, acá está los servicios"
  console.log("\nFase 2 con Acción Nuclear ('crea el item Casas de Punta Sal Hotel Karibian'):");
  const nuclear = "crea el item Casas de Punta Sal Hotel Karibian";
  const ceItemNuc = await scoreCE(nuclear, tItemCrear);
  const ceLugarNuc = await scoreCE(nuclear, tLugarCrear);
  console.log("  mitumbes_item_crear : Logit:", ceItemNuc.logit.toFixed(3), "Score:", ceItemNuc.score.toFixed(4));
  console.log("  mitumbes_lugar_crear: Logit:", ceLugarNuc.logit.toFixed(3), "Score:", ceLugarNuc.score.toFixed(4));
}

main().catch(console.error);
