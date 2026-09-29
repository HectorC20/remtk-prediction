#!/usr/bin/env node
/**
 * ensure-onnx-models.mjs
 *
 * Descarga los modelos ONNX de embeddings que usa remtk-prediction desde
 * Hugging Face al directorio de modelos, solo si faltan (idempotente).
 *
 * El servidor usa e5-large (default del pipeline) y e5-small (opcional / liviano,
 * ONNX_MODEL_SIZE=small); bge-m3-int8 solo se usa en tests (embed-bge.test.ts) y
 * se descarga como extra.
 *
 * Los pesos NO se incrustan en la imagen Docker: viven en el volumen ./models
 * (bind mount) y se pueblan aquí en el primer arranque. No bloquea el boot:
 * ante cualquier error el servicio degrada al embedding determinístico (hash).
 *
 * Variables de entorno:
 *   ONNX_ENABLED       "false" desactiva la descarga (default: true)
 *   ONNX_MODELS_PATH   carpeta base de modelos (default: ./models)
 */
import { createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const HF = "https://huggingface.co";

/**
 * Modelos a asegurar. Cada entrada declara `dir` (carpeta destino) y `files`
 * (archivos mínimos para cargar el modelo). Un archivo puede ser:
 *   - string: se descarga de `${base}/${archivo}` con el mismo nombre.
 *   - { src, dest }: `src` es la URL completa y `dest` el nombre local, para
 *     repos que exponen los ONNX en un subdirectorio y el tokenizer en la raíz
 *     (o con otro nombre de archivo), p. ej. `onnx/model_quantized.onnx` -> `model.onnx`.
 *
 * Los servicios del juicio (cross-encoder / NLI) leen SIEMPRE `model.onnx`,
 * `tokenizer.json` y `tokenizer_config.json` del mismo directorio.
 */
const MODELS = [
  {
    name: "e5-large (default del pipeline)",
    dir: "multilingual-e5-large-onnx",
    base: `${HF}/intfloat/multilingual-e5-large/resolve/main/onnx`,
    files: ["model.onnx", "model.onnx_data", "tokenizer.json", "tokenizer_config.json"],
  },
  {
    name: "e5-small (opcional / liviano)",
    dir: "multilingual-e5-small-onnx",
    base: `${HF}/intfloat/multilingual-e5-small/resolve/main/onnx`,
    files: ["model.onnx", "tokenizer.json", "tokenizer_config.json"],
  },
  {
    // Bi-encoder de intención alternativo a e5 (ONNX_MODEL_SIZE=gte): 768 dims,
    // pooling CLS y sin prefijos query:/passage:. ONNX int8 auto-contenido, con
    // el tokenizer en la RAÍZ del repo y los pesos en /onnx/ (model_int8.onnx).
    name: "gte-multilingual-base (ONNX_MODEL_SIZE=gte)",
    dir: "gte-multilingual-base-onnx",
    files: [
      { src: `${HF}/onnx-community/gte-multilingual-base/resolve/main/onnx/model_int8.onnx`, dest: "model_int8.onnx" },
      { src: `${HF}/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json`, dest: "tokenizer.json" },
      { src: `${HF}/onnx-community/gte-multilingual-base/resolve/main/tokenizer_config.json`, dest: "tokenizer_config.json" },
    ],
  },
  {
    name: "bge-m3-int8 (solo tests)",
    dir: "multilingual-bge-m3-int8-onnx",
    base: `${HF}/gpahal/bge-m3-onnx-int8/resolve/main`,
    files: ["model_quantized.onnx", "tokenizer.json", "tokenizer_config.json"],
  },
  {
    // Reranker del juicio post. Xenova expone los ONNX bajo /onnx/ (la variante
    // int8 es `model_quantized.onnx`) y el tokenizer en la RAÍZ del repo, por eso
    // cada archivo lleva su propia URL. Descargarlo mal dejaba el reranker inactivo
    // con "ENOENT .../ms-marco-minilm-l6-int8-onnx/tokenizer.json".
    name: "ms-marco-minilm-l6-int8 (Cross-Encoder / reranker)",
    dir: "ms-marco-minilm-l6-int8-onnx",
    files: [
      { src: `${HF}/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx`, dest: "model.onnx" },
      { src: `${HF}/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json`, dest: "tokenizer.json" },
      { src: `${HF}/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer_config.json`, dest: "tokenizer_config.json" },
    ],
  },
  {
    // Candidato a sustituir a e5-large como bi-encoder de intención (solo
    // tests, ver tests/embed-qwen3.test.ts). ONNX int8 auto-contenido (614 MB);
    // tokenizer en la RAÍZ del repo y pesos en /onnx/ (int8 = model_quantized.onnx).
    // Requiere last-token pooling + formato de instrucción (no mean-pooling).
    name: "qwen3-embedding-0.6b (candidato, solo tests)",
    dir: "qwen3-embedding-0.6b-onnx",
    files: [
      { src: `${HF}/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/onnx/model_quantized.onnx`, dest: "model.onnx" },
      { src: `${HF}/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/tokenizer.json`, dest: "tokenizer.json" },
      { src: `${HF}/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/tokenizer_config.json`, dest: "tokenizer_config.json" },
    ],
  },
  {
    // NLI del juicio (opcional: solo se activa con JUICIO_NLI_MODEL_PATH).
    // Mismo patrón que ms-marco: ONNX en /onnx/ (int8 = `model_int8.onnx`),
    // tokenizer en la raíz del repo.
    name: "nli-deberta-v3-small-int8 (NLI, opcional)",
    dir: "nli-deberta-v3-small-int8-onnx",
    files: [
      { src: `${HF}/Xenova/nli-deberta-v3-small/resolve/main/onnx/model_int8.onnx`, dest: "model.onnx" },
      { src: `${HF}/Xenova/nli-deberta-v3-small/resolve/main/tokenizer.json`, dest: "tokenizer.json" },
      { src: `${HF}/Xenova/nli-deberta-v3-small/resolve/main/tokenizer_config.json`, dest: "tokenizer_config.json" },
    ],
  },
];

const log = (...args) => console.log("[onnx]", ...args);

async function existsNonEmpty(path) {
  try {
    const s = await stat(path);
    return s.isFile() && s.size > 0;
  } catch {
    return false;
  }
}

function fmtBytes(bytes) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

async function download(url, dest) {
  const tmp = `${dest}.tmp`;
  log(`  descargando ${basename(dest)} desde ${url}`);
  const res = await fetch(url, {
    redirect: "follow",
    signal: AbortSignal.timeout(60 * 60 * 1000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} para ${url}`);
  if (!res.body) throw new Error(`respuesta sin body para ${url}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  const s = await stat(tmp);
  if (s.size === 0) {
    await rm(tmp, { force: true });
    throw new Error(`archivo vacío descargado: ${basename(dest)}`);
  }
  await rename(tmp, dest);
  log(`    ok ${basename(dest)} (${fmtBytes(s.size)})`);
}

/** Resuelve la URL de origen y el nombre local para cada archivo declarado. */
function resolveFile(model, entry) {
  if (typeof entry === "string") {
    return { url: `${model.base}/${entry}`, dest: entry };
  }
  return { url: entry.src, dest: entry.dest ?? basename(new URL(entry.src).pathname) };
}

async function ensureModel(model, baseDir) {
  const target = join(baseDir, model.dir);
  await mkdir(target, { recursive: true });
  let missing = 0;
  for (const entry of model.files) {
    const { url, dest } = resolveFile(model, entry);
    const out = join(target, dest);
    if (await existsNonEmpty(out)) continue;
    missing += 1;
    try {
      await download(url, out);
    } catch (err) {
      await rm(`${out}.tmp`, { force: true });
      throw new Error(`fallo al descargar ${dest}: ${err?.message ?? err}`);
    }
  }
  return missing;
}

async function main() {
  const enabled = String(process.env.ONNX_ENABLED ?? "true").trim().toLowerCase();
  if (["false", "0", "no"].includes(enabled)) {
    log("ONNX_ENABLED=false — se omite la descarga de modelos");
    return;
  }

  const baseDir = resolve(process.env.ONNX_MODELS_PATH || "./models");

  for (const model of MODELS) {
    try {
      const missing = await ensureModel(model, baseDir);
      log(`${model.name} ${missing === 0 ? "ya completo" : "listo"} en ${join(baseDir, model.dir)}`);
    } catch (err) {
      log(`WARN: ${model.name} no se pudo asegurar: ${err?.message ?? err}`);
    }
  }
}

main().catch((err) => {
  log(`WARN: ${err?.message ?? err}`);
  log("El servicio usará el embedding determinístico (fallback hash).");
});
