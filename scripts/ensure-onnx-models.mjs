#!/usr/bin/env node
/**
 * ensure-onnx-models.mjs
 *
 * Descarga los modelos ONNX de embeddings que usa remtk-prediction desde
 * Hugging Face al directorio de modelos, solo si faltan (idempotente).
 *
 * El servidor usa e5-small (default) y e5-large (bajo demanda); bge-m3-int8
 * solo se usa en tests (embed-bge.test.ts) y se descarga como extra.
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

/** Modelos a asegurar. `files` son los archivos mínimos para cargar el modelo. */
const MODELS = [
  {
    name: "e5-small (default)",
    dir: "multilingual-e5-small-onnx",
    base: "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx",
    files: ["model.onnx", "tokenizer.json", "tokenizer_config.json"],
  },
  {
    name: "e5-large (bajo demanda)",
    dir: "multilingual-e5-large-onnx",
    base: "https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/onnx",
    files: ["model.onnx", "model.onnx_data", "tokenizer.json", "tokenizer_config.json"],
  },
  {
    name: "bge-m3-int8 (solo tests)",
    dir: "multilingual-bge-m3-int8-onnx",
    base: "https://huggingface.co/gpahal/bge-m3-onnx-int8/resolve/main",
    files: ["model_quantized.onnx", "tokenizer.json", "tokenizer_config.json"],
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

async function ensureModel(model, baseDir) {
  const target = join(baseDir, model.dir);
  await mkdir(target, { recursive: true });
  let missing = 0;
  for (const file of model.files) {
    const dest = join(target, file);
    if (await existsNonEmpty(dest)) continue;
    missing += 1;
    try {
      await download(`${model.base}/${file}`, dest);
    } catch (err) {
      await rm(`${dest}.tmp`, { force: true });
      throw new Error(`fallo al descargar ${file}: ${err?.message ?? err}`);
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
