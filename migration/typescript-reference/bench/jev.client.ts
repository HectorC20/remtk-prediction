/**
 * Cliente de Experiential Cloud — System One (Jev).
 *
 * Reglas del proveedor respetadas aquí:
 * - UN solo intento por request: un timeout puede dejar un cobro con resultado
 *   desconocido, así que nunca se reintenta automáticamente.
 * - La elección devuelta es una RECOMENDACIÓN: el runner la compara, nunca
 *   ejecuta herramientas a partir de ella.
 *
 * Config (solo .env / entorno, nunca en código):
 *   EXPERIENTIAL_API_KEY   xpl_...   (obligatoria para el lado Jev)
 *   EXPERIENTIAL_BASE_URL  default https://api.experientiallabs.ai
 *   EXPERIENTIAL_MODEL     default jev-latest
 */
import { fileURLToPath } from "node:url";
import path from "node:path";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";
import type { BenchTurn } from "./cases";

try {
  process.loadEnvFile?.(path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", ".env"));
} catch {
  // Sin .env: se confía en las variables del entorno.
}

export const NONE_KEY = "__none__";

export interface JevRequest {
  history: BenchTurn[];
  text: string;
  tools: ToolDefinition[];
}

export interface JevOutcome {
  ok: boolean;
  choice?: string;
  confidence?: number;
  latencyMs: number;
  error?: string;
}

function buildBody(req: JevRequest): string {
  const criteria: Record<string, string> = {};
  for (const t of req.tools) {
    criteria[t.name] = `${t.intentSummary}: ${t.description}`;
  }
  criteria[NONE_KEY] =
    "Ninguna herramienta aplica: saludo, meta-pregunta sobre capacidades, rechazo de una propuesta, o la intención no cubre ninguna herramienta.";

  return JSON.stringify({
    model: process.env.EXPERIENTIAL_MODEL ?? "jev-latest",
    state: {
      history: req.history.map((t) => `${t.role}: ${t.content}`),
      text: req.text,
    },
    questions: {
      tool: {
        type: "choice",
        instructions:
          "Elige la ÚNICA herramienta que cumple la intención REAL del turno actual del usuario, usando el historial como contexto. " +
          "Respeta negaciones ('no envíes' ≠ enviar) y correferencias ('¿y mañana?' hereda el tema). " +
          "Un rechazo o un saludo no disparan herramientas. Si ninguna aplica, elige " + NONE_KEY + ".",
        criteria,
      },
    },
  });
}

/** Llama a /v1/systemone con UNA pregunta choice. Sin reintentos. */
export async function jevChoice(req: JevRequest): Promise<JevOutcome> {
  const key = process.env.EXPERIENTIAL_API_KEY;
  if (!key) {
    return { ok: false, latencyMs: 0, error: "EXPERIENTIAL_API_KEY no definida (.env o entorno)" };
  }
  const base = (process.env.EXPERIENTIAL_BASE_URL ?? "https://api.experientiallabs.ai").replace(/\/$/, "");
  const start = Date.now();
  try {
    const res = await fetch(`${base}/v1/systemone`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: buildBody(req),
      signal: AbortSignal.timeout(60_000),
    });
    const latencyMs = Date.now() - start;
    const raw = (await res.json()) as {
      answers?: { tool?: { choice?: string; confidence?: number } };
      error?: { message?: string };
    };
    if (!res.ok) {
      return { ok: false, latencyMs, error: `HTTP ${res.status}: ${raw.error?.message ?? JSON.stringify(raw)}` };
    }
    const answer = raw.answers?.tool;
    if (!answer?.choice) {
      return { ok: false, latencyMs, error: `respuesta sin answers.tool: ${JSON.stringify(raw)}` };
    }
    return { ok: true, choice: answer.choice, confidence: answer.confidence, latencyMs };
  } catch (err) {
    return {
      ok: false,
      latencyMs: Date.now() - start,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
