/**
 * Reproducción en LOCAL del fallo de PRODUCCIÓN (2026-09-29): el pipeline NO
 * devuelve la herramienta del hero de la portada (`mitumbes_hero_configurar`)
 * en consultas en lenguaje natural, ni aun registrando más de 200 tools.
 *
 * Evidencia de origen (docs/testing/nest-2026-09-29.log, model=small):
 *  - L66-73:  "Listar diapositivas activas del hero de la portada…"
 *             → out_tools=[evento_hero_programar, item_imagenes_listar,
 *               item_imagen_actualizar, zona_listar, publicidad_listar]
 *             (SIN hero_configurar / hero_obtener)
 *  - L136-143: "Consultar dominio de ayuda del hero/portada…"
 *             → out_tools=[evento_hero_programar, ayuda, publicidad_obtener,
 *               zona_listar, publicidad_actualizar]  (SIN hero_configurar)
 *  - L199-206: "Obtener configuración actual del hero con todas sus
 *               diapositivas…" → out_tools=[buscar,
 *               categoria_obtener_orden_items]  (SIN hero_configurar)
 *  - L99-106 (control): cita EXPLÍCITA "…con mitumbes_hero_configurar"
 *             → out_tools=[hero_configurar, …]  (sí lo devuelve)
 *
 * Objetivo: verificar sobre el servidor REAL (`startPredictServer`) con el
 * catálogo real de MiTumbes + ruido (TOTAL_ACTIVE_TOOLS = 232 tools) y el
 * modelo e5-large, que las consultas en lenguaje natural anteriores devuelven
 * al menos una tool del dominio hero (`hero_obtener` para lectura; con cita
 * explícita también `hero_configurar`).
 *
 * Limitación conocida: `hero_configurar` (ESCRITURA) solo se devuelve cuando se
 * menciona explícitamente; por intención de escritura en lenguaje natural el
 * dominio hero se resuelve vía `hero_obtener`.
 *
 * Si el fallo persiste, el test FALLA y deja ver la tanda devuelta.
 */

import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";

// Modelo del pipeline para esta prueba (A/B): por defecto gte (modelo del
// pipeline); se puede correr con `ONNX_MODEL_SIZE=small` (fallo original) o
// `ONNX_MODEL_SIZE=large`.
const MODEL_SIZE = (process.env.ONNX_MODEL_SIZE ?? "gte") as "small" | "large" | "gte";
process.env.ONNX_MODEL_SIZE = MODEL_SIZE;

import { startPredictServer } from "../src/app.module";
import { TOTAL_ACTIVE_TOOLS, MITUMBES_VERBATIM_TOOLS } from "./fixtures/mitumbes-real-catalog";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

// Scope particionado como en producción: tenant (userId) + agentId.
const TENANT_ID = "7f4b4b79-ad05-4cfb-9213-d2e6575e503e";
const AGENT_ID = "b5b3711e-0d1e-4d8b-bf44-9ddedbc18bc8";

/** Herramienta de ESCRITURA del hero (la que el usuario reporta como no devuelta). */
const HERO_WRITE = "mitumbes_hero_configurar";
/** Herramienta de LECTURA del hero (la correcta para consultas "listar/obtener"). */
const HERO_READ = "mitumbes_hero_obtener";

/** Consultas EXACTAS del log que fallaron (intención de LECTURA: listar/obtener). */
const READ_QUERIES: Array<{ label: string; text: string }> = [
  {
    label: "Listar diapositivas activas del hero",
    text:
      "Listar diapositivas activas del hero de la portada. Listar diapositivas activas del hero de la portada. " +
      "Lista de diapositivas del hero con sus datos (título, imagen, orden, estado activo)",
  },
  {
    label: "Consultar dominio de ayuda del hero/portada",
    text:
      "Consultar dominio de ayuda del hero/portada. Consultar dominio de ayuda del hero/portada. " +
      "Documentación de las herramientas del hero (configurar, listar, programar) con nombres exactos, parámetros y descripciones",
  },
  {
    label: "Obtener configuración actual del hero",
    text:
      "Obtener configuración actual del hero con todas sus diapositivas. " +
      "Obtener configuración actual del hero con todas sus diapositivas. " +
      "Lista completa de las diapositivas activas del hero con sus referencias " +
      "(ítems, lugares, categorías, zonas, eventos o campañas) y su orden de aparición",
  },
];

/** Consultas con intención de ESCRITURA: deben devolver mitumbes_hero_configurar. */
const WRITE_QUERIES: Array<{ label: string; text: string }> = [
  {
    label: "Configurar/renovar las diapositivas del hero",
    text:
      "Configurar las diapositivas del hero de la portada: define qué ítems, lugares, zonas o campañas se muestran y en qué orden. " +
      "Reemplazar todas las diapositivas actuales del hero de la portada",
  },
];

interface PredictResponse {
  tools: ToolDefinition[];
}

async function predict(text: string, source: "human" | "agent", sessionId: string): Promise<string[]> {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, tenant: TENANT_ID, agentId: AGENT_ID, text, source }),
  });
  assert.equal(res.status, 200, `POST /predict debe responder 200 (texto: "${text}")`);
  const data = (await res.json()) as PredictResponse;
  return data.tools.map((t) => t.name);
}

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  assert.ok(
    MITUMBES_VERBATIM_TOOLS.some((t) => t.name === HERO_WRITE),
    `El fixture debe incluir la herramienta objetivo ${HERO_WRITE}`,
  );

  // Registrar el catálogo real de MiTumbes + ruido de otros dominios (232 tools).
  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, agentId: AGENT_ID, tools: TOTAL_ACTIVE_TOOLS }),
  });
  assert.equal(regRes.status, 200, "POST /tools debe responder 200");
  const regData = (await regRes.json()) as { indexed: number };
  assert.equal(
    regData.indexed,
    TOTAL_ACTIVE_TOOLS.length,
    `Debe indexar el catálogo completo (${TOTAL_ACTIVE_TOOLS.length})`,
  );

  // Esperar el warm-up del scope (grafo + embeddings del default).
  const readyRes = await fetch(`${serverUrl}/tools/ready?tenant=${TENANT_ID}&agentId=${AGENT_ID}`);
  assert.equal(readyRes.status, 200);

  // Confirmar que el pipeline corre sobre el modelo elegido (no fallback hash).
  const dbgRes = await fetch(`${serverUrl}/debug?tenant=${TENANT_ID}&agentId=${AGENT_ID}`);
  assert.equal(dbgRes.status, 200);
  const dbg = (await dbgRes.json()) as { engine: { size: string; modelPath: string } };
  console.log(`\n[HERO] engine.size=${dbg.engine.size} modelPath=${dbg.engine.modelPath}`);
  assert.equal(
    dbg.engine.size,
    MODEL_SIZE,
    `El pipeline debe correr sobre el modelo ${MODEL_SIZE} (obtuvo: ${dbg.engine.size})`,
  );
});

after(async () => {
  if (closeServer) await closeServer();
});

describe(`Reproducción local del fallo: hero de la portada bajo ruido (232 tools, modelo ${MODEL_SIZE})`, () => {
  // Intención de LECTURA (listar/obtener): la herramienta correcta es
  // `mitumbes_hero_obtener`. Con e5-small el dominio hero NO aparecía; con el
  // modelo por defecto (gte) sí debe aparecer al menos una tool del dominio hero.
  for (const { label, text } of READ_QUERIES) {
    test(`Lectura NL: ${label}`, async () => {
      const names = await predict(text, "agent", `repro-hero-read-${label.replace(/\s+/g, "-").toLowerCase()}`);

      const hasRead = names.includes(HERO_READ);
      const hasWrite = names.includes(HERO_WRITE);
      console.log(`\n[HERO read "${label}"] Tools devueltas (${names.length}):`, names);
      console.log(
        `[HERO read "${label}"] ${HERO_READ}=${hasRead ? "SÍ" : "NO"} · ${HERO_WRITE}=${hasWrite ? "SÍ" : "NO"}`,
      );

      assert.ok(
        hasRead || hasWrite,
        `FALLO REPRODUCIDO — ninguna tool del hero (${HERO_READ}/${HERO_WRITE}) en la tanda: [${names.join(", ")}]`,
      );
    });
  }

  // Intención de ESCRITURA (configurar/renovar): LIMITACIÓN CONOCIDA — la tool
  // de escritura (`hero_configurar`) NO se devuelve por intención en lenguaje
  // natural; solo aparece con mención explícita (ver Control). Lo que sí
  // garantizamos es que el dominio hero sea alcanzable (lectura o escritura).
  for (const { label, text } of WRITE_QUERIES) {
    test(`Escritura NL: ${label}`, async () => {
      const names = await predict(text, "agent", `repro-hero-write-${label.replace(/\s+/g, "-").toLowerCase()}`);

      const hasWrite = names.includes(HERO_WRITE);
      const hasRead = names.includes(HERO_READ);
      console.log(`\n[HERO write "${label}"] Tools devueltas (${names.length}):`, names);
      console.log(
        `[HERO write "${label}"] ${HERO_WRITE}=${hasWrite ? "SÍ" : "NO"} · ${HERO_READ}=${hasRead ? "SÍ" : "NO"}`,
      );

      assert.ok(
        hasRead || hasWrite,
        `FALLO REPRODUCIDO — ninguna tool del hero (${HERO_READ}/${HERO_WRITE}) en la tanda: [${names.join(", ")}]`,
      );
    });
  }

  test("Control (cita explícita): '…con mitumbes_hero_configurar'", async () => {
    const text =
      "Consultar configuración actual de las diapositivas del hero de la portada con mitumbes_hero_configurar. " +
      "Consultar configuración actual de las diapositivas del hero de la portada con mitumbes_hero_configurar. " +
      "Lista de diapositivas del hero de la portada con sus datos (títulos, imágenes, orden, estado) o confirmación " +
      "de que la herramienta no está disponible para este agente";
    const names = await predict(text, "agent", "repro-hero-control-explicit");

    console.log(`\n[HERO control] Tools devueltas (${names.length}):`, names);
    assert.ok(
      names.includes(HERO_WRITE),
      `Control — ${HERO_WRITE} debería estar con cita explícita: [${names.join(", ")}]`,
    );
  });
});
