/**
 * Reproducción en LOCAL del fallo de PRODUCCIÓN (2026-09-29).
 *
 * Evidencia de origen (log real):
 *   13: prompt="tienes la herramienta mitumbes_item_imagen_adjuntar"
 *  154: [TaskExecutorV5] tarea "task-0": predicho | modo=consultar | enriquecido=0chars
 *       | omito=[] devueltas=[mitumbes_item_imagenes_listar,
 *                             mitumbes_item_imagen_actualizar,
 *                             mitumbes_item_imagen_eliminar]
 *  → `mitumbes_item_imagen_adjuntar` NO aparece en la tanda devuelta.
 *
 * Objetivo: verificar si ese mismo error se reproduce en local levantando el
 * servidor REAL (`startPredictServer`) con el catálogo real de MiTumbes más
 * ruido de otros dominios, y preguntando EXACTAMENTE lo mismo que en producción.
 *
 * El catálogo se registra particionado por scope `tenant::agentId`, igual que en
 * producción (scope `...::b5b3711e-...`), y proviene de
 * `tests/fixtures/mitumbes-real-catalog.ts`: las 98 tools REALES de MiTumbes
 * (76 primarias + 22 variantes `_varios`), replicadas fielmente como las publica
 * key-remtk al predictor (`tags: ["MiTumbes"]`, `category: "plugin_tool"` y la
 * descripción real completa), más ruido de otros dominios.
 *
 * Si el fallo se reproduce, el test FALLA y deja ver la tanda devuelta:
 *   AssertionError: mitumbes_item_imagen_adjuntar debe estar en la tanda devuelta.
 */

import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import { TOTAL_ACTIVE_TOOLS, MITUMBES_VERBATIM_TOOLS } from "./fixtures/mitumbes-real-catalog";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

// Scope particionado como en producción: tenant (userId) + agentId.
const TENANT_ID = "7f4b4b79-ad05-4cfb-9213-d2e6575e503e";
const AGENT_ID = "b5b3711e-0d1e-4d8b-bf44-9ddedbc18bc8";

const TARGET = "mitumbes_item_imagen_adjuntar";

// Textos EXACTOS del log de producción.
const PROMPT_USUARIO = "tienes la herramienta mitumbes_item_imagen_adjuntar";
const SUBTAREA_AGENTE =
  "Verificar existencia de la herramienta mitumbes_item_imagen_adjuntar";

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
    MITUMBES_VERBATIM_TOOLS.some((t) => t.name === TARGET),
    `El fixture debe incluir la herramienta objetivo ${TARGET}`,
  );

  // Registrar el catálogo real de MiTumbes + ruido de otros dominios.
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

  // Esperar el warm-up del scope para que el grafo y los embeddings estén listos.
  const readyRes = await fetch(`${serverUrl}/tools/ready?tenant=${TENANT_ID}&agentId=${AGENT_ID}`);
  assert.equal(readyRes.status, 200);
});

after(async () => {
  if (closeServer) await closeServer();
});

describe("Reproducción local del fallo de producción: mitumbes_item_imagen_adjuntar bajo ruido", () => {
  test("Prompt del usuario: 'tienes la herramienta mitumbes_item_imagen_adjuntar'", async () => {
    const names = await predict(PROMPT_USUARIO, "human", "repro-adjuntar-human-1");

    console.log(`\n[REPRO human] Tools devueltas (${names.length}):`, names);
    const reproducido = !names.includes(TARGET);
    console.log(
      `[REPRO human] ¿Se reproduce el fallo de producción (${TARGET} ausente)? ${reproducido ? "SÍ" : "NO"}`,
    );

    assert.ok(
      names.includes(TARGET),
      `FALLO REPRODUCIDO — ${TARGET} NO está en la tanda devuelta: [${names.join(", ")}]`,
    );
  });

  test("Subtarea del agente: 'Verificar existencia de la herramienta mitumbes_item_imagen_adjuntar'", async () => {
    // En el log esta consulta (task-0, modo=consultar) devolvió
    // [imagenes_listar, imagen_actualizar, imagen_eliminar] — sin `imagen_adjuntar`.
    const names = await predict(SUBTAREA_AGENTE, "agent", "repro-adjuntar-agent-1");

    console.log(`\n[REPRO agent] Tools devueltas (${names.length}):`, names);
    const reproducido = !names.includes(TARGET);
    console.log(
      `[REPRO agent] ¿Se reproduce el fallo de producción (${TARGET} ausente)? ${reproducido ? "SÍ" : "NO"}`,
    );

    assert.ok(
      names.includes(TARGET),
      `FALLO REPRODUCIDO — ${TARGET} NO está en la tanda devuelta: [${names.join(", ")}]`,
    );
  });
});
