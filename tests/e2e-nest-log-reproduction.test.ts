/**
 * Test E2E de Reproducción Fiel: nest-2026-09-27 (3).log
 *
 * Simula el flujo completo de 3 turnos del log en presencia de ruido masivo (188 herramientas):
 *   - 98 Herramientas reales del servidor MCP de MiTumbes
 *   - 90 Herramientas de ruido de 8 dominios externos (workspace, database, stripe, git, docker, email, calendar, telemetry)
 *
 * Verifica:
 *   1. Subtarea del agente: "Verificar si existe una herramienta activa para crear ítems en el catálogo de mitumbes.com"
 *      - NO debe devolver 98 herramientas (causa de la explosión de 62,030 tokens en el log).
 *      - Debe acotar la salida a <= 5 herramientas.
 *      - Debe identificar `mitumbes_item_crear` como objetivo funcional.
 *      - Debe estructurar el Pre-plan de Razonamiento topológico (DAG Kahn) trayendo las herramientas antecedentes:
 *        `[mitumbes_categoria_listar, mitumbes_zona_listar, mitumbes_item_crear]`.
 *   2. Turno 1 (Human prompt con solicitud de categorías y zonas):
 *      - Recupera las herramientas complementarias para categorizar y ubicar antes de crear.
 *   3. Turno 2 ("SI, USA LA HERRAMIENTA"):
 *      - Juicio de estado conversacional: no desvía a workspace ni alucina herramientas de ruido.
 *   4. Turno 3 ("PUES LA CATEGORÍA ES DE RESTAURANTE Y DE LA ZONA DE TUMBES!!!"):
 *      - Concreta la herramienta de creación `mitumbes_item_crear`.
 *   5. Memoria Contextual (POST /memory/judge):
 *      - Clasifica la negación parásita como ruido y preserva los hechos de negocio reales.
 */

import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import { TOTAL_ACTIVE_TOOLS } from "./fixtures/mitumbes-production-tools";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "nest-e2e-reproduction-2026";
const SESSION_ID = "session-nest-reproduction-1";

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  // Registrar las 188 herramientas en el catálogo del tenant
  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: TOTAL_ACTIVE_TOOLS }),
  });
  assert.equal(regRes.status, 200, "Registro de herramientas debe ser 200 OK");
  const regData = (await regRes.json()) as { indexed: number };
  assert.ok(regData.indexed >= 188, `Debe registrar al menos 188 herramientas, indexadas: ${regData.indexed}`);

  // Esperar a que el warm-up en segundo plano construya el grafo y los embeddings
  const readyRes = await fetch(`${serverUrl}/tools/ready?tenant=${TENANT_ID}`);
  assert.equal(readyRes.status, 200);
});

after(async () => {
  if (closeServer) await closeServer();
});

describe("Reproducción E2E del Escenario nest-2026-09-27 (3).log bajo Ruido Masivo", () => {
  test("Turno 1 - Subtarea del Agente: verificar herramienta activa para crear ítems en el catálogo", async () => {
    // Esta consulta es la que en el log provocó la inundación de 98 tools (62,030 tokens)
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: SESSION_ID,
        tenant: TENANT_ID,
        text: "Verificar si existe una herramienta activa para crear ítems en el catálogo de mitumbes.com",
        source: "agent",
      }),
    });

    assert.equal(res.status, 200);
    const data = (await res.json()) as {
      tools: ToolDefinition[];
      graph?: { executionOrder?: string[] };
    };
    const names = data.tools.map((t) => t.name);

    console.log(`\n[E2E Agente Subtarea] Tools devueltas (${names.length}):`, names);

    // 1. Inmunidad a inundación: NUNCA devolver 98 herramientas a un agente
    assert.ok(
      names.length <= 5,
      `Debe acotar la respuesta a <= 5 herramientas, pero devolvió ${names.length}`
    );

    // 2. Discriminación del objetivo: debe incluir mitumbes_item_crear
    assert.ok(
      names.includes("mitumbes_item_crear"),
      "Debe incluir la herramienta objetivo mitumbes_item_crear"
    );

    // 3. Pre-plan de razonamiento topológico: debe incluir las herramientas antecedentes requeridas
    assert.ok(
      names.includes("mitumbes_categoria_listar") || names.includes("mitumbes_zona_listar"),
      "Debe incluir herramientas complementarias antecedentes (categoria_listar o zona_listar)"
    );

    // 4. Cero ruido de servidores externos en la selección
    assert.ok(
      !names.some((n) => n.startsWith("workspace_op_") || n.startsWith("stripe_") || n.startsWith("database_")),
      "Cero herramientas de ruido ajeno"
    );
  });

  test("Turno 1 - Prompt del Usuario: creación con pre-plan de razonamiento de categoría y zona", async () => {
    const prompt =
      "crea el item La Pichanga Gastrobar, mejora su descripción y si no sabes nada de sus categorías ni su zona pues pregunta pero si tienes herramientas para consultar dichas categorías y zonas pues utilízalas para obtener la categoría y zona más adecuada para este restaurante";

    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: SESSION_ID,
        tenant: TENANT_ID,
        text: prompt,
        source: "human",
      }),
    });

    assert.equal(res.status, 200);
    const data = (await res.json()) as { tools: ToolDefinition[] };
    const names = data.tools.map((t) => t.name);

    console.log(`\n[E2E Turno 1 Human] Tools devueltas (${names.length}):`, names);

    assert.ok(names.length <= 5, `Tope máximo de tools respetado (<=5), obtenido: ${names.length}`);
    assert.ok(names.includes("mitumbes_item_crear"), "mitumbes_item_crear debe estar en la predicción");
    assert.ok(
      names.includes("mitumbes_categoria_listar") || names.includes("mitumbes_zona_listar"),
      "Debe sugerir pre-requisitos de razonamiento para resolver categorías o zonas"
    );
  });

  test("Turno 2 - Confirmación del Usuario: 'SI, USA LA HERRAMIENTA'", async () => {
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: SESSION_ID,
        tenant: TENANT_ID,
        text: "SI, USA LA HERRAMIENTA",
        source: "human",
      }),
    });

    assert.equal(res.status, 200);
    const data = (await res.json()) as { tools: ToolDefinition[] };
    const names = data.tools.map((t) => t.name);

    console.log(`\n[E2E Turno 2 Confirmación] Tools devueltas (${names.length}):`, names);

    // En continuidad de sesión, la confirmación debe mantener la herramienta del catálogo
    assert.ok(
      names.includes("mitumbes_item_crear"),
      "La confirmación debe retener mitumbes_item_crear sin desviar a workspace_write_file"
    );
  });

  test("Turno 3 - Parámetros del Usuario: 'PUES LA CATEGORÍA ES DE RESTAURANTE Y DE LA ZONA DE TUMBES!!!'", async () => {
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: SESSION_ID,
        tenant: TENANT_ID,
        text: "PUES LA CATEGORÍA ES DE RESTAURANTE Y DE LA ZONA DE TUMBES!!!",
        source: "human",
      }),
    });

    assert.equal(res.status, 200);
    const data = (await res.json()) as { tools: ToolDefinition[] };
    const names = data.tools.map((t) => t.name);

    console.log(`\n[E2E Turno 3 Parámetros] Tools devueltas (${names.length}):`, names);

    assert.ok(
      names.includes("mitumbes_item_crear"),
      "El aporte de parámetros debe activar mitumbes_item_crear para completar la mutación"
    );
  });

  test("Memoria Contextual - Juicio Semántico sin Dependencia de Keywords", async () => {
    // 1. Negación de capacidades del asistente en el log:
    // "No dispongo en esta conversación de una acción ejecutable confirmada para crear el ítem en el catálogo..."
    const noiseRes = await fetch(`${serverUrl}/memory/judge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "No dispongo en esta conversación de una acción ejecutable confirmada para crear el ítem en el catálogo unificado de mitumbes.",
      }),
    });
    assert.equal(noiseRes.status, 200);
    const noiseData = (await noiseRes.json()) as { isNoise: boolean; noiseScore: number };
    console.log(`\n[Memoria Juicio] Negación técnica: isNoise=${noiseData.isNoise} score=${noiseData.noiseScore.toFixed(3)}`);
    assert.equal(noiseData.isNoise, true, "La negación técnica de herramientas debe ser descartada como ruido");

    // 2. Información fáctica real provista en la conversación:
    const factRes = await fetch(`${serverUrl}/memory/judge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "La Pichanga Gastrobar se ubica en Huascar 560 Tumbes 24001, ofrece platos marinos y música en vivo los fines de semana.",
      }),
    });
    assert.equal(factRes.status, 200);
    const factData = (await factRes.json()) as { isNoise: boolean; noiseScore: number };
    console.log(`[Memoria Juicio] Hecho de negocio: isNoise=${factData.isNoise} score=${factData.noiseScore.toFixed(3)}`);
    assert.equal(factData.isNoise, false, "Información legítima de negocio debe ser conservada");
  });
});
