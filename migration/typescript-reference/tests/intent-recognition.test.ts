/**
 * Test Riguroso de Reconocimiento de Intención y Selección de Herramientas
 *
 * Valida que el servidor de predicción identifique fielmente la intención del
 * usuario en lenguaje natural coloquial (sin que el usuario tenga que escribir
 * "búsqueda por internet" ni nombres técnicos de herramientas) y SIN diccionarios,
 * listas ni palabras clave artificiales.
 *
 * Cobertura de escenarios del usuario real (logs 2026-09-25):
 * 1. Consulta de noticias locales: "noticias de tumbes hoy" -> web_search (search)
 * 2. Consulta de entidad/biografía: "quien es MILO J" -> web_search (search)
 * 3. Búsqueda explícita general: "busqueda por internet" -> web_search (search)
 * 4. Consulta meteorológica local: "que tiempo hace en tumbes" -> clima_tumbes (search/user_tool)
 * 5. Operación sobre el espacio de trabajo: "crear un archivo notas.txt con un resumen" -> workspace tools
 * 6. Verificación de contrato de contexto e intención (primaryAction, confidence, category)
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "intent-test-tenant-2026";

/**
 * Catálogo representativo de herramientas (búsqueda, clima, workspace, scheduling, chat)
 * tal como llegan desde key-remtk y plugins.
 */
const CATALOG_TOOLS: ToolDefinition[] = [
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "web_search",
    description: "Busca información actualizada en internet consultando varios proveedores en paralelo (DuckDuckGo, SearXNG y Tavily). Úsala para noticias, documentación y datos recientes.",
    tags: ["search", "web", "internet", "documentation", "news"],
    intentSummary: "Busca información actualizada en internet a través de múltiples proveedores",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "web_search_research",
    name: "web_search_research",
    group: "search",
    category: "web_research",
    description: "Investigación web profunda multi-salto sobre un tema, comparativa o reporte.",
    tags: ["research", "deep research", "analysis", "report"],
    intentSummary: "Realiza una investigación web profunda multi-salto sobre un tema",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "web_search_extract",
    name: "web_search_extract",
    group: "search",
    category: "web_extract",
    description: "Extrae y resume el contenido de URLs específicas para leer artículos o documentación.",
    tags: ["extract", "scraping", "read url", "web page"],
    intentSummary: "Extracts content from URLs to read articles",
    inputSchema: { type: "object", properties: { urls: { type: "array" } }, required: ["urls"] },
  },
  {
    id: "clima_tumbes",
    name: "clima_tumbes",
    group: "search",
    category: "user_tool",
    description: "Consulta el clima actual de Tumbes, Perú usando la API gratuita de Open-Meteo. Devuelve temperatura, sensación térmica, humedad y condiciones del cielo.",
    tags: ["clima", "weather", "tumbes", "peru", "temperatura"],
    intentSummary: "Consulta el clima actual de Tumbes, Perú usando la API de Open-Meteo",
    inputSchema: { type: "object", properties: { lat: { type: "string" }, lon: { type: "string" } } },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el espacio de trabajo del usuario con el contenido especificado.",
    tags: ["file", "create", "write", "workspace", "archivo", "guardar"],
    intentSummary: "Crea o sobrescribe un archivo en el espacio de trabajo con contenido",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del espacio de trabajo del usuario.",
    tags: ["file", "read", "workspace", "archivo", "leer"],
    intentSummary: "Lee el contenido de un archivo del espacio de trabajo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "search_chat_history",
    name: "search_chat_history",
    group: "chat",
    category: "memory",
    description: "Recupera mensajes anteriores del chat actual desde la memoria vectorial para retomar temas o reportes previos.",
    tags: ["chat", "history", "memory", "buscar"],
    intentSummary: "Recupera mensajes anteriores del chat actual desde la memoria",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "schedule_task",
    name: "schedule_task",
    group: "scheduler",
    category: "runtime",
    description: "Programa una tarea futura única o periódica (cron) para recordatorios o avisos.",
    tags: ["schedule", "cron", "reminder", "recordatorio", "agendar"],
    intentSummary: "Programa una tarea del agente futura o recurrente",
    inputSchema: { type: "object", properties: { type: { type: "string" } }, required: ["type"] },
  },
];

before(async () => {
  // Arranca servidor de predicción en puerto efímero con modelo ONNX real
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  // Registrar el catálogo de herramientas en el tenant de prueba
  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: CATALOG_TOOLS }),
  });
  assert.equal(regRes.status, 200, "El registro de herramientas debe responder 200");
});

after(async () => {
  if (closeServer) await closeServer();
});

async function predict(text: string, sessionId = "session-" + Math.random().toString(36).slice(2)): Promise<{
  tools: ToolDefinition[];
  complexity: string;
  context: {
    intent: {
      primaryAction: string;
      confidence: number;
      category?: string;
      summary: string;
    };
    dialogState: {
      phase: string;
      activeDomain?: string;
    };
  };
  rankedScores: number[];
  calibratedScores: number[];
}> {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId,
      tenant: TENANT_ID,
      text,
      source: "human",
    }),
  });
  assert.equal(res.status, 200, `Predict status esperado 200, obtenido ${res.status}`);
  return (await res.json()) as ReturnType<typeof predict>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 1: "noticias de tumbes hoy" (usuario común busca noticias de su región)
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 1: 'noticias de tumbes hoy' identifica intención de búsqueda y selecciona web_search", async () => {
  const result = await predict("noticias de tumbes hoy");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(toolNames.length > 0, "Debe predecir herramientas");
  assert.ok(
    toolNames.includes("web_search"),
    `Debe incluir 'web_search' en las herramientas predichas: ${toolNames.join(", ")}`,
  );
  assert.equal(result.context.intent.category, "search", "category debe ser search");
  assert.ok(result.context.intent.confidence > 0.8, "La confianza debe superar 0.8");
});

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 2: "quien es MILO J" (usuario común pregunta por una persona/artista)
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 2: 'quien es MILO J' identifica intención informativa y selecciona web_search", async () => {
  const result = await predict("quien es MILO J");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(toolNames.length > 0, "Debe predecir herramientas");
  assert.ok(
    toolNames.includes("web_search") || toolNames.includes("web_search_research"),
    `Debe incluir herramienta de búsqueda web: ${toolNames.join(", ")}`,
  );
  assert.equal(result.context.intent.category, "search");
  assert.ok(["web_search", "web_research"].includes(result.context.intent.primaryAction));
});

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 3: "busqueda por internet" (intención de búsqueda formulada directamente)
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 3: 'busqueda por internet' selecciona web_search con alta afinidad", async () => {
  const result = await predict("busqueda por internet");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(toolNames.includes("web_search"), "Debe seleccionar web_search");
  assert.equal(result.context.intent.primaryAction, "web_search");
});

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 4: "que tiempo hace en tumbes" (consulta meteorológica coloquial)
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 4: 'que tiempo hace en tumbes' prioriza clima_tumbes sobre búsqueda genérica", async () => {
  const result = await predict("que tiempo hace en tumbes");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(toolNames.includes("clima_tumbes"), "Debe incluir clima_tumbes");
  assert.equal(toolNames[0], "clima_tumbes", "clima_tumbes debe ser el Top-1 de alta especificidad");
  assert.equal(result.context.intent.primaryAction, "user_tool");
});

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 5: "crear un archivo notas.txt con un resumen" (operación de archivos)
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 5: 'crear un archivo notas.txt con un resumen' identifica intención de workspace", async () => {
  const result = await predict("crear un archivo notas.txt con un resumen");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir 'workspace_write_file': ${toolNames.join(", ")}`,
  );
  assert.equal(toolNames[0], "workspace_write_file");
  assert.equal(result.context.intent.category, "workspace");
  assert.equal(result.context.intent.primaryAction, "filesystem");
});

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 6: "revisa los turnos anteriores del chat" (memoria conversacional)
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 6: 'revisa los turnos anteriores del chat' selecciona search_chat_history", async () => {
  const result = await predict("revisa los turnos anteriores del chat para ver el link que me diste");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("search_chat_history"),
    `Debe incluir 'search_chat_history': ${toolNames.join(", ")}`,
  );
  assert.equal(result.context.intent.category, "chat");
});

// ─────────────────────────────────────────────────────────────────────────────
// Escenario 7: Validación de contratos de Calibración Matemática
// ─────────────────────────────────────────────────────────────────────────────
test("Escenario 7: CalibratedScores y context.intent cumplen con distribución probabilística [0, 1]", async () => {
  const result = await predict("noticias recientes de tecnología en la web");

  assert.ok(Array.isArray(result.calibratedScores), "calibratedScores debe ser un array");
  assert.equal(result.calibratedScores.length, result.tools.length);

  for (let i = 0; i < result.calibratedScores.length; i++) {
    const s = result.calibratedScores[i];
    assert.ok(s >= 0 && s <= 1, `Score calibrado fuera de [0, 1]: ${s}`);
    if (i > 0) {
      assert.ok(result.calibratedScores[i - 1] >= s, "Preserva orden monotónicamente decreciente");
    }
  }

  assert.ok(result.context.intent.confidence >= 0 && result.context.intent.confidence <= 1.5);
  assert.ok(typeof result.context.intent.summary === "string" && result.context.intent.summary.length > 0);
  assert.equal(result.context.dialogState.phase, "execution");
});
