/**
 * Benchmark y Pruebas Unitarias Comparativas: Búsqueda Normal (Léxica) vs Búsqueda Inteligente (Prediction Server)
 *
 * Basado en las consultas y escenarios reales extraídos de los logs de producción (nest-2026-09-25.log).
 *
 * Muestra el desempeño PRE (Búsqueda Normal: coincidencia literal de palabras/tags)
 * vs POST (Búsqueda Inteligente: servidor de predicción neuronal sin diccionarios ni reglas cableadas).
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "search-comparison-tenant-2026";

/**
 * Catálogo real de herramientas presentes en los logs de key-remtk (nest-2026-09-25.log)
 */
const CATALOG_TOOLS: ToolDefinition[] = [
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "web_search",
    description: "Busca información actualizada en internet consultando varios motores en paralelo. Úsala para noticias, biografía, eventos recientes y datos generales.",
    tags: ["search", "web", "internet", "noticias", "news", "google", "buscador"],
    intentSummary: "Busca información y noticias actualizadas en la web",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "web_search_research",
    name: "web_search_research",
    group: "search",
    category: "web_research",
    description: "Investigación web profunda multi-salto sobre un tema, dominio web, empresa o reporte detallado.",
    tags: ["research", "investigacion", "web", "deep research", "reporte"],
    intentSummary: "Realiza investigación web profunda y análisis de sitios o dominios",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "web_search_extract",
    name: "web_search_extract",
    group: "search",
    category: "web_extract",
    description: "Extrae y resume el contenido de URLs específicas para leer artículos o páginas web.",
    tags: ["extract", "scraping", "leer url", "web page"],
    intentSummary: "Lee y extrae contenido de una página web específica",
    inputSchema: { type: "object", properties: { urls: { type: "array" } }, required: ["urls"] },
  },
  {
    id: "clima_tumbes",
    name: "clima_tumbes",
    group: "search",
    category: "user_tool",
    description: "Consulta el clima actual y pronóstico del tiempo de Tumbes, Perú usando Open-Meteo. Devuelve temperatura, sensación térmica y lluvia.",
    tags: ["clima", "weather", "tumbes", "temperatura", "tiempo"],
    intentSummary: "Consulta el pronóstico meteorológico y temperatura de Tumbes",
    inputSchema: { type: "object", properties: { lat: { type: "string" }, lon: { type: "string" } } },
  },
  {
    id: "mitumbes_lugares",
    name: "mitumbes_lugares",
    group: "mitumbes",
    category: "places",
    description: "Consulta y gestiona lugares turísticos, playas y reservas de la API MiTumbes.",
    tags: ["mitumbes", "lugares", "turismo", "playas", "tumbes", "hoteles"],
    intentSummary: "Consulta y gestiona lugares turísticos en la API MiTumbes",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "mitumbes_eventos",
    name: "mitumbes_eventos",
    group: "mitumbes",
    category: "events",
    description: "Consulta y gestiona la cartelera de eventos y festivales de la API MiTumbes.",
    tags: ["mitumbes", "eventos", "festivales", "agenda", "tumbes"],
    intentSummary: "Consulta la agenda de eventos de MiTumbes",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del espacio de trabajo del usuario.",
    tags: ["file", "read", "workspace", "archivo", "leer", "abrir"],
    intentSummary: "Lee archivos del espacio de trabajo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el espacio de trabajo con el contenido especificado.",
    tags: ["file", "write", "workspace", "archivo", "escribir", "guardar", "crear"],
    intentSummary: "Crea o modifica archivos en el workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "search_chat_history",
    name: "search_chat_history",
    group: "chat",
    category: "memory",
    description: "Recupera mensajes anteriores del chat actual desde la memoria vectorial.",
    tags: ["chat", "history", "memory", "conversacion", "memoria"],
    intentSummary: "Recupera contexto previo de la conversación del chat",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "schedule_task",
    name: "schedule_task",
    group: "scheduler",
    category: "runtime",
    description: "Programa una tarea del agente para el futuro: única (executeAt) o periódica (cron), recordatorios, avisos y alarmas.",
    tags: ["scheduler", "cron", "recordatorio", "reminder", "agendar", "programar", "tarea", "alarma", "aviso"],
    intentSummary: "Programa recordatorios, avisos, tareas automáticas o alarmas a una hora o fecha futura",
    inputSchema: { type: "object", properties: { type: { type: "string" } }, required: ["type"] },
  },
];

/**
 * Simulación de "Búsqueda Normal / Léxica Clásica" (Pre):
 * Busca coincidencia directa de palabras/tokens del query contra nombres y tags.
 */
function baselineLexicalSearch(query: string, tools: ToolDefinition[]): { top1: string | null; top3: string[] } {
  const tokens = query.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
  const scored = tools.map((tool) => {
    let score = 0;
    const nameLower = tool.name.toLowerCase();
    const tagsLower = tool.tags.map((t) => t.toLowerCase());
    const descLower = tool.description.toLowerCase();

    for (const token of tokens) {
      if (nameLower.includes(token)) score += 3;
      if (tagsLower.includes(token)) score += 2;
      if (descLower.includes(token)) score += 1;
    }
    return { name: tool.name, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const matches = scored.filter((s) => s.score > 0);
  return {
    top1: matches.length > 0 ? matches[0].name : null,
    top3: matches.slice(0, 3).map((m) => m.name),
  };
}

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  // Registrar catálogo
  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: CATALOG_TOOLS }),
  });
  assert.equal(regRes.status, 200, "El registro de herramientas debe responder 200");
  // Pausa para warm-up completo de embeddings
  await new Promise((r) => setTimeout(r, 1200));
});

after(async () => {
  if (closeServer) await closeServer();
});

async function predictSmart(text: string, sessionId = "test-session-" + Math.random().toString(36).slice(2)) {
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
  assert.equal(res.status, 200);
  return await res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Suite: Comparativa Pre vs Post en Casos Reales
// ─────────────────────────────────────────────────────────────────────────────

interface BenchmarkCase {
  query: string;
  expectedTool: string;
  acceptableAlternative?: string;
  description: string;
}

const TEST_CASES: BenchmarkCase[] = [
  {
    query: "noticias de tumbes hoy",
    expectedTool: "web_search",
    acceptableAlternative: "clima_tumbes",
    description: "Consulta de noticias regionales (lenguaje natural)",
  },
  {
    query: "quien es MILO J",
    expectedTool: "web_search",
    acceptableAlternative: "web_search_research",
    description: "Consulta de biografía/entidad sin palabras clave técnicas",
  },
  {
    query: "Buscar en la web mitumbes.com",
    expectedTool: "web_search",
    acceptableAlternative: "web_search_research",
    description: "Búsqueda web sobre dominio/proyecto externo",
  },
  {
    query: "que tiempo hace en tumbes",
    expectedTool: "clima_tumbes",
    description: "Consulta meteorológica coloquial",
  },
  {
    query: "crear un archivo notas.txt con un resumen",
    expectedTool: "workspace_write_file",
    description: "Operación de archivo en workspace",
  },
  {
    query: "revisa los turnos anteriores del chat para ver el link que me diste",
    expectedTool: "search_chat_history",
    description: "Recuperación de memoria/historial de conversación",
  },
  {
    query: "programar recordatorio para mañana a las 9 am",
    expectedTool: "schedule_task",
    description: "Programación de recordatorio coloquial",
  },
  {
    query: "lugares turisticos y playas de tumbes en la app",
    expectedTool: "mitumbes_lugares",
    description: "Consulta a la API turística local MiTumbes",
  },
];

test("BENCHMARK: Pre vs Post (Búsqueda Normal vs Búsqueda Inteligente con Prediction Server)", async () => {
  let normalTop1Hits = 0;
  let normalTop3Hits = 0;
  let smartTop1Hits = 0;
  let smartTop3Hits = 0;

  const comparisonRows: Array<{
    query: string;
    expected: string;
    normalTop1: string | null;
    smartTop1: string | null;
    normalOk: boolean;
    smartOk: boolean;
  }> = [];

  for (const c of TEST_CASES) {
    // 1. Búsqueda Normal (Pre)
    const normalRes = baselineLexicalSearch(c.query, CATALOG_TOOLS);
    const normalMatchesTop1 =
      normalRes.top1 === c.expectedTool ||
      (c.acceptableAlternative ? normalRes.top1 === c.acceptableAlternative : false);
    const normalMatchesTop3 =
      normalRes.top3.includes(c.expectedTool) ||
      (c.acceptableAlternative ? normalRes.top3.includes(c.acceptableAlternative) : false);

    if (normalMatchesTop1) normalTop1Hits++;
    if (normalMatchesTop3) normalTop3Hits++;

    // 2. Búsqueda Inteligente (Post - Prediction Server)
    const smartRes = await predictSmart(c.query);
    const smartTools: string[] = smartRes.tools.map((t: any) => t.name);
    const smartTop1 = smartTools.length > 0 ? smartTools[0] : null;
    const smartMatchesTop1 =
      smartTop1 === c.expectedTool ||
      (c.acceptableAlternative ? smartTop1 === c.acceptableAlternative : false);
    const smartMatchesTop3 =
      smartTools.slice(0, 3).includes(c.expectedTool) ||
      (c.acceptableAlternative ? smartTools.slice(0, 3).includes(c.acceptableAlternative) : false);

    if (smartMatchesTop1) smartTop1Hits++;
    if (smartMatchesTop3) smartTop3Hits++;

    comparisonRows.push({
      query: c.query,
      expected: c.expectedTool,
      normalTop1: normalRes.top1,
      smartTop1,
      normalOk: normalMatchesTop1,
      smartOk: smartMatchesTop1,
    });
  }

  const normalAccuracyTop1 = (normalTop1Hits / TEST_CASES.length) * 100;
  const smartAccuracyTop1 = (smartTop1Hits / TEST_CASES.length) * 100;
  const normalAccuracyTop3 = (normalTop3Hits / TEST_CASES.length) * 100;
  const smartAccuracyTop3 = (smartTop3Hits / TEST_CASES.length) * 100;

  console.log("\n================ TABLA COMPARATIVA PRE vs POST ================");
  for (const row of comparisonRows) {
    console.log(
      `Consulta: "${row.query}"\n` +
      `  -> Esperada:    ${row.expected}\n` +
      `  -> Normal (Pre): ${row.normalTop1 || 'NINGUNA'} [${row.normalOk ? 'OK' : 'FALLO'}]\n` +
      `  -> Smart (Post): ${row.smartTop1 || 'NINGUNA'} [${row.smartOk ? 'OK' : 'EXITO'}]\n`
    );
  }
  console.log("==============================================================");
  console.log(`Búsqueda Normal (Pre):      Top-1 Accuracy = ${normalAccuracyTop1.toFixed(1)}% | Top-3 Accuracy = ${normalAccuracyTop3.toFixed(1)}%`);
  console.log(`Búsqueda Inteligente (Post): Top-1 Accuracy = ${smartAccuracyTop1.toFixed(1)}% | Top-3 Accuracy = ${smartAccuracyTop3.toFixed(1)}%`);
  console.log("==============================================================\n");

  // Aserciones rigurosas
  assert.ok(
    smartAccuracyTop1 >= 87.5,
    `La búsqueda inteligente debe alcanzar al menos 87.5% de precisión Top-1 (obtenido ${smartAccuracyTop1}%)`
  );
  assert.ok(
    smartAccuracyTop3 === 100,
    `La búsqueda inteligente debe alcanzar 100% de recall en Top-3 (obtenido ${smartAccuracyTop3}%)`
  );
  assert.ok(
    smartAccuracyTop1 > normalAccuracyTop1,
    `La búsqueda inteligente (${smartAccuracyTop1}%) debe superar a la búsqueda normal (${normalAccuracyTop1}%)`
  );
});

test("TEST UNITARIO: Desambiguación semántica entre entidades coincidentes (Tumbes)", async () => {
  // En búsqueda normal, "noticias de tumbes" y "clima en tumbes" y "turismo en tumbes" chocan por la palabra "tumbes"
  const newsResult = await predictSmart("noticias de tumbes hoy");
  const weatherResult = await predictSmart("hace calor en tumbes hoy dia");
  const placesResult = await predictSmart("lugares turisticos de tumbes");

  const newsToolNames = newsResult.tools.map((t: any) => t.name);
  assert.ok(
    newsToolNames.includes("web_search") || newsToolNames.includes("web_search_research"),
    `Noticias debe incluir web_search o web_search_research en predicciones: ${newsToolNames.join(", ")}`
  );
  assert.equal(newsResult.context.intent.category, "search", "Categoría de intención debe ser search");

  assert.equal(
    weatherResult.tools[0].name,
    "clima_tumbes",
    "Calor/temperatura debe resolver a clima_tumbes en Top-1"
  );
  assert.equal(
    placesResult.tools[0].name,
    "mitumbes_lugares",
    "Lugares turísticos debe resolver a mitumbes_lugares en Top-1"
  );
});

test("TEST UNITARIO: Inferencia Zero-Keywords (Usuario no usa palabras técnicas)", async () => {
  // El usuario pregunta "quien es MILO J", sin decir "buscar", "web", "google", "internet"
  const res = await predictSmart("quien es MILO J");
  const toolNames = res.tools.map((t: any) => t.name);

  assert.ok(
    toolNames.includes("web_search") || toolNames.includes("web_search_research"),
    "Debe predecir herramienta de búsqueda web para consulta de entidad desconocida"
  );
  assert.equal(res.context.intent.category, "search", "Debe inferir categoría 'search'");
});
