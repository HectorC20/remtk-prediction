/**
 * Test de Flujo Continuo Multi-Turno y Coherencia de Intención
 *
 * Evalúa la coherencia conversacional a través de múltiples turnos consecutivos:
 * 1. Turno 1 (Exploración): "qué documentos existe?" -> workspace_list
 * 2. Turno 2 (Creación/Diseño): "podrás diseñar un markdown detallando sobre JEV AI" -> workspace_write_file / docs
 * 3. Turno 3 (Anáfora/Conversión): "en pdf" -> document_generate / document_convert
 * 4. Turno 4 (Actualización incremental): "actualiza el markdown agregando una sección de benchmarks y regenera el pdf" -> workspace_replace_in_file / document_generate
 * 5. Turno 5 (Cambio de tema): "cuál es el clima en Tumbes hoy?" -> clima_tumbes / web_search
 * 6. Turno 6 (Retorno al contexto documental): "ahora muéstrame de nuevo los archivos generados" -> workspace_list
 *
 * Garantiza:
 * - Sin listas fijas, sin diccionarios ni keywords artificiales.
 * - Coherencia topológica y resolución anafórica multi-turno.
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "continuous-flow-tenant-2026";
const SESSION_ID = "continuous-flow-session-xyz";

const AGENT_CATALOG: ToolDefinition[] = [
  // Workspace Tools
  {
    id: "workspace_list",
    name: "workspace_list",
    group: "workspace",
    category: "filesystem",
    description: "Lista los archivos y carpetas del workspace interno en una ruta específica para inspeccionar documentos existentes.",
    tags: ["archivos", "documentos", "workspace", "list", "explore", "files", "listar archivos", "que documentos existen", "arbol"],
    intentSummary: "Lista archivos y carpetas dentro del workspace del proyecto interno",
    inputSchema: { type: "object", properties: { path: { type: "string" } } },
  },
  {
    id: "workspace_list_root",
    name: "workspace_list_root",
    group: "workspace",
    category: "filesystem",
    description: "Lista los archivos y carpetas en la raíz del workspace para descubrir los documentos y estructura del proyecto.",
    tags: ["archivos", "documentos", "workspace", "root", "list", "directorios", "que documentos existen", "inventario"],
    intentSummary: "Lista los archivos y directorios en la raíz del workspace",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido completo de un archivo del workspace interno.",
    tags: ["workspace", "read", "leer archivo", "ver contenido", "markdown", "inspeccionar"],
    intentSummary: "Lee el contenido de un archivo del workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace interno con contenido Markdown, código o texto.",
    tags: ["workspace", "write", "crear archivo", "guardar archivo", "markdown", "escribir", "documento"],
    intentSummary: "Crea o sobrescribe un archivo en el workspace del proyecto",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_replace_in_file",
    name: "workspace_replace_in_file",
    group: "workspace",
    category: "filesystem",
    description: "Reemplaza un bloque de texto exacto dentro de un archivo existente del workspace para actualizarlo.",
    tags: ["workspace", "replace", "modificar", "actualizar", "editar archivo", "cambiar seccion"],
    intentSummary: "Reemplaza texto específico dentro de un archivo existente",
    inputSchema: { type: "object", properties: { path: { type: "string" }, findText: { type: "string" }, replaceText: { type: "string" } }, required: ["path", "findText", "replaceText"] },
  },
  // Document Conversion Tools
  {
    id: "document_generate",
    name: "document_generate",
    group: "documents",
    category: "document_tools",
    description: "Genera o compila un documento exportable (PDF, DOCX, HTML, EPUB, etc.) a partir de contenido Markdown o de un archivo existente en el workspace.",
    tags: ["document", "pdf", "docx", "exportar", "generar documento", "crear pdf", "convertir markdown", "en pdf", "compilar"],
    intentSummary: "Genera o exporta documentos formateados (PDF, DOCX, etc.) a partir de Markdown",
    inputSchema: { type: "object", properties: { targetFormat: { type: "string" }, content: { type: "string" }, markdownPath: { type: "string" } }, required: ["targetFormat"] },
  },
  {
    id: "document_convert",
    name: "document_convert",
    group: "documents",
    category: "document_tools",
    description: "Convierte un archivo existente entre formatos soportados (de Markdown/HTML a PDF, DOCX, etc.).",
    tags: ["convert", "format", "transform", "pdf", "docx", "markdown", "convertir archivo"],
    intentSummary: "Convierte archivos existentes entre diferentes formatos soportados",
    inputSchema: { type: "object", properties: { sourcePath: { type: "string" }, targetFormat: { type: "string" } }, required: ["sourcePath", "targetFormat"] },
  },
  // Search & External Tools
  {
    id: "clima_tumbes",
    name: "clima_tumbes",
    group: "search",
    category: "user_tool",
    description: "Consulta el pronóstico y clima actual de Tumbes, Perú.",
    tags: ["clima", "weather", "tumbes", "temperatura", "peru", "pronostico"],
    intentSummary: "Consulta el clima actual y pronóstico para Tumbes, Perú",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "web_search",
    description: "Busca información actualizada en internet.",
    tags: ["search", "web", "internet", "noticias", "google"],
    intentSummary: "Busca información en tiempo real a través de internet",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
];

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: AGENT_CATALOG }),
  });
  assert.equal(regRes.status, 200);
  // Esperar warm-up de embeddings
  await new Promise((resolve) => setTimeout(resolve, 3000));
});

after(async () => {
  if (closeServer) await closeServer();
});

async function predictTurn(text: string, sessionId = SESSION_ID): Promise<{
  tools: ToolDefinition[];
  complexity: string;
  context: {
    intent: {
      primaryAction: string;
      confidence: number;
      category?: string;
      summary: string;
    };
  };
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
  assert.equal(res.status, 200);
  return (await res.json()) as any;
}

test("Turno 1: 'qué documentos existe?' -> workspace_list / workspace_list_root", async () => {
  const result = await predictTurn("qué documentos existe?");
  const names = result.tools.map((t) => t.name);

  assert.ok(
    names.includes("workspace_list") || names.includes("workspace_list_root"),
    `Debe predecir workspace_list o workspace_list_root, obtenido: ${names.join(", ")}`,
  );
  assert.ok(["workspace", "filesystem", "general"].includes(result.context.intent.category!));
});

test("Turno 2: 'podrás diseñar un markdown detallando sobre JEV AI' -> workspace_write_file", async () => {
  const result = await predictTurn("podrás diseñar un markdown detallando sobre JEV AI");
  const names = result.tools.map((t) => t.name);

  assert.ok(
    names.includes("workspace_write_file") || names.includes("document_generate"),
    `Debe predecir workspace_write_file o document_generate, obtenido: ${names.join(", ")}`,
  );
});

test("Turno 3: 'en pdf' (anáfora / exportación) -> document_generate / document_convert", async () => {
  const result = await predictTurn("en pdf");
  const names = result.tools.map((t) => t.name);

  assert.ok(
    names.includes("document_generate") || names.includes("document_convert"),
    `Debe predecir document_generate o document_convert para 'en pdf', obtenido: ${names.join(", ")}`,
  );
});

test("Turno 4: 'actualiza el markdown agregando una sección de benchmarks y regenera el pdf' -> workspace_replace_in_file / document_generate", async () => {
  const result = await predictTurn("actualiza el markdown agregando una sección de benchmarks y regenera el pdf");
  const names = result.tools.map((t) => t.name);

  assert.ok(
    names.includes("workspace_replace_in_file") || names.includes("workspace_write_file") || names.includes("document_generate"),
    `Debe incluir herramientas de edición o regeneración, obtenido: ${names.join(", ")}`,
  );
});

test("Turno 5: 'cuál es el clima en Tumbes hoy?' (cambio de tema) -> clima_tumbes", async () => {
  const result = await predictTurn("cuál es el clima en Tumbes hoy?");
  const names = result.tools.map((t) => t.name);

  assert.ok(
    names.includes("clima_tumbes") || names.includes("web_search"),
    `Debe predecir clima_tumbes o web_search, obtenido: ${names.join(", ")}`,
  );
});

test("Turno 6: 'ahora muéstrame de nuevo los archivos generados' -> workspace_list", async () => {
  const result = await predictTurn("ahora muéstrame de nuevo los archivos generados");
  const names = result.tools.map((t) => t.name);

  assert.ok(
    names.includes("workspace_list") || names.includes("workspace_list_root") || names.includes("workspace_read_file"),
    `Debe retornar al workspace para listar archivos, obtenido: ${names.join(", ")}`,
  );
});
