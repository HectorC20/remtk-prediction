/**
 * Test Riguroso de Generación de Archivos Web (HTML, CSS, JS, Frontend)
 *
 * Valida que solicitudes para generar páginas web, landing pages, componentes
 * interactivos o estilos (HTML, CSS, JS) seleccionen de forma exacta las herramientas
 * de persistencia del workspace (`workspace_write_file`, `workspace_replace_in_file`,
 * `workspace_read_file`) en lugar de limitarse a razonamiento puro sin herramientas.
 *
 * Cobertura:
 * 1. "créame una página web para un restaurante con HTML y CSS" -> workspace_write_file
 * 2. "genera una landing page en el sandbox con index.html, styles.css y script.js" -> workspace_write_file
 * 3. "crea un archivo styles.css con diseño responsive y animaciones" -> workspace_write_file
 * 4. "diseña un componente interactivo en javascript app.js" -> workspace_write_file
 * 5. "actualiza el diseño agregando modo oscuro al styles.css" -> workspace_replace_in_file / workspace_write_file
 * 6. "inspecciona el código de index.html generado" -> workspace_read_file
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "web-file-tenant-2026";

const WORKSPACE_CATALOG: ToolDefinition[] = [
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace interno del proyecto con el contenido especificado (HTML, CSS, JavaScript, TypeScript, React, Astro, NestJS, Go, Python, SQL, Docker, etc.), páginas web, landing pages, frontend, configuraciones, scripts o seeds.",
    tags: [
      "workspace", "create file", "overwrite", "new file", "write",
      "crear archivo", "escribir archivo", "nuevo archivo", "guardar archivo", "codigo", "script", "generar",
      "html", "css", "javascript", "js", "web", "pagina web", "landing page", "frontend", "sitio web", "index.html",
    ],
    intentSummary: "Crea o sobrescribe un archivo en el workspace del proyecto (código HTML/CSS/JS, frontend, web, scripts, datos)",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
        overwrite: { type: "boolean" },
      },
      required: ["path", "content"],
    },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del workspace del proyecto (HTML, CSS, JS, etc.) para inspeccionar código o verificar estructura.",
    tags: ["workspace", "leer archivo", "inspeccionar", "revisar contenido", "read file", "ver codigo"],
    intentSummary: "Inspecciona el contenido actual de un archivo del proyecto antes de editarlo o usarlo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_replace_in_file",
    name: "workspace_replace_in_file",
    group: "workspace",
    category: "filesystem",
    description: "Reemplaza un texto exacto dentro de un archivo existente del workspace para actualizar estilos, scripts o estructura.",
    tags: ["workspace", "editar archivo", "reemplazar bloque", "cambio puntual", "modificar seccion", "actualizar css", "editar html"],
    intentSummary: "Aplica reemplazos puntuales dentro de un archivo existente del proyecto interno",
    inputSchema: { type: "object", properties: { path: { type: "string" }, findText: { type: "string" }, replaceText: { type: "string" } }, required: ["path", "findText", "replaceText"] },
  },
  {
    id: "workspace_list",
    name: "workspace_list",
    group: "workspace",
    category: "filesystem",
    description: "Lista los archivos y carpetas del workspace interno en una ruta específica.",
    tags: ["workspace", "list", "explore", "files", "archivos", "documentos"],
    intentSummary: "Lista archivos y carpetas dentro del workspace del proyecto",
    inputSchema: { type: "object", properties: { path: { type: "string" } } },
  },
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "web_search",
    description: "Busca documentación actualizada o recursos en internet.",
    tags: ["search", "web", "internet", "docs"],
    intentSummary: "Busca información actualizada en internet",
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
    body: JSON.stringify({ tenant: TENANT_ID, tools: WORKSPACE_CATALOG }),
  });
  assert.equal(regRes.status, 200);
  // Esperar warm-up de embeddings
  await new Promise((resolve) => setTimeout(resolve, 4500));
});

after(async () => {
  if (closeServer) await closeServer();
});

async function predict(text: string): Promise<{
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
      sessionId: "web-test-" + Math.random().toString(36).slice(2),
      tenant: TENANT_ID,
      text,
      source: "human",
    }),
  });
  assert.equal(res.status, 200);
  return (await res.json()) as any;
}

test("1. 'créame una página web para un restaurante con HTML y CSS' -> workspace_write_file", async () => {
  const result = await predict("créame una página web para un restaurante con HTML y CSS");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para crear la página web, obtenido: ${toolNames.join(", ")}`,
  );
  assert.ok(result.context.intent.category !== undefined);
});

test("2. 'genera una landing page en el sandbox con index.html, styles.css y script.js' -> workspace_write_file", async () => {
  const result = await predict("genera una landing page en el sandbox con index.html, styles.css y script.js");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para los archivos de la landing page, obtenido: ${toolNames.join(", ")}`,
  );
});

test("3. 'crea un archivo styles.css con diseño responsive y animaciones' -> workspace_write_file", async () => {
  const result = await predict("crea un archivo styles.css con diseño responsive y animaciones");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para styles.css, obtenido: ${toolNames.join(", ")}`,
  );
});

test("4. 'diseña un componente interactivo en javascript app.js' -> workspace_write_file", async () => {
  const result = await predict("diseña un componente interactivo en javascript app.js");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para app.js, obtenido: ${toolNames.join(", ")}`,
  );
});

test("5. 'actualiza el diseño agregando modo oscuro al styles.css' -> workspace_replace_in_file / workspace_write_file", async () => {
  const result = await predict("actualiza el diseño agregando modo oscuro al styles.css");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_replace_in_file") || toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_replace_in_file o workspace_write_file para actualizar CSS, obtenido: ${toolNames.join(", ")}`,
  );
});

test("6. 'inspecciona el código de index.html generado' -> workspace_read_file", async () => {
  const result = await predict("inspecciona el código de index.html generado");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_read_file"),
    `Debe incluir workspace_read_file para inspeccionar HTML, obtenido: ${toolNames.join(", ")}`,
  );
});

test("7. 'intenta crear de nuevo la página html' -> workspace_write_file", async () => {
  const result = await predict("intenta crear de nuevo la página html");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para reintentar crear la página HTML, obtenido: ${toolNames.join(", ")}`,
  );
});

test("8. 'Reintentar la creación de la página HTML' -> workspace_write_file", async () => {
  const result = await predict("Reintentar la creación de la página HTML");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para reintentar creación, obtenido: ${toolNames.join(", ")}`,
  );
});

test("9. 'guardar archivo jev-ai.html en el workspace' -> workspace_write_file", async () => {
  const result = await predict("guardar archivo jev-ai.html en el workspace");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para guardar jev-ai.html, obtenido: ${toolNames.join(", ")}`,
  );
});

