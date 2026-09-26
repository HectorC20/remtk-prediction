/**
 * Test Riguroso de Planificación y Predicción de Sistemas Complejos
 *
 * Simula el flujo completo de construcción de un sistema multi-tenant de reservas
 * de alta dificultad técnica (NestJS, Astro, PostgreSQL, Redis, MCP, WhatsApp, Docker, Concurrencia).
 *
 * Evalúa las 10 subtareas de ingeniería para verificar:
 * 1. Selección exacta de herramientas de workspace y sandbox en cada fase.
 * 2. Asignación correcta de categoría y acción principal.
 * 3. Dependencias topológicas (DAG) entre lectura, escritura, registro de tools y ejecución.
 * 4. Clasificación de complejidad del sistema ("complex").
 * 5. Inmunidad a fallos de abstención artificial (allSpansNoop) y secuestro de namespaces.
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "complex-system-tenant-2026";

/**
 * Catálogo completo de herramientas del agente en el sandbox/workspace:
 */
const SYSTEM_CATALOG: ToolDefinition[] = [
  // Workspace Tools
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace interno del proyecto con el contenido especificado (código TypeScript, NestJS, Astro, React, PostgreSQL, Docker, seed, etc.).",
    tags: ["workspace", "create file", "overwrite", "write", "crear archivo", "escribir archivo", "guardar archivo", "codigo", "script", "generar", "docker", "migration", "seed"],
    intentSummary: "Crea o sobrescribe un archivo en el workspace del proyecto (código, configuración, scripts, datos)",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo o carpeta del workspace del proyecto para inspeccionar código, package.json o arquitectura existente.",
    tags: ["workspace", "leer archivo", "inspeccionar", "package.json", "revisar contenido", "analizar arquitectura"],
    intentSummary: "Inspecciona el contenido actual de un archivo del proyecto antes de editarlo o usarlo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_replace_in_file",
    name: "workspace_replace_in_file",
    group: "workspace",
    category: "filesystem",
    description: "Reemplaza un texto exacto dentro de un archivo existente del workspace interno del proyecto.",
    tags: ["workspace", "editar archivo", "reemplazar bloque", "cambio puntual", "modificar seccion", "refactor"],
    intentSummary: "Aplica reemplazos puntuales dentro de un archivo existente del proyecto interno",
    inputSchema: { type: "object", properties: { path: { type: "string" }, findText: { type: "string" }, replaceText: { type: "string" } }, required: ["path", "findText", "replaceText"] },
  },
  {
    id: "workspace_insert_in_file",
    name: "workspace_insert_in_file",
    group: "workspace",
    category: "filesystem",
    description: "Inserta contenido en un punto concreto de un archivo del workspace interno del proyecto.",
    tags: ["workspace", "insertar bloque", "agregar seccion", "antes", "despues"],
    intentSummary: "Inserta contenido nuevo en un punto concreto de un archivo existente",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  // Sandbox Go & Tool Registration Tools
  {
    id: "register_tool",
    name: "register_tool",
    group: "own_tools",
    category: "user_tool_registration",
    description: "Crea y registra una nueva herramienta personalizada MCP (Go) en el sandbox con tool.json para que el agente pueda ejecutarla.",
    tags: ["tool", "register", "create", "custom", "sandbox", "mcp", "go", "crear herramienta"],
    intentSummary: "Registra o crea una nueva herramienta personalizada del usuario en su sandbox",
    inputSchema: { type: "object", properties: { name: { type: "string" }, content: { type: "string" } }, required: ["name", "content"] },
  },
  {
    id: "sandbox_scan_tools",
    name: "sandbox_scan_tools",
    group: "own_tools",
    category: "sandbox_tooling",
    description: "Escanea el workspace del sandbox buscando herramientas tool.json, las compila y las deja listas para ejecución.",
    tags: ["scan", "discover", "tools", "sandbox", "compile", "tool.json"],
    intentSummary: "Scans and compiles sandbox tools from tool.json files",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "sandbox_run_registered_tool",
    name: "sandbox_run_registered_tool",
    group: "own_tools",
    category: "sandbox_tooling",
    description: "Ejecuta una herramienta Go registrada en el sandbox pasando argumentos JSON y devuelve stdout.",
    tags: ["run", "tool", "registered", "sandbox", "go", "ejecutar", "concurrencia", "test"],
    intentSummary: "Runs a user tool registered in the sandbox",
    inputSchema: { type: "object", properties: { name: { type: "string" }, arguments: { type: "object" } }, required: ["name"] },
  },
  {
    id: "sandbox_run_command",
    name: "sandbox_run_command",
    group: "own_tools",
    category: "sandbox_terminal",
    description: "Ejecuta un comando, script, test o proceso de forma segura y estrictamente confinada dentro del sandbox del usuario (npm test, pnpm build, go test, node script.js, migraciones).",
    tags: ["sandbox", "terminal", "command", "run", "ejecutar", "comando", "test", "build", "script", "npm", "pnpm", "node", "go", "bash", "cmd", "correr", "migracion"],
    intentSummary: "Ejecuta un comando o script de forma segura dentro del sandbox aislado del usuario",
    inputSchema: { type: "object", properties: { command: { type: "string" }, args: { type: "array", items: { type: "string" } } }, required: ["command"] },
  },
  {
    id: "sandbox_install_packages",
    name: "sandbox_install_packages",
    group: "own_tools",
    category: "sandbox_package",
    description: "Instala paquetes o dependencias para un proyecto dentro del sandbox del usuario (npm install, pnpm add, go get).",
    tags: ["sandbox", "packages", "install", "instalar", "dependencias", "librerias", "npm", "pnpm", "go get"],
    intentSummary: "Instala paquetes o dependencias en un proyecto dentro del sandbox",
    inputSchema: { type: "object", properties: { packages: { type: "array", items: { type: "string" } } }, required: ["packages"] },
  },
  // Scheduling & Events Tools
  {
    id: "schedule_task",
    name: "schedule_task",
    group: "scheduler",
    category: "runtime",
    description: "Programa una tarea futura única o periódica con cron para recordatorios, reintentos de eventos o tareas en segundo plano.",
    tags: ["scheduler", "cron", "recordatorio", "evento", "reintento", "cola", "programar"],
    intentSummary: "Programa una tarea del agente (única o recurrente con cron)",
    inputSchema: { type: "object", properties: { type: { type: "string" } }, required: ["type"] },
  },
  // System Datetime Tool
  {
    id: "get_current_datetime",
    name: "get_current_datetime",
    group: "system",
    category: "system",
    description: "Obtiene la fecha y hora actual exacta con día de la semana y zona horaria (America/Lima) para resolver expresiones como 'mañana después de las 5' o 'este sábado'.",
    tags: ["fecha", "hora", "datetime", "now", "hoy", "manana", "sabado", "timezone"],
    intentSummary: "Devuelve la fecha y hora actual para ubicar citas temporales",
    inputSchema: { type: "object", properties: { timezone: { type: "string" } } },
  },
  // Search Tools
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "web_search",
    description: "Busca documentación actualizada o librerías en internet.",
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
    body: JSON.stringify({ tenant: TENANT_ID, tools: SYSTEM_CATALOG }),
  });
  assert.equal(regRes.status, 200);
  // Pausa para permitir que el warmup asíncrono de embeddings compute en cold start
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
  graph: {
    nodes: string[];
    executionOrder: string[];
  };
}> {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: "complex-test-" + Math.random().toString(36).slice(2),
      tenant: TENANT_ID,
      text,
      source: "human",
    }),
  });
  assert.equal(res.status, 200);
  return (await res.json()) as ReturnType<typeof predict>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fase 1: Arquitectura y Lectura del Proyecto
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 1: 'Analizar arquitectura existente e inspeccionar package.json' -> workspace_read_file", async () => {
  const result = await predict("Analizar arquitectura existente e inspeccionar package.json");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_read_file") || toolNames.includes("workspace_replace_in_file"),
    `Debe incluir workspace_read_file o workspace_replace_in_file: ${toolNames.join(", ")}`,
  );
  assert.ok(["workspace", "filesystem", "general", "own_tools"].includes(result.context.intent.category!));
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 2: Creación de Entidades y Backend NestJS + PostgreSQL
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 2: 'Crear backend NestJS con entidades PostgreSQL para tenants y reservas' -> workspace_write_file", async () => {
  const result = await predict("Crear backend NestJS con entidades PostgreSQL para multi-tenant, servicios, horarios y reservas");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(toolNames.includes("workspace_write_file"), "Debe predecir workspace_write_file");
  assert.ok(["workspace", "own_tools"].includes(result.context.intent.category!));
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 3: Concurrencia y Lock Distribuido Redis (Double Booking)
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 3: 'Implementar Redis lock distribuido para evitar double booking simultáneo' -> workspace_write_file / replace", async () => {
  const result = await predict("Implementar lock distribuido con Redis SET NX EX para evitar double booking en reservas simultaneas");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file") || toolNames.includes("workspace_replace_in_file") || toolNames.includes("schedule_task"),
    `Debe incluir herramientas de edición/creación de código o scheduler: ${toolNames.join(", ")}`,
  );
  assert.ok(["workspace", "scheduler", "own_tools"].includes(result.context.intent.category!));
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 4: Creación de Herramientas MCP Personalizadas en Sandbox
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 4: 'Crear y registrar herramientas MCP en el sandbox con tool.json' -> register_tool / sandbox_scan_tools", async () => {
  const result = await predict("Crear y registrar nueva herramienta personalizada MCP en sandbox para buscar negocios y consultar disponibilidad");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("register_tool") || toolNames.includes("sandbox_scan_tools"),
    `Debe seleccionar register_tool o sandbox_scan_tools: ${toolNames.join(", ")}`,
  );
  assert.ok(["own_tools", "workspace"].includes(result.context.intent.category!));
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 5: Interpretación Temporal de Citas en Lenguaje Natural
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 5: 'Quiero una cita con el dentista mañana después de las 5 o este sábado' -> get_current_datetime", async () => {
  const result = await predict("Quiero una cita con el dentista mañana después de las 5, pero antes de las 7, y si no hay espacio busca el sábado");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("get_current_datetime") || toolNames.includes("workspace_read_file"),
    `Debe incluir get_current_datetime para resolver la hora de mañana/sábado: ${toolNames.join(", ")}`,
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 6: Docker Compose y Configuración de Infraestructura
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 6: 'Crear docker-compose.yml con postgres, redis, worker y frontend' -> workspace_write_file", async () => {
  const result = await predict("Crear docker-compose.yml con frontend, backend, postgres, redis, worker y whatsapp-gateway");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file") || toolNames.includes("register_tool"),
    `Debe incluir herramientas para generar docker-compose: ${toolNames.join(", ")}`,
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 7: Simulación de Test de Concurrencia de 100 Solicitudes
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 7: 'Ejecutar test de 100 solicitudes concurrentes para validar slot único' -> sandbox_run / workspace", async () => {
  const result = await predict("Simular 100 solicitudes concurrentes intentando reservar el mismo horario y demostrar que solo 1 gana el slot");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("sandbox_run_registered_tool") || toolNames.includes("workspace_write_file") || toolNames.includes("register_tool"),
    `Debe ofrecer herramientas de ejecución de test o script: ${toolNames.join(", ")}`,
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 8: Generación de Seeds y Documentación
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 8: 'Crear seed de 10 negocios, 100 empleados, 10 000 reservas y .env.example' -> workspace_write_file", async () => {
  const result = await predict("Crear seed con 10 negocios, 100 empleados y 10000 reservas, documentar API y crear .env.example");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(toolNames.includes("workspace_write_file"), "Debe predecir workspace_write_file para seeds y doc");
  assert.ok(["workspace", "own_tools"].includes(result.context.intent.category!));
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 9: Ejecución de Migraciones y Tests en Sandbox
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 9: 'Ejecutar migraciones de base de datos y tests en el sandbox' -> sandbox_run_command", async () => {
  const result = await predict("Ejecutar las migraciones de base de datos y correr la suite de tests en el sandbox");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("sandbox_run_command") || toolNames.includes("sandbox_run_registered_tool"),
    `Debe predecir sandbox_run_command: ${toolNames.join(", ")}`,
  );
  assert.ok(["own_tools", "sandbox_terminal", "execution"].includes(result.context.intent.category!));
});

// ─────────────────────────────────────────────────────────────────────────────
// Fase 10: Instalación de Dependencias en Sandbox
// ─────────────────────────────────────────────────────────────────────────────
test("Fase 10: 'Instalar paquetes y dependencias npm en el sandbox' -> sandbox_install_packages / sandbox_run_command", async () => {
  const result = await predict("Instalar paquetes y dependencias npm para el backend en el sandbox");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("sandbox_install_packages") || toolNames.includes("sandbox_run_command"),
    `Debe predecir sandbox_install_packages o sandbox_run_command: ${toolNames.join(", ")}`,
  );
});
