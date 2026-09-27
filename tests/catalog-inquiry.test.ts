import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "catalog-test-tenant-2026";

const MITUMBES_TOOLS: ToolDefinition[] = [
  {
    id: "mitumbes_item_crear",
    name: "mitumbes_item_crear",
    group: "mitumbes",
    category: "content",
    description: "Crea un ítem nuevo en el catálogo unificado de MiTumbes: lugares, hoteles, restaurantes y rutas.",
    tags: ["mitumbes", "crear", "item", "hotel"],
    intentSummary: "Crea un ítem nuevo en MiTumbes",
    inputSchema: {},
  },
  {
    id: "mitumbes_item_listar",
    name: "mitumbes_item_listar",
    group: "mitumbes",
    category: "content",
    description: "Lista ítems del sistema MiTumbes con filtros opcionales.",
    tags: ["mitumbes", "listar", "item"],
    intentSummary: "Lista ítems de MiTumbes",
    inputSchema: {},
  },
  {
    id: "mitumbes_item_actualizar",
    name: "mitumbes_item_actualizar",
    group: "mitumbes",
    category: "content",
    description: "Actualiza parcialmente un ítem existente en MiTumbes.",
    tags: ["mitumbes", "actualizar", "item"],
    intentSummary: "Actualiza un ítem de MiTumbes",
    inputSchema: {},
  },
  {
    id: "mitumbes_hero_obtener",
    name: "mitumbes_hero_obtener",
    group: "mitumbes",
    category: "portal",
    description: "Devuelve las diapositivas y campañas activas que se muestran en el hero de la portada de MiTumbes.",
    tags: ["mitumbes", "hero", "portada"],
    intentSummary: "Obtiene los banners del hero de MiTumbes",
    inputSchema: {},
  },
  {
    id: "mitumbes_usuario_listar",
    name: "mitumbes_usuario_listar",
    group: "mitumbes",
    category: "users",
    description: "Lista usuarios y cuentas del sistema MiTumbes.",
    tags: ["mitumbes", "usuario", "listar"],
    intentSummary: "Lista usuarios de MiTumbes",
    inputSchema: {},
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Escribe un archivo en el workspace.",
    tags: ["workspace", "write"],
    intentSummary: "Escribe archivos",
    inputSchema: {},
  },
];

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: MITUMBES_TOOLS }),
  });
  assert.equal(regRes.status, 200);
});

after(async () => {
  if (closeServer) await closeServer();
});

test("Consulta de catálogo: 'Qué herramientas de mitumbes tienes' devuelve todas las herramientas del namespace mitumbes", async () => {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: "session-catalog-1",
      tenant: TENANT_ID,
      text: "Qué herramientas de mitumbes tienes",
      source: "human",
    }),
  });

  assert.equal(res.status, 200);
  const data = (await res.json()) as { tools: ToolDefinition[]; context?: { intent?: { primaryAction?: string } } };
  const names = data.tools.map((t) => t.name);

  // Debe incluir todas las herramientas de la familia mitumbes, no solo 2
  assert.ok(names.includes("mitumbes_item_crear"), "Debe incluir mitumbes_item_crear");
  assert.ok(names.includes("mitumbes_item_listar"), "Debe incluir mitumbes_item_listar");
  assert.ok(names.includes("mitumbes_item_actualizar"), "Debe incluir mitumbes_item_actualizar");
  assert.ok(names.includes("mitumbes_hero_obtener"), "Debe incluir mitumbes_hero_obtener");
  assert.ok(names.includes("mitumbes_usuario_listar"), "Debe incluir mitumbes_usuario_listar");
  // No debe incluir herramientas de otras familias como workspace
  assert.ok(!names.includes("workspace_write_file"), "No debe mezclar herramientas ajenas a mitumbes");
});

test("Subtarea de agente: 'Listar herramientas MCP disponibles de mitumbes' entrega el inventario completo de la familia", async () => {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: "session-catalog-2",
      tenant: TENANT_ID,
      text: "Listar herramientas MCP disponibles de mitumbes para el usuario",
      source: "agent",
    }),
  });

  assert.equal(res.status, 200);
  const data = (await res.json()) as { tools: ToolDefinition[] };
  const names = data.tools.map((t) => t.name);

  assert.ok(names.includes("mitumbes_item_crear"));
  assert.ok(names.includes("mitumbes_item_listar"));
  assert.ok(names.includes("mitumbes_hero_obtener"));
});
