/**
 * Tests unitarios de Juicio Agudo para Consultas Vagas, Implícitas y Seguimientos
 *
 * Valida que remtk-prediction:
 * 1. Resuelva consultas vagas ("el primero", "intenta de nuevo", "hazlo") preservando
 *    el contexto histórico y aprovechando confirmCache sin descarte destructivo de historial.
 * 2. Emplee juicio agudo en solicitudes implícitas ("crea el item Casas de Punta Sal...")
 *    sin contaminar la predicción con herramientas destructivas ajenas
 *    (mitumbes_item_imagen_eliminar, mitumbes_publicidad_eliminar).
 * 3. Mantenga la poda adaptativa por gap sin inflar artificialmente el número de tools
 *    ni asignar puntajes sintéticos ficticios a sub-ventanas no relevantes.
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "vague-implicit-test-tenant";
const SCOPE_KEY = TENANT_ID;

// Catálogo con herramientas de gestión MiTumbes y Workspace
const TEST_CATALOG: ToolDefinition[] = [
  {
    id: "mitumbes_item_crear",
    name: "mitumbes_item_crear",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Crea un ítem turístico o comercial (hotel, restaurante, atractivo) en MiTumbes.",
    tags: ["item", "crear", "hotel", "hospedaje", "alojamiento", "restaurante", "registrar"],
    intentSummary: "Crea un ítem o negocio en MiTumbes",
    inputSchema: { type: "object", properties: { name: { type: "string" }, type: { type: "string" } }, required: ["name"] },
  },
  {
    id: "mitumbes_item_actualizar",
    name: "mitumbes_item_actualizar",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Actualiza los datos de un ítem existente en MiTumbes.",
    tags: ["item", "actualizar", "modificar", "editar"],
    intentSummary: "Actualiza un ítem existente",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  },
  {
    id: "mitumbes_item_listar",
    name: "mitumbes_item_listar",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Lista o busca ítems registrados en MiTumbes por filtros o texto.",
    tags: ["item", "listar", "buscar", "consultar", "hoteles"],
    intentSummary: "Lista ítems en MiTumbes",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "mitumbes_item_eliminar",
    name: "mitumbes_item_eliminar",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Elimina de forma permanente un ítem del sistema MiTumbes.",
    tags: ["item", "eliminar", "borrar", "destruir"],
    intentSummary: "Elimina un ítem del sistema",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  },
  {
    id: "mitumbes_item_imagen_eliminar",
    name: "mitumbes_item_imagen_eliminar",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Elimina una imagen de la galería de un ítem en MiTumbes.",
    tags: ["imagen", "eliminar", "borrar", "galeria", "foto"],
    intentSummary: "Elimina una imagen de la galería de un ítem",
    inputSchema: { type: "object", properties: { itemId: { type: "string" }, imageId: { type: "string" } }, required: ["itemId", "imageId"] },
  },
  {
    id: "mitumbes_publicidad_eliminar",
    name: "mitumbes_publicidad_eliminar",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Elimina una campaña publicitaria en MiTumbes.",
    tags: ["publicidad", "eliminar", "campaña", "borrar"],
    intentSummary: "Elimina una campaña publicitaria",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  },
  {
    id: "mitumbes_ruta_crear",
    name: "mitumbes_ruta_crear",
    group: "plugin_mitumbes",
    category: "plugin_tool",
    description: "Crea una ruta turística con múltiples paradas.",
    tags: ["ruta", "crear", "turismo", "itinerario"],
    intentSummary: "Crea una nueva ruta turística",
    inputSchema: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace con el contenido dado.",
    tags: ["workspace", "write", "guardar archivo", "crear archivo", "persistir"],
    intentSummary: "Escribe contenido en un archivo del workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del workspace.",
    tags: ["workspace", "read", "leer archivo"],
    intentSummary: "Lee un archivo del workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
];

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: TEST_CATALOG }),
  });
  assert.equal(regRes.status, 200);
  // Esperar warm-up
  await new Promise((resolve) => setTimeout(resolve, 3000));
});

after(async () => {
  if (closeServer) await closeServer();
});

async function predict(body: {
  sessionId: string;
  text: string;
  history?: Array<{ role: "user" | "assistant" | "system"; content: string }>;
  priorPlan?: string;
}): Promise<{
  tools: ToolDefinition[];
  complexity: string;
  rankedScores: number[];
  context: any;
}> {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tenant: TENANT_ID,
      source: "human",
      ...body,
    }),
  });
  assert.equal(res.status, 200);
  return (await res.json()) as any;
}

test("Caso 1: Juicio Agudo en Solicitud Extensa de Creación sin Destructivas", async () => {
  const sessionId = "session-sharp-create-1";
  const userPrompt =
    "crea el item Casas de Punta Sal Hotel Karibian, es un hotel, ES DE PUNTA SAL, la descripcion te lo inventas, " +
    "acá está los servicios 🎄🌴 Esta Navidad, vuelve a lo simple… vuelve a la familia 🌴🎄 Disfruta unas fiestas " +
    "diferentes en el Punta Sal Hotel Karibian, un lugar rústico y acogedor, perfecto para compartir en familia. " +
    "Teléfono 972 670 007, google maps https://maps.app.goo.gl/iaz6U77wxYC68BK97, en ES, EN, PT, QUECHUA";

  const result = await predict({
    sessionId,
    text: userPrompt,
  });

  const toolNames = result.tools.map((t) => t.name);

  // 1. Debe predecir la creación del ítem como herramienta principal
  assert.ok(
    toolNames.includes("mitumbes_item_crear"),
    `Debe incluir mitumbes_item_crear, obtenido: ${toolNames.join(", ")}`,
  );

  // 2. Juicio agudo: NO debe incluir herramientas destructivas o no solicitadas
  assert.ok(
    !toolNames.includes("mitumbes_item_imagen_eliminar"),
    `NO debe incluir mitumbes_item_imagen_eliminar`,
  );
  assert.ok(
    !toolNames.includes("mitumbes_publicidad_eliminar"),
    `NO debe incluir mitumbes_publicidad_eliminar`,
  );
  assert.ok(
    !toolNames.includes("mitumbes_item_eliminar"),
    `NO debe incluir mitumbes_item_eliminar`,
  );

  // 3. Cantidad de herramientas acotada y precisa (no inflada a 20)
  assert.ok(
    toolNames.length <= 5,
    `El juicio debe ser agudo y acotado (<= 5 tools), obtenido: ${toolNames.length} tools (${toolNames.join(", ")})`,
  );
});

test("Caso 2: Turno Vago 'el primero' mantiene contexto y devuelve el plan previo", async () => {
  const sessionId = "session-followup-vague-1";

  // Turno 1: el usuario pide listar o buscar hoteles
  const turn1 = await predict({
    sessionId,
    text: "lista los hoteles en Punta Sal",
  });
  const t1Names = turn1.tools.map((t) => t.name);
  assert.ok(t1Names.includes("mitumbes_item_listar"), "Turno 1 debe sugerir mitumbes_item_listar");

  // Turno 2: el usuario responde vagamente "el primero"
  const turn2 = await predict({
    sessionId,
    text: "el primero",
    history: [
      { role: "user", content: "lista los hoteles en Punta Sal" },
      { role: "assistant", content: "He encontrado las opciones para listar los hoteles de Punta Sal." },
    ],
  });

  const t2Names = turn2.tools.map((t) => t.name);
  // Al decir "el primero", el sistema no debe quedar con 0 tools ni perder la coherencia
  assert.ok(
    t2Names.length > 0,
    "Turno vago 'el primero' debe mantener herramientas activas del contexto",
  );
  assert.ok(
    t2Names.includes("mitumbes_item_listar") || t2Names.includes("mitumbes_item_crear"),
    `Debe mantener la continuidad de ítems, obtenido: ${t2Names.join(", ")}`,
  );
});

test("Caso 3: Turno 'intenta de nuevo' preserva contexto tras fallo o retry", async () => {
  const sessionId = "session-retry-vague-1";

  // Turno 1: Crear ítem
  const turn1 = await predict({
    sessionId,
    text: "crea el hotel Karibian en Punta Sal",
  });
  assert.ok(turn1.tools.some((t) => t.name === "mitumbes_item_crear"));

  // Turno 2: "intenta de nuevo"
  const turn2 = await predict({
    sessionId,
    text: "intenta de nuevo",
    history: [
      { role: "user", content: "crea el hotel Karibian en Punta Sal" },
      { role: "assistant", content: "Hubo un bloqueo temporal en la carga." },
    ],
  });

  const t2Names = turn2.tools.map((t) => t.name);
  assert.ok(
    t2Names.includes("mitumbes_item_crear"),
    `'intenta de nuevo' debe preservar la herramienta de creación, obtenido: ${t2Names.join(", ")}`,
  );
});

test("Caso 4: Turno elíptico breve con historial 'hazlo'", async () => {
  const sessionId = "session-hazlo-1";

  const turn1 = await predict({
    sessionId,
    text: "guarda la información del hotel Karibian en un archivo de texto",
  });
  assert.ok(turn1.tools.some((t) => t.name === "workspace_write_file"));

  const turn2 = await predict({
    sessionId,
    text: "hazlo",
    history: [
      { role: "user", content: "guarda la información del hotel Karibian en un archivo de texto" },
      { role: "assistant", content: "¿Deseas que proceda a guardar el archivo en el workspace?" },
    ],
  });

  const t2Names = turn2.tools.map((t) => t.name);
  assert.ok(
    t2Names.includes("workspace_write_file"),
    `'hazlo' debe ejecutar la acción pendiente (workspace_write_file), obtenido: ${t2Names.join(", ")}`,
  );
});
