import { describe, test, before } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { createSystem } from "../src/index";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

process.env.QDRANT_ENABLED = "0";

const CATALOG_TOOLS: ToolDefinition[] = [
  {
    id: "knowledge_search",
    name: "knowledge_search",
    group: "knowledge",
    category: "search",
    description: "Busca artículos y documentación técnica en la base de conocimiento.",
    tags: ["doc", "knowledge"],
    intentSummary: "Busca en la base de conocimiento",
    inputSchema: { type: "object", properties: { q: { type: "string" } }, required: ["q"] },
  },
  {
    id: "ask_plan_requirements",
    name: "ask_plan_requirements",
    group: "planner",
    category: "requirements",
    description: "Pregunta requerimientos para armar un plan técnico.",
    tags: ["plan", "requirements"],
    intentSummary: "Consulta requerimientos de plan",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o escribe un archivo en el workspace.",
    tags: ["workspace", "write"],
    intentSummary: "Escribe un archivo",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
];

const TENANT_ID = "diag-unrelated-" + Date.now();
let system: Awaited<ReturnType<typeof createSystem>>;

before(async () => {
  const cfg = loadConfig();
  system = await createSystem(cfg);
  await system.orchestrator.registerTools(TENANT_ID, CATALOG_TOOLS);
});

describe("Aislamiento de turno corto no relacionado con historial", () => {
  test("Turno 'bien' tras consulta de archivo debe aislarse y devolver 0 herramientas", async () => {
    const sessionId = "session-unrelated-bien";

    // 1. Turno previo sobre archivo
    await system.orchestrator.predict({
      tenant: TENANT_ID,
      sessionId,
      text: "de qué trata el archivo que subí y qué contiene",
      source: "human",
    });

    // 2. Turno 'bien' con historial anterior
    const history = [
      { role: "user" as const, content: "de qué trata el archivo que subí y qué contiene" },
      { role: "assistant" as const, content: "El archivo contiene la especificación de requerimientos de la base de datos." },
    ];

    const res = await system.orchestrator.predict({
      tenant: TENANT_ID,
      sessionId,
      text: "bien",
      source: "human",
      history,
    });

    console.log("\n--- [Diagnóstico Turno 'bien'] ---");
    console.log("Tools devueltas:", res.tools.map((t) => t.name));
    console.log("Ranked scores:", res.rankedScores);
    console.log("Cantidad devuelta:", res.tools.length);

    assert.equal(res.tools.length, 0, "No debe forzar herramientas para una respuesta corta no relacionada como 'bien'");
  });
});
