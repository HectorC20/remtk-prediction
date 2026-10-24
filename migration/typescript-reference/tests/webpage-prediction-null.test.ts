import { describe, test, before } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { createSystem } from "../src/index";
import { VarianteCService } from "../src/predict/services/variante-c.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

process.env.QDRANT_ENABLED = "0";

const CATALOG_TOOLS: ToolDefinition[] = [
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace interno del proyecto con el contenido especificado (HTML, CSS, JavaScript, TypeScript, React, Astro, NestJS, Go, Python, SQL, Docker, etc.), páginas web, landing pages, frontend, configuraciones, scripts o seeds.",
    tags: ["workspace", "create file", "crear archivo", "html", "css", "javascript", "web", "pagina web"],
    intentSummary: "Crea o sobrescribe un archivo en el workspace del proyecto (código HTML/CSS/JS, frontend, web, scripts, datos)",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del workspace del proyecto (HTML, CSS, JS, etc.) para inspeccionar código o verificar estructura.",
    tags: ["workspace", "leer archivo", "inspeccionar", "read file"],
    intentSummary: "Inspecciona el contenido actual de un archivo del proyecto antes de editarlo o usarlo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_list",
    name: "workspace_list",
    group: "workspace",
    category: "filesystem",
    description: "Lista los archivos y carpetas del workspace interno en una ruta específica.",
    tags: ["workspace", "list", "archivos"],
    intentSummary: "Lista archivos y carpetas dentro del workspace del proyecto",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "webSearch",
    description: "Busca en la web e internet información sobre artistas, canciones, noticias, clima o documentación.",
    tags: ["search", "web", "internet", "musica", "cancion"],
    intentSummary: "Busca información en internet",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
];

const TENANT_ID = "diag-webpage-null-" + Date.now();
let system: Awaited<ReturnType<typeof createSystem>>;
let varianteC: VarianteCService;

before(async () => {
  const cfg = loadConfig();
  system = await createSystem(cfg);
  varianteC = new VarianteCService(system.engine);
  await system.orchestrator.registerTools(TENANT_ID, CATALOG_TOOLS);
  await varianteC.indexScope(TENANT_ID, CATALOG_TOOLS);
});

describe("Diagnóstico de Retorno Nulo en Creación de Página Web", () => {
  const prompt = "creame una página web sobre lo que hemos hablado";
  const history = [
    { role: "user" as const, content: "busca en internet a JOJI Y SUS ULTIMAS CANCIONES" },
    { role: "assistant" as const, content: "Joji regresó con Piss In The Wind y temas como PIXELATED KISSES." },
    { role: "user" as const, content: "el track completo" },
    { role: "assistant" as const, content: "Tracklist completo de Piss In The Wind: 1. PIXELATED KISSES, 2. If It Only Gets Better, 3. Past Won't Leave My Bed..." },
  ];

  test("Evaluar qué predice VarianteC de forma aislada", async () => {
    const ctx = history.map((m) => m.content).join("\n");
    const res = await varianteC.rank(TENANT_ID, prompt, ctx);
    assert.ok(res, "VarianteC debe producir ranking");
    console.log("\n[VarianteC Diagnóstico]:");
    console.log("Autonomous:", res.autonomous, "Margen:", res.margin.toFixed(4), "Anaphora:", res.anaphora.toFixed(4));
    console.log("Top candidatas:", res.out.map((c) => `${c.name} (fused: ${c.fused.toFixed(4)}, s1: ${c.s1.toFixed(4)}, s2: ${c.s2.toFixed(4)}, isMut: ${c.isMutation})`));
    const arb = varianteC.arbitrate(res);
    console.log("Decisión de arbitraje:", arb.decision, "Tools:", arb.tools);
  });

  test("Evaluar qué predice Orchestrator.predict", async () => {
    const res = await system.orchestrator.predict({
      tenant: TENANT_ID,
      sessionId: "session-web-test",
      text: prompt,
      source: "human",
      history,
    });
    console.log("\n[Orchestrator Diagnóstico]:");
    console.log("Tools devueltas:", res.tools.map((t) => t.name));
    console.log("Ranked scores:", res.rankedScores);
    console.log("Calibrated scores:", res.calibratedScores);
    console.log("Complexity:", res.complexity);
    console.log("Intent:", res.context?.intent);
  });
});
