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
    description: "Crea o sobrescribe un archivo en el workspace interno del proyecto con el contenido especificado (HTML, CSS, JavaScript, etc.).",
    tags: ["workspace", "crear archivo", "write file", "guardar archivo"],
    intentSummary: "Crea o escribe un archivo en el workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del workspace del proyecto para inspeccionar su código o estructura.",
    tags: ["workspace", "leer archivo", "read file", "inspeccionar archivo"],
    intentSummary: "Lee o inspecciona el contenido de un archivo del workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_list",
    name: "workspace_list",
    group: "workspace",
    category: "filesystem",
    description: "Lista los archivos y carpetas del workspace interno del proyecto.",
    tags: ["workspace", "listar archivos", "list"],
    intentSummary: "Lista archivos en el workspace",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "webSearch",
    description: "Busca en la web e internet información sobre artistas, noticias, clima o documentación.",
    tags: ["search", "web", "internet", "musica", "noticias"],
    intentSummary: "Busca información en internet",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
];

const TENANT_ID = "diag-null-scenarios-" + Date.now();
let system: Awaited<ReturnType<typeof createSystem>>;
let varianteC: VarianteCService;

before(async () => {
  const cfg = loadConfig();
  system = await createSystem(cfg);
  varianteC = new VarianteCService(system.engine);
  await system.orchestrator.registerTools(TENANT_ID, CATALOG_TOOLS);
  await varianteC.indexScope(TENANT_ID, CATALOG_TOOLS);
});

describe("Diagnóstico de Escenarios que Producen Retorno Nulo", () => {
  test("Escenario 1: Introspección de catálogo general sin prefijo de familia ('qué herramientas propias se tiene ?')", async () => {
    const prompt = "qué herramientas propias se tiene ?";
    const res = await system.orchestrator.predict({
      tenant: TENANT_ID,
      sessionId: "session-introspect",
      text: prompt,
      source: "human",
    });

    console.log("\n--- [Escenario 1: Introspección de Catálogo] ---");
    console.log("Prompt:", prompt);
    console.log("Tools devueltas:", res.tools.map((t) => t.name));
    console.log("Ranked scores:", res.rankedScores);
    console.log("Calibrated scores:", res.calibratedScores);
    console.log("Complexity:", res.complexity);

    const topScore = res.rankedScores?.[0] ?? 0;
    const topCalibrated = res.calibratedScores?.[0] ?? 0;
    const isNullOrBelowGate = res.tools.length === 0 || (topScore < 0.35 && topCalibrated < 0.40);
    console.log("¿Provoca retorno nulo / noTools en key-remtk?:", isNullOrBelowGate);
  });

  test("Escenario 2: Lectura/inspección referencial dependiente ('léeme el archivo que creaste')", async () => {
    const prompt = "léeme el archivo que creaste";
    const history = [
      { role: "user" as const, content: "creame un script index.js con una función suma" },
      { role: "assistant" as const, content: "Archivo index.js creado exitosamente con la función suma." },
    ];
    const ctx = history.map((m) => m.content).join("\n");

    const vcRank = await varianteC.rank(TENANT_ID, prompt, ctx);
    assert.ok(vcRank);
    const vcArb = varianteC.arbitrate(vcRank);

    console.log("\n--- [Escenario 2: Lectura referencial - Variante C] ---");
    console.log("Prompt:", prompt);
    console.log("Autonomous:", vcRank.autonomous, "Anaphora:", vcRank.anaphora.toFixed(4));
    console.log("Top candidata:", vcRank.out[0]?.name, "isMutation:", vcRank.out[0]?.isMutation, "fused:", vcRank.out[0]?.fused.toFixed(4));
    console.log("Decisión de arbitraje:", vcArb.decision, "Tools:", vcArb.tools);

    const res = await system.orchestrator.predict({
      tenant: TENANT_ID,
      sessionId: "session-read-dep",
      text: prompt,
      source: "human",
      history,
    });

    console.log("[Escenario 2: Orchestrator]:");
    console.log("Tools devueltas:", res.tools.map((t) => t.name));
    console.log("Ranked scores:", res.rankedScores);
  });

  test("Escenario 3: Guardado referencial corto ('guárdalo en un archivo')", async () => {
    const prompt = "guárdalo en un archivo";
    const history = [
      { role: "user" as const, content: "busca en internet a JOJI Y SUS ULTIMAS CANCIONES" },
      { role: "assistant" as const, content: "Joji regresó con Piss In The Wind y temas como PIXELATED KISSES." },
      { role: "user" as const, content: "el track completo" },
      { role: "assistant" as const, content: "Tracklist completo: 1. PIXELATED KISSES, 2. If It Only Gets Better, 3. Past Won't Leave My Bed..." },
    ];
    const ctx = history.map((m) => m.content).join("\n");

    const vcRank = await varianteC.rank(TENANT_ID, prompt, ctx);
    assert.ok(vcRank);
    const vcArb = varianteC.arbitrate(vcRank);

    console.log("\n--- [Escenario 3: Guardado referencial - Variante C] ---");
    console.log("Prompt:", prompt);
    console.log("Autonomous:", vcRank.autonomous, "Anaphora:", vcRank.anaphora.toFixed(4));
    console.log("Top candidata:", vcRank.out[0]?.name, "isMutation:", vcRank.out[0]?.isMutation, "fused:", vcRank.out[0]?.fused.toFixed(4));
    console.log("Decisión de arbitraje:", vcArb.decision, "Tools:", vcArb.tools);

    const res = await system.orchestrator.predict({
      tenant: TENANT_ID,
      sessionId: "session-save-dep",
      text: prompt,
      source: "human",
      history,
    });

    console.log("[Escenario 3: Orchestrator]:");
    console.log("Tools devueltas:", res.tools.map((t) => t.name));
    console.log("Ranked scores:", res.rankedScores);
  });
});
