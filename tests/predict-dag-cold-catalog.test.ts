/**
 * Regresión del pre-plan DAG en catálogo frío (docs/testing/nest-2026-09-28.log).
 *
 * El log muestra la 2ª pasada del planificador pidiendo `mitumbes_item_crear` sin
 * sus antecedentes (`mitumbes_categoria_listar`, `mitumbes_zona_listar`): el mini-
 * agente no tenía de dónde sacar los uuid y quemó los ciclos en validaciones 400.
 * Dos causas, ambas cubiertas aquí con catálogo sintético, modo hash (sin ONNX ni
 * Qdrant) y el grafo retrasado a propósito:
 *
 *   §1 `waitForWarmupBounded` — la cota corta en seco un warm-up en vuelo.
 *   §2 `/predict` espera el warm-up (`PREDICT_WARMUP_WAIT_MS`) y toma la ruta
 *      topológica en vez de caer a la plana.
 *   §3 ruta plana: los antecedentes se recomponen con las pre-relaciones
 *      DECLARADAS (`ToolDefinition.prerequisites`), publicando aristas
 *      PREREQUISITE + orden Kahn.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createSystem } from "../src/main";
import { juicioConfigDefaults } from "../src/shared/constants/juicio";
import { resolveScopeKey } from "../src/shared/scope";
import type { AppConfig } from "../src/config";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

const GROUP = "plugin_mitumbes";

function tool(
  name: string,
  description: string,
  params: Record<string, string> = {},
  required: string[] = [],
  prerequisites?: string[],
): ToolDefinition {
  return {
    id: name,
    name,
    description,
    inputSchema: {
      type: "object",
      properties: Object.fromEntries(Object.entries(params).map(([k, t]) => [k, { type: t }])),
      ...(required.length > 0 ? { required } : {}),
    },
    group: GROUP,
    category: "plugin_tool",
    tags: ["MiTumbes"],
    intentSummary: "",
    ...(prerequisites ? { prerequisites } : {}),
  } as ToolDefinition;
}

// Catálogo del log: la mutación exige categoryId/zoneId uuid y los declara como
// pre-requisitos suyos; el ruido ajeno no comparte grupo.
const CATALOG: ToolDefinition[] = [
  tool("mitumbes_categoria_listar", "Devuelve el catálogo completo de categorías disponibles"),
  tool("mitumbes_zona_listar", "Devuelve la lista de zonas geográficas disponibles"),
  tool(
    "mitumbes_item_crear",
    "Crea un ítem en el catálogo unificado de lugares",
    { slug: "string", type: "string", title: "string", categoryId: "string", zoneId: "string" },
    ["slug", "type", "title", "categoryId", "zoneId"],
    ["mitumbes_categoria_listar", "mitumbes_zona_listar"],
  ),
  tool("workspace_write_file", "Escribe un archivo en el workspace local", { path: "string", content: "string" }),
];

function config(overrides: Partial<AppConfig>): AppConfig {
  return {
    portPredict: 0,
    portEmbed: 0,
    portQdrant: 0,
    onnxEnabled: false,
    onnxModelsPath: "./models",
    onnxModelSize: "small",
    adaptiveMinTools: 2,
    adaptiveMaxTools: 5,
    adaptiveGapThreshold: 0.15,
    adaptiveMinScore: 0,
    keywordBoost: 0.15,
    nameAffinityBoost: 0.1,
    familyGatePenalty: 0.2,
    keywordTopK: 20,
    recallLimit: 50,
    maxOutputTools: 50,
    maxCategories: 60,
    qdrantEnabled: false,
    qdrantUrl: "http://localhost:6333",
    qdrantApiKey: "",
    toolsCollection: "mcp_tools",
    keywordsCollection: "tool_keywords",
    synonymsCollection: "query_synonyms",
    memoriesCollection: "contextual_memories",
    learnEnabled: false,
    learnWeight: 0.25,
    learnEta: 0.5,
    learnNegativeGamma: 0.15,
    learnDecayLambda: 0.02,
    learnMinEvents: 3,
    learnSeedWeight: 0.3,
    learnMaxTermsPerTool: 64,
    learnTermMinWeight: 0.05,
    learnMaxPostings: 200,
    learnPersist: false,
    predictWarmupWaitMs: 0,
    ...juicioConfigDefaults,
    ...overrides,
  } as AppConfig;
}

/** Retrasa la construcción del grafo para reproducir el catálogo frío del log. */
function slowGraphBuild(system: Awaited<ReturnType<typeof createSystem>>, ms: number): void {
  const real = system.graphCache.buildTenantGraph.bind(system.graphCache);
  system.graphCache.buildTenantGraph = async (scopeKey: string, tools: ToolDefinition[]) => {
    await new Promise((resolve) => setTimeout(resolve, ms));
    return real(scopeKey, tools);
  };
}

/** Turno de la 2ª pasada: el planificador nombra la herramienta a ejecutar. */
const PLAN_TEXT =
  "Redactar y guardar el ítem La Pichanga Gastrobar en ES, EN, PT y QU";

async function build(overrides: Partial<AppConfig>) {
  const system = await createSystem(config(overrides));
  slowGraphBuild(system, 400);
  await system.orchestrator.registerTools("dag-frio", CATALOG);
  return system;
}

test("§1 waitForWarmupBounded corta en la cota y confirma al concluir", async () => {
  const system = await build({});
  const scopeKey = resolveScopeKey("dag-frio");

  assert.equal(
    await system.orchestrator.waitForWarmupBounded(scopeKey, 0),
    false,
    "con un warm-up en vuelo la cota 0 debe devolver false",
  );
  assert.ok(
    await system.orchestrator.waitForWarmupBounded(scopeKey, 5_000),
    "el warm-up debe concluir dentro de la cota",
  );
  assert.equal(
    await system.orchestrator.waitForWarmupBounded(scopeKey, 0),
    true,
    "sin warm-up pendiente no hay nada que esperar",
  );
});

test("§2 /predict espera el warm-up y toma la ruta topológica", async () => {
  const system = await build({ predictWarmupWaitMs: 5_000 });

  const result = await system.orchestrator.predict({
    sessionId: "sess-wait",
    tenant: "dag-frio",
    text: PLAN_TEXT,
    keywords: ["mitumbes_item_crear"],
    source: "agent",
  });

  const trace = system.debugger.snapshot().traces[0];
  assert.notEqual(trace.warmupWaitTimeout, true, "la espera debió concluir dentro de la cota");
  assert.ok(result.tools.length > 0, "la predicción no puede quedar vacía");
  const order = result.graph.executionOrder;
  assert.ok(
    order.indexOf("mitumbes_categoria_listar") < order.indexOf("mitumbes_item_crear") &&
      order.indexOf("mitumbes_zona_listar") < order.indexOf("mitumbes_item_crear"),
    `los antecedentes deben ejecutarse antes de la mutación: ${order.join(" > ")}`,
  );
});

test("§3 ruta plana: repone los antecedentes declarados con su DAG", async () => {
  // Cota 0: el warm-up de 400 ms sigue en vuelo, así que /predict degrada a la
  // ruta plana — exactamente el caso del log cuando el registro iba en curso.
  const system = await build({ predictWarmupWaitMs: 0 });

  const result = await system.orchestrator.predict({
    sessionId: "sess-flat",
    tenant: "dag-frio",
    text: PLAN_TEXT,
    keywords: ["mitumbes_item_crear"],
    source: "agent",
  });

  const trace = system.debugger.snapshot().traces[0];
  assert.equal(trace.warmupWaitTimeout, true, "debió tomarse la ruta plana por cota agotada");

  const names = result.tools.map((t) => t.name);
  assert.ok(names.includes("mitumbes_item_crear"), `falta la mutación: ${names.join(",")}`);
  assert.ok(
    names.includes("mitumbes_categoria_listar") && names.includes("mitumbes_zona_listar"),
    `faltan los antecedentes declarados: ${names.join(",")}`,
  );

  const prereqEdges = result.graph.edges.filter(
    (e) => e.type === "PREREQUISITE" && e.to === "mitumbes_item_crear",
  );
  assert.equal(prereqEdges.length, 2, "deben publicarse las 2 aristas PREREQUISITE del plan");
  const order = result.graph.executionOrder;
  assert.ok(
    order.indexOf("mitumbes_categoria_listar") < order.indexOf("mitumbes_item_crear") &&
      order.indexOf("mitumbes_zona_listar") < order.indexOf("mitumbes_item_crear"),
    `orden Kahn roto: ${order.join(" > ")}`,
  );
  assert.equal(
    result.calibratedScores?.length,
    result.tools.length,
    "la recalibración debe quedar alineada con el nuevo orden",
  );
});
