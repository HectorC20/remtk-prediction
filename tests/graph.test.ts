/** Tests del enrutador topológico (herméticos, sin ONNX real ni Qdrant). */
import { test } from "node:test";
import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding-engine";
import { RerankService, topologicalSort } from "../src/predict/services/rerank.service";
import { ToolGraphCacheService } from "../src/predict/services/graph-cache.service";
import { SessionStateCacheService } from "../src/predict/services/session-state-cache.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";
import type { GraphEdge, TenantToolGraph } from "../src/shared/interfaces/graph.interface";

function makeConfig() {
  return loadConfig({
    ...process.env,
    ONNX_ENABLED: "0",
    QDRANT_ENABLED: "0",
    ONNX_ADAPTIVE_MIN_SCORE: "0",
  } as NodeJS.ProcessEnv);
}

function def(name: string): ToolDefinition {
  return {
    id: name,
    name,
    group: "g",
    category: "c",
    description: "",
    tags: [],
    intentSummary: "",
    inputSchema: {},
  };
}

test("topologicalSort: pre-requisitos antes que dependientes", () => {
  const edges: GraphEdge[] = [{ from: "a", to: "b", type: "PREREQUISITE", weight: 0.85 }];
  const scores = new Map([
    ["b", 0.9],
    ["a", 0.8],
  ]);
  assert.deepEqual(topologicalSort(["a", "b"], edges, scores), ["a", "b"]);
});

test("topologicalSort: tolera ciclos y devuelve todos los nodos", () => {
  const edges: GraphEdge[] = [
    { from: "a", to: "b", type: "PREREQUISITE", weight: 0.85 },
    { from: "b", to: "a", type: "PREREQUISITE", weight: 0.85 },
  ];
  const scores = new Map([
    ["a", 0.9],
    ["b", 0.8],
  ]);
  const order = topologicalSort(["a", "b"], edges, scores);
  assert.equal(order.length, 2);
  assert.deepEqual([...order].sort(), ["a", "b"]);
});

test("ToolGraphCacheService: construye nodos, aristas y matriz de adyacencia", async () => {
  const cfg = makeConfig();
  const engine = new EmbeddingEngineService(cfg);
  const cache = new ToolGraphCacheService(engine);

  const tools: ToolDefinition[] = [
    { ...def("fetch_customer_profile"), group: "crm" },
    { ...def("generate_history_pdf"), group: "reports", prerequisites: ["fetch_customer_profile"] },
    { ...def("send_email"), group: "mail", conflicts: ["send_whatsapp"] },
    { ...def("send_whatsapp"), group: "messaging" },
    { ...def("web_search"), group: "search" },
    { ...def("scrape_url"), group: "search" },
  ];

  const graph = await cache.buildTenantGraph("tenant-graph", tools);
  assert.equal(graph.nodes.size, 6);
  assert.equal(graph.adjacencyMatrix.length, 36); // 6 x 6

  const edges = cache.edges("tenant-graph");
  const has = (type: string) => edges.some((e) => e.type === type);
  assert.ok(has("PREREQUISITE"), "debe haber arista PREREQUISITE");
  assert.ok(has("MUTUALLY_EXCLUSIVE"), "debe haber arista MUTUALLY_EXCLUSIVE");
  assert.ok(has("CO_OCCURRENCE"), "debe haber arista CO_OCCURRENCE");

  const prereq = edges.find((e) => e.type === "PREREQUISITE");
  assert.deepEqual([prereq?.from, prereq?.to], ["fetch_customer_profile", "generate_history_pdf"]);
});

test("graphFilter: propaga pre-requisitos y ordena topológicamente", () => {
  const cfg = makeConfig();
  const rerank = new RerankService(cfg);

  const nodes = new Map();
  nodes.set("a", { id: "a", name: "a", embedding: new Float32Array([1, 0]), prerequisites: [], conflicts: [], definition: def("a") });
  nodes.set("b", { id: "b", name: "b", embedding: new Float32Array([0, 1]), prerequisites: [], conflicts: [], definition: def("b") });
  const graph: TenantToolGraph = {
    tenant: "t",
    versionHash: "h",
    nodes,
    adjacencyMatrix: new Float32Array([0, 0.85, 0, 0]), // A[a][b] = 0.85
    toolIndexMap: new Map([
      ["a", 0],
      ["b", 1],
    ]),
  };
  const edges: GraphEdge[] = [{ from: "a", to: "b", type: "PREREQUISITE", weight: 0.85 }];
  const catalog = { get: (name: string) => graph.nodes.get(name)?.definition };

  const result = rerank.graphFilter({
    zt: new Float32Array([1, 0]),
    graph,
    edges,
    lexicalScores: new Map(),
    catalog,
    modelSize: "hash",
  });

  assert.deepEqual(result.tools.map((t) => t.name), ["a", "b"]);
  assert.equal(result.graph.executionOrder.join(","), "a,b");
  assert.ok(Math.abs(result.rankedScores[0] - 1.0) < 1e-6);
  assert.ok(Math.abs(result.rankedScores[1] - 0.17) < 1e-3); // 0 + 0.20 * 0.85 * 1.0
});

test("graphFilter: resuelve mutexes dejando el de mayor score", () => {
  const cfg = makeConfig();
  const rerank = new RerankService(cfg);

  const nodes = new Map();
  nodes.set("a", { id: "a", name: "a", embedding: new Float32Array([0, 1]), prerequisites: [], conflicts: [], definition: def("a") });
  nodes.set("b", { id: "b", name: "b", embedding: new Float32Array([1, 0]), prerequisites: [], conflicts: [], definition: def("b") });
  const graph: TenantToolGraph = {
    tenant: "t",
    versionHash: "h",
    nodes,
    adjacencyMatrix: new Float32Array(4),
    toolIndexMap: new Map([
      ["a", 0],
      ["b", 1],
    ]),
  };
  const edges: GraphEdge[] = [{ from: "a", to: "b", type: "MUTUALLY_EXCLUSIVE", weight: 1.0 }];
  const catalog = { get: (name: string) => graph.nodes.get(name)?.definition };

  const result = rerank.graphFilter({
    zt: new Float32Array([1, 0]),
    graph,
    edges,
    lexicalScores: new Map(),
    catalog,
    modelSize: "hash",
  });

  assert.deepEqual(result.tools.map((t) => t.name), ["b"]);
});

test("SessionStateCacheService: mezcla convexa sin cambio de tema", async () => {
  const cfg = makeConfig();
  const engine = new EmbeddingEngineService(cfg);
  const state = new SessionStateCacheService(engine);

  const first = await state.resolve("s1", "hola mundo");
  assert.equal(first.topicShift, true);

  const second = await state.resolve("s1", "hola mundo");
  assert.equal(second.topicShift, false);

  // Vector normalizado L2 (norma ~1).
  let norm = 0;
  for (const v of second.zt) norm += v * v;
  assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-6);
});

test("SessionStateCacheService: reset ante cambio de tema", async () => {
  const cfg = makeConfig();
  const engine = new EmbeddingEngineService(cfg);
  const state = new SessionStateCacheService(engine);

  await state.resolve("s2", "hola mundo");
  const shifted = await state.resolve("s2", "zzz qqq");
  assert.equal(shifted.topicShift, true);
});

test("graphFilter: latencia < 15 ms con 50 tools", async () => {
  const cfg = makeConfig();
  const engine = new EmbeddingEngineService(cfg);
  const cache = new ToolGraphCacheService(engine);

  const tools: ToolDefinition[] = Array.from({ length: 50 }, (_, i) => ({
    ...def(`tool_${i}`),
    group: `g${i % 5}`,
    description: `descripcion de la herramienta ${i}`,
    tags: [`tag${i}`],
    intentSummary: `intento ${i}`,
  }));
  const graph = await cache.buildTenantGraph("lat", tools);
  const rerank = new RerankService(cfg);
  const zt = (await engine.embedQuery("consulta de prueba")).embedding;
  const catalog = { get: (name: string) => graph.nodes.get(name)?.definition };

  const start = performance.now();
  rerank.graphFilter({
    zt,
    graph,
    edges: cache.edges("lat"),
    lexicalScores: new Map(),
    catalog,
    modelSize: "hash",
  });
  const elapsed = performance.now() - start;

  assert.ok(elapsed < 15, `latencia ${elapsed.toFixed(2)}ms >= 15ms`);
});
