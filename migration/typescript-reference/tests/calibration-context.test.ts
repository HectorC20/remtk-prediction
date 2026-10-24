/** Tests unitarios para CalibrationService y Predicción de Contexto DAG */
import { test } from "node:test";
import assert from "node:assert/strict";
import { CalibrationService } from "../src/predict/services/calibration.service";
import { RerankService } from "../src/predict/services/rerank.service";
import { loadConfig } from "../src/config";
import type { TenantToolGraph, GraphEdge } from "../src/shared/interfaces/graph.interface";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

test("CalibrationService: corrección de anisotropía elimina el suelo basal", () => {
  // Coseno por debajo del suelo basal (~0.68) se amortigua hacia 0
  const noise = CalibrationService.correctAnisotropy(0.60);
  assert.ok(noise < 0.05, `esperaba amortiguación a < 0.05, obtuve ${noise}`);

  // Coseno idéntico a 1.0 da 1.0
  assert.equal(CalibrationService.correctAnisotropy(1.0), 1.0);

  // Coseno por encima del suelo se reescala proporcionalmente
  const scaled = CalibrationService.correctAnisotropy(0.84, 0.68);
  assert.ok(Math.abs(scaled - 0.5) < 1e-3, `esperaba ~0.5, obtuve ${scaled}`);
});

test("CalibrationService: sigmoide y Temperature Scaling devuelven P in [0, 1]", () => {
  const probHigh = CalibrationService.calibrateProbability(1.2);
  assert.ok(probHigh > 0.85 && probHigh <= 1.0, `probHigh fuera de rango: ${probHigh}`);

  const probLow = CalibrationService.calibrateProbability(-0.5);
  assert.ok(probLow >= 0 && probLow < 0.15, `probLow fuera de rango: ${probLow}`);

  const ranked = [1.3, 0.9, 0.6, 0.2];
  const calibrated = CalibrationService.calibrateRankedScores(ranked);
  assert.equal(calibrated.length, ranked.length);

  for (let i = 0; i < calibrated.length; i++) {
    assert.ok(calibrated[i] >= 0 && calibrated[i] <= 1, `score no está en [0,1]: ${calibrated[i]}`);
    if (i > 0) {
      assert.ok(calibrated[i - 1] >= calibrated[i], "no preserva orden decreciente");
    }
  }
});

test("RerankService & Context: predice contexto estructural y anticipa herramientas siguientes en el DAG", () => {
  const cfg = loadConfig({
    ...process.env,
    ONNX_ENABLED: "0",
    QDRANT_ENABLED: "0",
    ONNX_ADAPTIVE_MIN_SCORE: "0",
  } as NodeJS.ProcessEnv);
  const rerank = new RerankService(cfg);

  const def = (name: string): ToolDefinition => ({
    id: name,
    name,
    group: "auth_suite",
    category: "security",
    description: "",
    tags: [],
    intentSummary: "",
    inputSchema: {},
  });

  const nodes = new Map();
  nodes.set("login", { id: "login", name: "login", embedding: new Float32Array([1, 0]), prerequisites: [], conflicts: [], definition: def("login") });
  nodes.set("get_profile", { id: "get_profile", name: "get_profile", embedding: new Float32Array([0, 1]), prerequisites: ["login"], conflicts: [], definition: def("get_profile") });

  const graph: TenantToolGraph = {
    tenant: "tenant-test",
    versionHash: "hash-1",
    nodes,
    adjacencyMatrix: new Float32Array(4),
    toolIndexMap: new Map([["login", 0], ["get_profile", 1]]),
  };

  const edges: GraphEdge[] = [
    { from: "login", to: "get_profile", type: "PREREQUISITE", weight: 0.9 },
  ];

  const result = rerank.graphFilter({
    zt: new Float32Array([1, 0]), // Vector apunta a login
    graph,
    edges,
    lexicalScores: new Map(),
    learned: { scores: new Map(), weight: 0, terms: [] },
    catalog: { get: (name) => def(name) },
    modelSize: "hash",
  });

  assert.ok(result.context, "debe generar PredictedContext");
  assert.equal(result.context.intent.category, "auth_suite");
  assert.ok(result.calibratedScores && result.calibratedScores.length > 0, "debe incluir calibratedScores");
  assert.ok(result.context.intent.confidence >= 0 && result.context.intent.confidence <= 1, "confianza en [0,1]");

  // Como "login" es prerequisito de "get_profile", si solo se ejecuta "login", get_profile debe anticiparse
  if (result.tools.some((t) => t.name === "login") && !result.tools.some((t) => t.name === "get_profile")) {
    assert.ok(result.context.anticipation.suggestedNextTools.includes("get_profile"), "debe anticipar get_profile");
  }
});
