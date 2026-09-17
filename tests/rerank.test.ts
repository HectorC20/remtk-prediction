/** Tests unitarios del re-rank y utilidades (herméticos, sin modelo ni Qdrant). */
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import {
  isForeignFamily,
  matchTokenSet,
  namedFamilies,
  toolFamily,
} from "../src/predict/keywords";
import { RerankService, adaptiveThreshold, estimateComplexity } from "../src/predict/services/rerank.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

/** Config hermética: sin piso de relevancia (el test juzga el desempate, no el corte). */
function gateConfig() {
  return loadConfig({
    ...process.env,
    ONNX_ENABLED: "0",
    QDRANT_ENABLED: "0",
    ONNX_ADAPTIVE_MIN_SCORE: "0",
  } as NodeJS.ProcessEnv);
}

function toolDef(name: string): ToolDefinition {
  return {
    id: name,
    name,
    group: "plugin",
    category: "plugin_tool",
    description: "",
    tags: [],
    intentSummary: "",
    inputSchema: {},
  };
}

test("cosine: normaliza y acota a [0,1]", () => {
  assert.equal(EmbeddingEngineService.cosine([1, 0, 0], [1, 0, 0]), 1);
  assert.equal(EmbeddingEngineService.cosine([1, 0, 0], [-1, 0, 0]), 0);
  assert.equal(EmbeddingEngineService.cosine([1, 0], [0, 1]), 0);
  const v = EmbeddingEngineService.cosine([1, 1], [1, 1]);
  assert.ok(Math.abs(v - 1) < 1e-6, `esperaba ~1, obtuve ${v}`);
});

test("cosine: tolera longitudes distintas (min)", () => {
  assert.equal(EmbeddingEngineService.cosine([1, 0, 0], [1]), 1);
});

test("adaptiveThreshold: sin candidatas → vacío", () => {
  assert.deepEqual(adaptiveThreshold([], 2, 5, 0.15), []);
});

test("adaptiveThreshold: con menos que el mínimo devuelve ese número menor", () => {
  const scored = [
    { name: "a", score: 0.9 },
    { name: "b", score: 0.8 },
  ];
  assert.deepEqual(adaptiveThreshold(scored, 5, 10, 0.15), ["a", "b"]);
});

test("adaptiveThreshold: corta en el primer gap > umbral", () => {
  const scored = [
    { name: "a", score: 0.95 },
    { name: "b", score: 0.92 },
    { name: "c", score: 0.91 },
    { name: "d", score: 0.5 }, // gap 0.41 > 0.15 → corta en 3
    { name: "e", score: 0.49 },
  ];
  assert.deepEqual(adaptiveThreshold(scored, 2, 5, 0.15), ["a", "b", "c"]);
});

test("adaptiveThreshold: scores planos sin gap → devuelve el cluster denso", () => {
  const scored = [
    { name: "a", score: 0.9 },
    { name: "b", score: 0.89 },
    { name: "c", score: 0.88 },
  ];
  const out = adaptiveThreshold(scored, 2, 5, 0.15);
  assert.deepEqual(out, ["a", "b", "c"]);
});

test("estimateComplexity: rangos por cantidad y promedio", () => {
  const one = [{ name: "a", score: 0.5 }];
  assert.equal(estimateComplexity(one, 1), "simple");
  assert.equal(estimateComplexity(Array.from({ length: 12 }, (_, i) => ({ name: `t${i}`, score: 0.6 })), 12), "moderate");
  assert.equal(estimateComplexity(Array.from({ length: 25 }, (_, i) => ({ name: `t${i}`, score: 0.7 })), 25), "complex");
  assert.equal(estimateComplexity([], 0), "simple");
});

test("familia: el namespace es el discriminante entre complementos", () => {
  // El token de familia sale del mismo plegado (plurales incluidos) que la
  // consulta, así que `mitumbes` y `mitumb` comparan iguales a ambos lados.
  const mitumbes = toolFamily("mitumbes_item_crear");
  assert.ok(mitumbes);
  assert.equal(toolFamily("mitumbes_lugar_listar"), mitumbes);
  assert.notEqual(toolFamily("schedule_task"), mitumbes);
  // Nombre solo de verbo/ruido: no tiene identidad ni familia.
  assert.equal(toolFamily("mcp_list"), undefined);
});

test("namedFamilies: solo ancla familias realmente nombradas en la consulta", () => {
  const names = ["mitumbes_item_crear", "schedule_task"];
  const conFamilia = namedFamilies(matchTokenSet("mitumbes categoria item"), names);
  assert.deepEqual([...conFamilia], [toolFamily("mitumbes_item_crear")!]);
  // Consulta que no nombra ninguna familia del catálogo → sin ancla.
  assert.equal(namedFamilies(matchTokenSet("programa una tarea"), names).size, 0);
  assert.equal(isForeignFamily(conFamilia, "schedule_task"), true);
  assert.equal(isForeignFamily(conFamilia, "mitumbes_lugar_listar"), false);
  // Sin familia nombrada nadie es ajeno.
  assert.equal(isForeignFamily(new Set(), "schedule_task"), false);
});

test("puerta de familia: la consulta que nombra `mitumbes` descarta las `schedule_*`", () => {
  const svc = new RerankService(gateConfig());
  const names = [
    "schedule_delete",
    "schedule_task",
    "mitumbes_categoria_listar",
    "mitumbes_item_obtener",
  ];
  const catalog = new Map(names.map((n) => [n, toolDef(n)]));
  const learned = { scores: new Map<string, number>(), weight: 0, terms: [] };
  // El coseno prefiere las tools de programación: su descripción habla de
  // "tareas del agente", que es justo el vocabulario del replanteo.
  const reduced = [
    { name: "schedule_delete", score: 0.95 },
    { name: "schedule_task", score: 0.93 },
    { name: "mitumbes_categoria_listar", score: 0.8 },
    { name: "mitumbes_item_obtener", score: 0.78 },
  ];

  // Réplica del replanteo del log: la categoría de la 1ª pasada ancla la familia.
  const conFamilia = svc.filter(
    reduced,
    new Map(),
    learned,
    catalog,
    "small",
    undefined,
    matchTokenSet("mitumbes categoria item verificar crear"),
  );
  assert.deepEqual(
    conFamilia.tools.map((t) => t.name),
    ["mitumbes_categoria_listar", "mitumbes_item_obtener"],
  );

  // Sin familia nombrada la puerta no actúa: manda la señal semántica de siempre.
  const sinFamilia = svc.filter(
    reduced,
    new Map(),
    learned,
    catalog,
    "small",
    undefined,
    matchTokenSet("programar una tarea del agente"),
  );
  assert.deepEqual(sinFamilia.tools.map((t) => t.name), ["schedule_delete", "schedule_task"]);
});
