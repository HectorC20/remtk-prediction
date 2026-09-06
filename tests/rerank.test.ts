/** Tests unitarios del re-rank y utilidades (herméticos, sin modelo ni Qdrant). */
import { test } from "node:test";
import assert from "node:assert/strict";
import { EmbeddingEngineService } from "../src/embedding/embedding-engine";
import { adaptiveThreshold, estimateComplexity } from "../src/predict/services/rerank.service";

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

test("adaptiveThreshold: scores planos sin gap → devuelve el mínimo", () => {
  const scored = [
    { name: "a", score: 0.9 },
    { name: "b", score: 0.89 },
    { name: "c", score: 0.88 },
  ];
  const out = adaptiveThreshold(scored, 2, 5, 0.15);
  assert.deepEqual(out, ["a", "b"]);
});

test("estimateComplexity: rangos por cantidad y promedio", () => {
  const one = [{ name: "a", score: 0.5 }];
  assert.equal(estimateComplexity(one, 1), "simple");
  assert.equal(estimateComplexity(Array.from({ length: 12 }, (_, i) => ({ name: `t${i}`, score: 0.6 })), 12), "moderate");
  assert.equal(estimateComplexity(Array.from({ length: 25 }, (_, i) => ({ name: `t${i}`, score: 0.7 })), 25), "complex");
  assert.equal(estimateComplexity([], 0), "simple");
});
