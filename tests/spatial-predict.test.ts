import { test } from "node:test";
import assert from "node:assert/strict";
import { SpatialPredictService } from "../src/predict/services/spatial-predict.service";

test("SpatialPredictService - Direct creation with cursor coordinates", async () => {
  const service = new SpatialPredictService();
  const start = performance.now();
  const res = await service.predict({
    query: "Pon un cuadro rojo acá",
    cursor: { x: 0.5, y: 0.5 },
  });
  const latency = performance.now() - start;

  assert.equal(res.tool, "CREATE_SHAPE");
  assert.equal(res.parameters.shapeType, "rectangle");
  assert.equal(res.parameters.color, "#dc2626");
  assert.equal(res.parameters.relativeX, 0.5);
  assert.equal(res.parameters.relativeY, 0.5);
  assert.ok(res.confidence >= 0.85);
  assert.ok(latency < 50, `Latency was ${latency}ms, expected < 50ms`);
});

test("SpatialPredictService - Direct creation of circle with custom color", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Dibuja un círculo azul",
    cursor: { x: 0.25, y: 0.75 },
  });

  assert.equal(res.tool, "CREATE_SHAPE");
  assert.equal(res.parameters.shapeType, "circle");
  assert.equal(res.parameters.color, "#2563eb");
  assert.equal(res.parameters.relativeX, 0.25);
  assert.equal(res.parameters.relativeY, 0.75);
});

test("SpatialPredictService - Anaphoric reference (change color of selected object)", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Píntalo de verde",
    cursor: { x: 0.1, y: 0.1 },
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "CHANGE_COLOR");
  assert.equal(res.parameters.targetId, "rect_123");
  assert.equal(res.parameters.color, "#16a34a");
  assert.ok(res.confidence >= 0.9);
});

test("SpatialPredictService - Relative movement (direction-based)", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Muévelo a la derecha",
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "MOVE_ELEMENT");
  assert.equal(res.parameters.targetId, "rect_123");
  assert.ok(res.parameters.deltaX! > 0);
});

test("SpatialPredictService - Absolute positioning: center", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Muévelo al centro",
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "MOVE_ELEMENT");
  assert.equal(res.parameters.targetId, "rect_123");
  assert.equal(res.parameters.relativeX, 0.5);
  assert.equal(res.parameters.relativeY, 0.5);
});

test("SpatialPredictService - Absolute positioning: top-right corner", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Ponlo en la esquina superior derecha",
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "MOVE_ELEMENT");
  assert.equal(res.parameters.targetId, "rect_123");
  assert.equal(res.parameters.relativeX, 0.85);
  assert.equal(res.parameters.relativeY, 0.15);
});

test("SpatialPredictService - Movement with fine magnitude (un poco)", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Muévelo un poco a la izquierda",
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "MOVE_ELEMENT");
  assert.ok(res.parameters.deltaX! < 0);
  assert.ok(Math.abs(res.parameters.deltaX!) <= 0.05);
});

test("SpatialPredictService - Movement with large magnitude (bastante)", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Bájalo bastante",
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "MOVE_ELEMENT");
  assert.ok(res.parameters.deltaY! >= 0.20);
});

test("SpatialPredictService - Target cursor location (tráelo acá)", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Tráelo acá",
    cursor: { x: 0.72, y: 0.35 },
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "MOVE_ELEMENT");
  assert.equal(res.parameters.relativeX, 0.72);
  assert.equal(res.parameters.relativeY, 0.35);
});

test("SpatialPredictService - Resize element", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: "Hazlo más grande",
    selectedId: "rect_123",
  });

  assert.equal(res.tool, "RESIZE_ELEMENT");
  assert.equal(res.parameters.targetId, "rect_123");
  assert.ok(res.parameters.scale! > 1.0);
});

test("SpatialPredictService - Add text label", async () => {
  const service = new SpatialPredictService();
  const res = await service.predict({
    query: 'Escribe "Login Button"',
    cursor: { x: 0.4, y: 0.6 },
  });

  assert.equal(res.tool, "ADD_TEXT");
  assert.equal(res.parameters.text, "Login Button");
  assert.equal(res.parameters.relativeX, 0.4);
  assert.equal(res.parameters.relativeY, 0.6);
});
