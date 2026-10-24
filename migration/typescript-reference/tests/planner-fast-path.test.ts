import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { decidePlannerFastPath, PlannerFastPathService } from "../src/predict/services/planner-fast-path.service";

describe("PlannerFastPathService: Optimización de Fast-Path para Arneses", () => {
  const service = new PlannerFastPathService({ defaultPlannerTokens: 2500 });

  test("Escenario A: Turno sin herramientas ('como estás', 'interesante') -> Saltea planificador (Fast-Path DIRECT_CHAT)", () => {
    const decision = service.decide({
      tools: [],
      text: "como estás",
      hasPendingPlan: false,
    });

    assert.equal(decision.action, "DIRECT_CHAT");
    assert.equal(decision.skipPlanner, true);
    assert.equal(decision.estimatedTokensSaved, 2500);
    assert.equal(decision.suggestedChatContext?.situation, "DIRECT_CHAT");
    assert.equal(decision.suggestedChatContext?.toolCount, 0);
  });

  test("Escenario B: Turno con múltiples herramientas (requiere planificador DAG)", () => {
    const decision = service.decide({
      tools: [
        { name: "agenda_list" },
        { name: "agenda_create", prerequisites: ["agenda_list"] },
      ],
      text: "agenda una reunión para mañana",
      hasPendingPlan: false,
    });

    assert.equal(decision.action, "FULL_PLANNER");
    assert.equal(decision.skipPlanner, false);
    assert.equal(decision.estimatedTokensSaved, 0);
  });

  test("Escenario C: Turno corto pero con plan previo pendiente -> Requiere planificador para confirmar", () => {
    const decision = service.decide({
      tools: [],
      text: "sí, dale",
      hasPendingPlan: true,
    });

    assert.equal(decision.action, "FULL_PLANNER");
    assert.equal(decision.skipPlanner, false);
    assert.ok(decision.reason.includes("plan pendiente"));
  });

  test("Escenario D: Función utilitaria externa decidePlannerFastPath en una sola línea", () => {
    const directDecision = decidePlannerFastPath({
      tools: [],
      text: "bien",
      hasPendingPlan: false,
      averagePlannerTokens: 2200,
    });

    assert.equal(directDecision.action, "DIRECT_CHAT");
    assert.equal(directDecision.skipPlanner, true);
    assert.equal(directDecision.estimatedTokensSaved, 2200);
  });
});
