import { test } from "node:test";
import assert from "node:assert/strict";
import { SkillMemoryService } from "../src/predict/services/skill-memory.service";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { loadConfig } from "../src/config";
import { parseKeyRemtk } from "../src/shared/scope";

test("key-remtk Multi-tenant & Multi-agent Isolation", async (t) => {
  const engine = new EmbeddingEngineService(loadConfig());
  const service = new SkillMemoryService(engine);

  const tenantA_agent1 = "tenant-alpha::agent-1";
  const tenantB_agent2 = "tenant-beta::agent-2";
  const commonSessionId = "session-100";

  await t.test("parseKeyRemtk extracts tenant and agentId from header or body", () => {
    const fromHeader = parseKeyRemtk("7f4b4b79::ee208755");
    assert.equal(fromHeader.tenant, "7f4b4b79");
    assert.equal(fromHeader.agentId, "ee208755");
    assert.equal(fromHeader.scopeKey, "7f4b4b79::ee208755");

    const fallback = parseKeyRemtk(undefined, "user123", "agent999");
    assert.equal(fallback.tenant, "user123");
    assert.equal(fallback.agentId, "agent999");
    assert.equal(fallback.scopeKey, "user123::agent999");
  });

  await t.test("learnSkill isolates learned skills between different key-remtk scopes", async () => {
    // Tenant A aprende la habilidad de crear y actualizar ítems gastronómicos
    await service.learnSkill(
      tenantA_agent1,
      {
        id: "skill-gastronomia",
        name: "Gestión de Restaurantes y Bares",
        description: "Crear y actualizar restaurantes, gastrobares y locales de comida",
        intentSummary: "crear restaurante gastronomico",
        tools: ["item_crear", "item_actualizar"],
        parameterGrounding: {
          categoryId: [{ id: "cat-resto-1", name: "Restaurantes y Bares" }],
        },
      },
    );

    // Tenant B aprende una habilidad completamente distinta
    await service.learnSkill(
      tenantB_agent2,
      {
        id: "skill-inmuebles",
        name: "Gestión Inmobiliaria",
        description: "Crear y actualizar propiedades, departamentos y casas",
        intentSummary: "crear propiedad inmobiliaria",
        tools: ["inmueble_crear", "inmueble_actualizar"],
        parameterGrounding: {
          categoryId: [{ id: "cat-prop-9", name: "Departamentos" }],
        },
      },
    );

    // Predicción bajo Tenant A
    const resA = await service.predictSkill({
      sessionId: commonSessionId,
      keyRemtk: tenantA_agent1,
      text: "actualiza la foto del restaurante",
    });
    assert.equal(resA.matchedSkill?.id, "skill-gastronomia");

    // Predicción bajo Tenant B con el mismo texto no debe matchear la skill del Tenant A
    const resB = await service.predictSkill({
      sessionId: commonSessionId,
      keyRemtk: tenantB_agent2,
      text: "actualiza la foto del departamento",
    });
    assert.equal(resB.matchedSkill?.id, "skill-inmuebles");
  });

  await t.test("trackEntity isolates active session entities between key-remtk scopes", async () => {
    // Tenant A registra "La Pichanga Gastrobar" en session-100
    await service.trackEntity(
      commonSessionId,
      {
        id: "3265a5e6-86a2-4156-a23f-39cfd7ff1171",
        type: "item",
        name: "La Pichanga Gastrobar",
        state: { lifecycle: "created", pendingAction: "image_upload" },
      },
      tenantA_agent1,
    );

    // Tenant B registra "Residencial Punta Sal" en la misma sessionId común
    await service.trackEntity(
      commonSessionId,
      {
        id: "99999999-0000-0000-0000-000000000001",
        type: "propiedad",
        name: "Residencial Punta Sal",
        state: { lifecycle: "created", pendingAction: "image_upload" },
      },
      tenantB_agent2,
    );

    // Tenant A recupera SOLO "La Pichanga Gastrobar"
    const entitiesA = service.getSessionEntities(commonSessionId, tenantA_agent1);
    assert.equal(entitiesA.length, 1);
    assert.equal(entitiesA[0].name, "La Pichanga Gastrobar");

    // Tenant B recupera SOLO "Residencial Punta Sal"
    const entitiesB = service.getSessionEntities(commonSessionId, tenantB_agent2);
    assert.equal(entitiesB.length, 1);
    assert.equal(entitiesB[0].name, "Residencial Punta Sal");

    // Predicción de Skill de Tenant A resuelve la entidad correcta y pre-llena ID y URL de adjunto
    const predA = await service.predictSkill({
      sessionId: commonSessionId,
      keyRemtk: tenantA_agent1,
      text: "te pido que actualices la imagen",
      attachments: [{ url: "http://backend.remtk.com/images/pichanga.jpg" }],
    });

    assert.equal(predA.resolvedEntity?.id, "3265a5e6-86a2-4156-a23f-39cfd7ff1171");
    assert.equal(predA.preResolvedArgs.id, "3265a5e6-86a2-4156-a23f-39cfd7ff1171");
    assert.equal(predA.preResolvedArgs.image, "http://backend.remtk.com/images/pichanga.jpg");
    assert.equal(predA.suggestedAction, "execute_direct");
  });
});
