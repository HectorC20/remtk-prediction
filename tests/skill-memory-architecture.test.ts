/**
 * Test de la Nueva Arquitectura: Memoria de Habilidades y Grafo de Entidades Paramétricas.
 *
 * Valida:
 * 1. Aprendizaje de Habilidades con Parameter Grounding (Categorías y Zonas conocidas).
 * 2. Registro dinámico del ciclo de vida y acciones pendientes de entidades por sesión.
 * 3. Resolución exacta del caso nest-2026-09-28 (5).log:
 *    - Usuario dice: "actualiza la imagen de ese item que se ha generado" + URL adjunta.
 *    - La nueva arquitectura resuelve:
 *      * Entidad: La Pichanga Gastrobar (ID: 3265a5e6-86a2-4156-a23f-39cfd7ff1171)
 *      * Herramienta: mitumbes_item_actualizar
 *      * Parámetros pre-resueltos: { id: "3265a5e6...", image: "http://backend.remtk.com/..." }
 *      * suggestedAction: "execute_direct"
 *      * Cero [FALTAN_DATOS] y cero ambigüedad histórica.
 */

import { test, describe, before } from "node:test";
import assert from "node:assert/strict";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { SkillMemoryService } from "../src/predict/services/skill-memory.service";
import { loadConfig } from "../src/config";

let engine: EmbeddingEngineService;
let skillService: SkillMemoryService;

const TENANT_ID = "mitumbes-prod";
const SESSION_ID = "session-test-pichanga-2026";

before(async () => {
  const config = loadConfig();
  engine = new EmbeddingEngineService(config);
  skillService = new SkillMemoryService(engine);

  // 1. Entrenar / Registrar Habilidad 1: Creación de Ítems con Parámetros Grounded
  await skillService.learnSkill(TENANT_ID, {
    id: "skill:mitumbes:item_crear",
    name: "Crear ítem en catálogo",
    description: "Crea un nuevo ítem turístico, restaurante, hotel o actividad",
    intentSummary: "crear nuevo item lugar hotel restaurante actividad en el catálogo",
    tools: ["mitumbes_categoria_listar", "mitumbes_zona_listar", "mitumbes_item_crear"],
    parameterGrounding: {
      categoryId: [
        { id: "cat-uuid-restaurante", name: "Restaurante y Gastronomía" },
        { id: "cat-uuid-hotel", name: "Hoteles y Hospedajes" },
        { id: "cat-uuid-playa", name: "Playas y Atractivos Naturales" },
      ],
      zoneId: [
        { id: "zone-uuid-tumbes", name: "Tumbes Centro" },
        { id: "zone-uuid-puntasal", name: "Punta Sal" },
        { id: "zone-uuid-zorritos", name: "Zorritos" },
      ],
    },
  });

  // 2. Entrenar / Registrar Habilidad 2: Actualización de Imagen de Ítem
  await skillService.learnSkill(TENANT_ID, {
    id: "skill:mitumbes:item_actualizar_imagen",
    name: "Actualizar imagen o foto de ítem",
    description: "Actualiza la imagen de portada o foto principal de un ítem existente",
    intentSummary: "actualizar cambiar subir asignar imagen foto del item generado o existente",
    tools: ["mitumbes_item_actualizar"],
  });
});

describe("Nueva Arquitectura: Memoria de Habilidades y Grafo de Entidades", () => {
  test("1. Parameter Grounding: 'crea un hotel en punta sal' resuelve IDs sin ejecutar listados", async () => {
    const prediction = await skillService.predictSkill({
      sessionId: "session-new-user-1",
      tenant: TENANT_ID,
      text: "crea un hotel en punta sal llamado Karibian",
    });

    console.log("\n[Test Parameter Grounding] Resultado:", {
      skill: prediction.matchedSkill?.name,
      groundedParams: prediction.groundedParameters,
      preResolved: prediction.preResolvedArgs,
      suggestedAction: prediction.suggestedAction,
    });

    assert.ok(prediction.matchedSkill, "Debe hacer match con la habilidad de crear ítem");
    assert.equal(prediction.matchedSkill?.id, "skill:mitumbes:item_crear");

    // Debe resolver automáticamente los UUIDs por similitud semántica
    assert.equal(prediction.preResolvedArgs.categoryId, "cat-uuid-hotel");
    assert.equal(prediction.preResolvedArgs.zoneId, "zone-uuid-puntasal");
    assert.equal(prediction.suggestedAction, "execute_direct");
  });

  test("2. Reproducción del Caso del Log: resolución de 'actualiza la imagen de ese item'", async () => {
    // Paso A: Simular que en el turno anterior se creó "La Pichanga Gastrobar"
    await skillService.trackEntity(
      SESSION_ID,
      {
        id: "3265a5e6-86a2-4156-a23f-39cfd7ff1171",
        type: "item",
        name: "La Pichanga Gastrobar",
        slug: "la-pichanga-gastrobar",
        state: {
          lifecycle: "created",
          pendingAction: "image_upload",
          missingFields: ["image"],
        },
      },
      TENANT_ID,
    );

    // Paso B: El usuario envía exactamente la consulta y el adjunto del log
    const attachedUrl =
      "http://backend.remtk.com/api/v1/files/public/418d60e4-7dd0-44aa-9c8b-b7386c23f754/view";

    const prediction = await skillService.predictSkill({
      sessionId: SESSION_ID,
      tenant: TENANT_ID,
      text: "actualiza la imagen de ese item que se ha generado",
      attachments: [
        {
          fileId: "418d60e4-7dd0-44aa-9c8b-b7386c23f754",
          url: attachedUrl,
        },
      ],
    });

    console.log("\n[Test Log Reproduction] Resultado para Mini-Agente / Planificador:", {
      skill: prediction.matchedSkill?.name,
      entity: prediction.resolvedEntity,
      preResolvedArgs: prediction.preResolvedArgs,
      contextPrompt: prediction.contextPrompt,
      suggestedAction: prediction.suggestedAction,
    });

    // Verificaciones:
    // 1. Debe haber identificado a "La Pichanga Gastrobar" con su UUID real
    assert.ok(prediction.resolvedEntity, "Debe resolver la entidad de la sesión");
    assert.equal(prediction.resolvedEntity?.id, "3265a5e6-86a2-4156-a23f-39cfd7ff1171");
    assert.equal(prediction.resolvedEntity?.name, "La Pichanga Gastrobar");

    // 2. Debe haber emparejado la habilidad de actualización de imagen
    assert.equal(prediction.matchedSkill?.id, "skill:mitumbes:item_actualizar_imagen");
    assert.ok(
      prediction.matchedSkill?.tools.includes("mitumbes_item_actualizar"),
      "Debe seleccionar mitumbes_item_actualizar",
    );

    // 3. Los argumentos requeridos por la herramienta están 100% pre-resueltos
    assert.equal(prediction.preResolvedArgs.id, "3265a5e6-86a2-4156-a23f-39cfd7ff1171");
    assert.equal(prediction.preResolvedArgs.image, attachedUrl);

    // 4. Acción sugerida inmediata: ejecución directa sin [FALTAN_DATOS]
    assert.equal(prediction.suggestedAction, "execute_direct");

    // 5. El contextPrompt generado es estructurado y compacto
    assert.ok(
      prediction.contextPrompt.includes("3265a5e6-86a2-4156-a23f-39cfd7ff1171"),
      "El contexto para el mini-agente contiene el ID sin alucinaciones",
    );
  });
});
