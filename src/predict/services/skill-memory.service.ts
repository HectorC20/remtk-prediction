/**
 * SkillMemoryService: Implementación alternativa de Memoria de Habilidades y
 * Grafo de Entidades Paramétricas.
 *
 * Resuelve:
 * 1. Anáforas y referencias dinámicas ("ese item generado") buscando en el grafo de
 *    entidades de la sesión según estado y demanda (activityScore).
 * 2. Anclaje de Parámetros (Parameter Grounding): asocia nombres a IDs conocidos
 *    (ej: "Restaurantes" -> cat-123) aprendidos de consultas anteriores de catálogo.
 * 3. Pre-resolución de argumentos para que los mini-agentes no emitan [FALTAN_DATOS].
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { log } from "src/logger";
import { parseKeyRemtk, resolveScopeKey } from "src/shared/scope";
import type {
  GroundedParameterMatch,
  ParameterAnchor,
  SessionEntityState,
  SkillDefinition,
  SkillPredictInput,
  SkillPredictResult,
} from "src/shared/interfaces/skill.interface";

export class SkillMemoryService {
  /** Habilidades aprendidas indexadas por scopeKey -> skillId -> SkillDefinition */
  private readonly skillsByScope = new Map<string, Map<string, SkillDefinition>>();

  /** Grafo de entidades por partición: `${scopeKey}::${sessionId}` -> entityId -> SessionEntityState */
  private readonly entitiesByPartition = new Map<string, Map<string, SessionEntityState>>();

  /** Cache de embeddings de acciones de ciclo de vida para evitar re-computación */
  private readonly actionEmbeddingCache = new Map<string, Float32Array>();

  /** Piso de similitud para que una ancla describa a la consulta. No decide solo: ver ANCLA_MARGEN. */
  private static readonly ANCLA_MINIMA = 0.82;

  /**
   * Distancia mínima entre la mejor ancla y la siguiente de OTRO identificador.
   * Un ancla es un valor que otra consulta usó en el pasado, no una verdad del
   * turno actual; en este espacio vectorial dos nombres de negocio cualesquiera
   * cosenean ~0.85, por encima del piso, así que sin margen el turno nuevo
   * heredaba el id de otro negocio. Sin rival claro se abstiene y el
   * mini-agente resuelve el identificador listando.
   *
   * 0.015 separa lo que miden las dos baterías: las anclas que NO describen a la
   * entidad nombrada en el turno dejan márgenes de 0.000-0.010, y las
   * resoluciones correctas 0.018 o más. Más arriba de esto se empieza a
   * descartar lo correcto ("un hotel en Punta Sal" -> Hoteles gana por 0.018).
   */
  private static readonly ANCLA_MARGEN = 0.015;

  constructor(private readonly engine: EmbeddingEngineService) {}

  /** Clave compuesta para aislar entidades por scopeKey y sessionId */
  private partitionKey(scopeKey: string, sessionId: string): string {
    return `${scopeKey}::${sessionId}`;
  }

  /** Obtiene o calcula el embedding denso de una acción */
  private async getActionEmbedding(action: string): Promise<Float32Array> {
    const cached = this.actionEmbeddingCache.get(action);
    if (cached) return cached;
    const readableAction = action.replace(/[_-]+/g, " ").trim();
    const embRes = await this.engine.embedPassage(readableAction, "small", { high: false });
    this.actionEmbeddingCache.set(action, embRes.embedding);
    return embRes.embedding;
  }

  /**
   * Registra o actualiza el estado de una entidad en la sesión activa bajo un scope/key-remtk.
   * Por ejemplo, tras crear un ítem: lifecycle="created", pendingAction="image_upload".
   *
   * El scope es obligatorio: la partición es `${scopeKey}::${sessionId}` y predictSkill solo
   * lee la del turno, así que un scope por defecto escribiría entidades invisibles.
   */
  async trackEntity(
    sessionId: string,
    entity: {
      id: string;
      type: string;
      name: string;
      slug?: string;
      state?: Partial<SessionEntityState["state"]>;
      attributes?: Record<string, unknown>;
    },
    tenantOrKeyRemtk: string,
    agentId?: string,
  ): Promise<SessionEntityState> {
    const { scopeKey } = parseKeyRemtk(tenantOrKeyRemtk, tenantOrKeyRemtk, agentId);
    const pKey = this.partitionKey(scopeKey, sessionId);

    let sessionMap = this.entitiesByPartition.get(pKey);
    if (!sessionMap) {
      sessionMap = new Map();
      this.entitiesByPartition.set(pKey, sessionMap);
    }

    const existing = sessionMap.get(entity.id);
    const textToEmbed = `${entity.type} ${entity.name} ${entity.slug ?? ""}`.trim();
    const embRes = await this.engine.embedPassage(textToEmbed, "small", { high: false });

    const updated: SessionEntityState = {
      id: entity.id,
      type: entity.type,
      name: entity.name,
      slug: entity.slug ?? existing?.slug,
      attributes: { ...(existing?.attributes ?? {}), ...(entity.attributes ?? {}) },
      state: {
        lifecycle: entity.state?.lifecycle ?? existing?.state.lifecycle ?? "created",
        pendingAction: entity.state?.pendingAction ?? existing?.state.pendingAction,
        missingFields: entity.state?.missingFields ?? existing?.state.missingFields ?? [],
      },
      activityScore: (existing?.activityScore ?? 0.8) + 0.2,
      updatedAt: Date.now(),
      embedding: embRes.embedding,
    };

    sessionMap.set(entity.id, updated);
    log(
      `[skill-memory] entity tracked scope=${scopeKey} session=${sessionId} id=${updated.id} name="${updated.name}" ` +
        `action=${updated.state.pendingAction ?? "none"} lifecycle=${updated.state.lifecycle}`,
    );
    return updated;
  }

  /**
   * Obtiene la lista de entidades registradas para una sesión y scope.
   */
  getSessionEntities(sessionId: string, tenantOrKeyRemtk: string, agentId?: string): SessionEntityState[] {
    const { scopeKey } = parseKeyRemtk(tenantOrKeyRemtk, tenantOrKeyRemtk, agentId);
    const pKey = this.partitionKey(scopeKey, sessionId);
    const sessionMap = this.entitiesByPartition.get(pKey);
    return sessionMap ? Array.from(sessionMap.values()) : [];
  }

  /**
   * Aprende o actualiza una habilidad para un tenant/scope/key-remtk.
   */
  async learnSkill(
    tenantOrKeyRemtk: string,
    skill: {
      id: string;
      name: string;
      description: string;
      intentSummary: string;
      tools: string[];
      parameterGrounding?: Record<string, { id: string; name: string; metadata?: Record<string, unknown> }[]>;
    },
    agentId?: string,
  ): Promise<SkillDefinition> {
    const { scopeKey } = parseKeyRemtk(tenantOrKeyRemtk, tenantOrKeyRemtk, agentId);
    let scopeSkills = this.skillsByScope.get(scopeKey);
    if (!scopeSkills) {
      scopeSkills = new Map();
      this.skillsByScope.set(scopeKey, scopeSkills);
    }

    const embRes = await this.engine.embedPassage(
      `${skill.name}: ${skill.intentSummary} ${skill.description}`,
      "small",
      { high: false },
    );

    // Procesar anclas de parámetros si vienen en el aprendizaje
    const groundedMap: Record<string, ParameterAnchor[]> = {};
    if (skill.parameterGrounding) {
      for (const [key, anchors] of Object.entries(skill.parameterGrounding)) {
        groundedMap[key] = [];
        for (const a of anchors) {
          const aEmb = await this.engine.embedPassage(`${key}: ${a.name}`, "small", { high: false });
          groundedMap[key].push({
            id: a.id,
            name: a.name,
            embedding: aEmb.embedding,
            metadata: a.metadata,
          });
        }
      }
    }

    const existing = scopeSkills.get(skill.id);
    const def: SkillDefinition = {
      id: skill.id,
      name: skill.name,
      description: skill.description,
      intentSummary: skill.intentSummary,
      intentEmbedding: embRes.embedding,
      tools: skill.tools,
      parameterGrounding: {
        ...(existing?.parameterGrounding ?? {}),
        ...groundedMap,
      },
      reinforcementScore: (existing?.reinforcementScore ?? 0) + 1,
    };

    scopeSkills.set(skill.id, def);
    log(`[skill-memory] skill learned scope=${scopeKey} id=${def.id} tools=[${def.tools.join(",")}]`);
    return def;
  }

  /**
   * Predice la habilidad adecuada y resuelve entidades y parámetros grounded para la consulta.
   */
  async predictSkill(input: SkillPredictInput): Promise<SkillPredictResult> {
    const { scopeKey } = parseKeyRemtk(input.keyRemtk, input.tenant, input.agentId);
    const pKey = this.partitionKey(scopeKey, input.sessionId);

    const queryText = (input.text ?? "").trim();
    const queryEmb = await this.engine.embedQuery(queryText, "small", { high: true });

    const scopeSkills = this.skillsByScope.get(scopeKey);
    const sessionEntities = this.entitiesByPartition.get(pKey);

    let bestSkill: SkillDefinition | undefined;
    let bestSkillScore = 0;

    if (scopeSkills) {
      for (const skill of scopeSkills.values()) {
        if (!skill.intentEmbedding) continue;
        const sim = EmbeddingEngineService.cosine(queryEmb.embedding, skill.intentEmbedding);
        if (sim > bestSkillScore) {
          bestSkillScore = sim;
          bestSkill = skill;
        }
      }
    }

    // 1. Resolver entidad de la sesión (puramente vectorial y por actividad)
    let matchedEntity: SessionEntityState | undefined;
    let bestEntitySim = 0;

    if (sessionEntities && sessionEntities.size > 0) {
      for (const entity of sessionEntities.values()) {
        if (!entity.embedding) continue;
        let sim = EmbeddingEngineService.cosine(queryEmb.embedding, entity.embedding);

        // Bonificación vectorial si la acción pendiente coincide semánticamente con la consulta
        if (entity.state.pendingAction) {
          const actionEmb = await this.getActionEmbedding(entity.state.pendingAction);
          const actionSim = EmbeddingEngineService.cosine(queryEmb.embedding, actionEmb);
          if (actionSim > 0.6) {
            sim += actionSim * 0.3;
          }
        }

        // Ponderar con activityScore de la sesión
        const finalScore = sim * 0.7 + entity.activityScore * 0.3;
        if (finalScore > bestEntitySim) {
          bestEntitySim = finalScore;
          matchedEntity = entity;
        }
      }
    }

    // 2. Resolver parámetros grounded por competencia, no por piso absoluto: la
    //    ancla más parecida a la consulta tiene que dejar atrás, por un margen, a
    //    cualquier ancla de OTRO identificador del scope. Ver ANCLA_MARGEN.
    const groundedMatches: GroundedParameterMatch[] = [];
    const preResolvedArgs: Record<string, unknown> = {};

    if (bestSkill?.parameterGrounding) {
      const demas = [...(scopeSkills?.values() ?? [])].filter((s) => s.id !== bestSkill.id);
      for (const paramKey of Object.keys(bestSkill.parameterGrounding)) {
        const mejorPorId = new Map<string, { ancla: ParameterAnchor; sim: number }>();
        const considerar = (lista: ParameterAnchor[] | undefined) => {
          for (const a of lista ?? []) {
            const sim = EmbeddingEngineService.cosine(queryEmb.embedding, a.embedding);
            const previo = mejorPorId.get(a.id);
            if (!previo || sim > previo.sim) mejorPorId.set(a.id, { ancla: a, sim });
          }
        };
        considerar(bestSkill.parameterGrounding[paramKey]);
        // La misma ancla repetida en otra habilidad no compite: corrobora, y por
        // eso se indexa por id y no por ocurrencia.
        for (const otra of demas) considerar(otra.parameterGrounding?.[paramKey]);

        const ranking = [...mejorPorId.values()].sort((x, y) => y.sim - x.sim);
        const ganadora = ranking[0];
        const rival = ranking[1];
        if (!ganadora) continue;
        const margen = rival ? ganadora.sim - rival.sim : Number.POSITIVE_INFINITY;
        if (ganadora.sim >= SkillMemoryService.ANCLA_MINIMA && margen >= SkillMemoryService.ANCLA_MARGEN) {
          groundedMatches.push({
            key: paramKey,
            matchedId: ganadora.ancla.id,
            matchedName: ganadora.ancla.name,
            confidence: ganadora.sim,
          });
          preResolvedArgs[paramKey] = ganadora.ancla.id;
        } else {
          log(
            `[skill-memory] ancla ambigua clave=${paramKey} se abstiene ` +
              `(id=${ganadora.ancla.id} sim=${ganadora.sim.toFixed(3)} ` +
              `rival=${rival ? `${rival.ancla.id}:${rival.sim.toFixed(3)}` : "ninguno"} margen=${
                Number.isFinite(margen) ? margen.toFixed(3) : "-"
              })`,
          );
        }
      }
    }

    // Si encontramos la entidad referenciada, inyectamos su ID
    if (matchedEntity) {
      preResolvedArgs.id = matchedEntity.id;
      preResolvedArgs.entityName = matchedEntity.name;
    }

    // Si viene archivo adjunto y la habilidad o entidad lo requiere, inyectamos la URL
    if (input.attachments && input.attachments.length > 0) {
      const att = input.attachments[0];
      if (att.url) {
        preResolvedArgs.image = att.url;
        preResolvedArgs.imageUrl = att.url;
      }
      if (att.fileId) {
        preResolvedArgs.fileId = att.fileId;
      }
    }

    // Determinar la acción sugerida
    let suggestedAction: "execute_direct" | "discover_prerequisites" | "ask_user" = "ask_user";
    if (bestSkill && bestSkillScore >= 0.75) {
      // Si tenemos los identificadores principales (ej: ID para actualizar, o categoría/zona para crear)
      if (preResolvedArgs.id || (preResolvedArgs.categoryId && preResolvedArgs.zoneId)) {
        suggestedAction = "execute_direct";
      } else {
        suggestedAction = "discover_prerequisites";
      }
    }

    // Generar contexto compacto para mini-agentes / planificador (< 200 tokens)
    let contextPrompt = "";
    if (matchedEntity) {
      contextPrompt += `[ENTIDAD_SESION_ACTIVA] ID: "${matchedEntity.id}" | Nombre: "${matchedEntity.name}" | Tipo: "${matchedEntity.type}" | Estado: ${matchedEntity.state.lifecycle} (${matchedEntity.state.pendingAction ?? "sin_pendientes"})\n`;
    }
    if (groundedMatches.length > 0) {
      contextPrompt += `[PARAMETROS_GROUNDED] ${groundedMatches.map((m) => `${m.key}="${m.matchedName}" (ID: ${m.matchedId})`).join(", ")}\n`;
    }
    if (preResolvedArgs.image) {
      contextPrompt += `[ADJUNTO_DISPONIBLE] image="${preResolvedArgs.image}"\n`;
    }

    return {
      matchedSkill: bestSkill
        ? {
            id: bestSkill.id,
            name: bestSkill.name,
            confidence: bestSkillScore,
            tools: bestSkill.tools,
          }
        : undefined,
      resolvedEntity: matchedEntity
        ? {
            id: matchedEntity.id,
            name: matchedEntity.name,
            type: matchedEntity.type,
            matchedPendingAction: matchedEntity.state.pendingAction,
            // `bestEntitySim` combina similitud y activityScore (que crece sin
            // tope por turno); como confianza se reporta recortada a [0,1].
            confidence: Math.min(1, bestEntitySim),
          }
        : undefined,
      groundedParameters: groundedMatches,
      preResolvedArgs,
      suggestedAction,
      contextPrompt: contextPrompt.trim(),
    };
  }

  /** Limpia el estado de una sesión (cleanup) */
  clearSession(sessionId: string, tenantOrKeyRemtk: string, agentId?: string): void {
    const { scopeKey } = parseKeyRemtk(tenantOrKeyRemtk, tenantOrKeyRemtk, agentId);
    const pKey = this.partitionKey(scopeKey, sessionId);
    this.entitiesByPartition.delete(pKey);
  }
}
