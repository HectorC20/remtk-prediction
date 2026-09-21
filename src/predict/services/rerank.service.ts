/**
 * RerankService: Capa 3 del pipeline. Fusiona el score cross-idioma de keywords
 * (capa 2) con la confirmación léxica de Qdrant (BM25) y aplica el umbral
 * adaptativo. El resultado final se recorta al tope configurado de herramientas
 * de salida (env opcional `MAX_OUTPUT_TOOLS`, 0-150).
 *
 * El pipeline corre en dos etapas sobre el mismo pool de candidatas:
 *   - Etapa A (categorías): agrupa las candidatas por `group` y conserva las
 *     mejores categorías (env `MAX_CATEGORIES`, default 60). Las categorías con
 *     herramientas cuyo NOMBRE casa con las palabras clave de la consulta nunca
 *     se recortan: dentro de un complemento el grupo es compartido y por tanto
 *     no discrimina familia ninguna.
 *   - Etapa B (decisor): dentro de esas categorías aplica la fusión, el umbral
 *     adaptativo y el tope final de herramientas (env `MAX_OUTPUT_TOOLS`).
 *
 * Migrado de rerank_service.go + adaptive_threshold.go del server Go.
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import type { AppConfig } from "src/config";
import { log } from "src/logger";
import type { PredictedContext, ScoredTool, ToolComplexity, ToolDefinition } from "src/shared/interfaces/domain.interface";
import type { GraphEdge, TenantToolGraph } from "src/shared/interfaces/graph.interface";
import type { LearnedScores } from "src/shared/interfaces/lexical.interface";
import {
  COMPLEXITY_AVG_HIGH,
  COMPLEXITY_AVG_MID,
  COMPLEXITY_COUNT_HIGH,
  COMPLEXITY_COUNT_MID,
  graphPropagationAlphaDefault,
} from "src/shared/constants/predict";
import { isForeignFamily, nameAffinity, namedFamilies } from "../keywords";
import { CalibrationService } from "./calibration.service";

import type { GraphRerankResult, RerankResult } from "src/shared/interfaces";

export type { RerankResult, GraphRerankResult };

/** Catálogo mínimo para resolver definiciones por nombre (evita acoplar Qdrant). */
type Catalog = { get(name: string): ToolDefinition | undefined };

export class RerankService {
  constructor(private readonly config: AppConfig) {}

  /**
   * Fusiona el ranking cross-idioma (reduced) con el score léxico normalizado
   * de Qdrant y el perfil léxico aprendido del canal, y aplica el piso de
   * relevancia + umbral adaptativo.
   *
   * final = cosineKeywords + keywordBoost * lexicalQdrant
   *                          + learnWeightEfectivo * aprendido
   *                          + nameAffinityBoost * afinidad(consulta, nombre)
   *                          - familyGatePenalty * familiaAjena(consulta, nombre)
   *
   * `queryTokens` son los tokens de match de la consulta (incluidas las
   * palabras clave delegadas). Alimentan la afinidad por NOMBRE, que es la
   * señal que separa familias dentro de un mismo complemento, y la puerta de
   * FAMILIA: si la consulta nombra el namespace de alguna tool del pool, las de
   * otras familias (p. ej. `schedule_*` cuando se habla de `mitumbes`) se
   * penalizan hasta caer fuera de la banda del umbral adaptativo.
   */
  filter(
    reduced: ScoredTool[],
    lexicalScores: Map<string, number>,
    learned: LearnedScores,
    catalog: { get(name: string): ToolDefinition | undefined },
    modelSize: string,
    exclude?: string[],
    queryTokens?: Set<string>,
    pinnedNames?: string[],
  ): RerankResult {
    const learnWeight = learned.weight;
    // Exclusión de la segunda pasada: las tools ya ofrecidas se descartan ANTES
    // de la etapa A para que tampoco puntúen su categoría ni ocupen cupo.
    const excluded = exclusionSet(exclude);
    const candidates = excluded ? reduced.filter((r) => !excluded.has(r.name.toLowerCase())) : reduced;
    // Herramientas nombradas EXPLÍCITAMENTE en las palabras clave delegadas: se
    // fijan al frente de la salida y se garantiza su presencia. El score por
    // embeddings no distingue el verbo del nombre (dos tools del mismo
    // complemento comparten tokens de identidad), así que sin este ancla la
    // herramienta pedida puede quedarse fuera del umbral.
    const pinned = pinnedList(pinnedNames, excluded);
    const pinnedSet = new Set(pinned.map((n) => n.toLowerCase()));
    // Etapa A: conserva solo las mejores categorías (grupos) del pool, sin
    // descartar las que contienen herramientas afines al nombre de la consulta.
    const isAffine = affinityPredicate(queryTokens);
    const pool = keepTopCategories(
      candidates,
      (name) => catalog.get(name)?.group,
      this.config.maxCategories,
      (name) => isAffine?.(name) === true || pinnedSet.has(name.toLowerCase()),
    );
    // Puerta de familia: namespaces nombrados explícitamente en las palabras
    // clave. Vacío = la consulta no nombra familia ninguna → no se penaliza.
    const families = namedFamilies(queryTokens, candidates.map((c) => c.name));
    const fused: ScoredTool[] = pool.map((r) => ({
      name: r.name,
      score:
        r.score +
        this.config.keywordBoost * (lexicalScores.get(r.name) ?? 0) +
        learnWeight * (learned.scores.get(r.name) ?? 0) +
        this.config.nameAffinityBoost * nameAffinity(queryTokens, r.name) -
        (isForeignFamily(families, r.name) ? this.config.familyGatePenalty : 0),
    }));
    fused.sort((a, b) => b.score - a.score);

    // Piso de relevancia sobre la señal FUSIONADA (§6.6): `fused` ya incluye
    // `coseno + keywordBoost·BM25 + learnWeight·aprendido`, y su máximo es el
    // primer elemento (está ordenado). Así tanto una confirmación léxica como un
    // término aprendido del canal pueden rescatar una herramienta que el coseno
    // dejaba por debajo del mínimo. Evaluarlo solo sobre el coseno (como antes)
    // apagaba el turno entero cuando la capa semántica no discriminaba nada
    // aunque BM25 sí confirmara el intent.
    const relevance = fused[0]?.score ?? 0;
    if (fused.length > 0) {
      log(
        `[rerank] fused=${fused.length} top1=${fused[0].name}(${fused[0].score.toFixed(3)}) ` +
          `relevancia=${relevance.toFixed(3)} learnW=${learnWeight.toFixed(2)}`,
      );
    }

    // Piso absoluto de relevancia: si la mejor señal fusionada no supera el
    // mínimo, no se selecciona ninguna herramienta. Las nombradas explícitamente
    // se devuelven igual: son una orden del planificador, no una predicción.
    if (this.config.adaptiveMinScore > 0 && relevance < this.config.adaptiveMinScore) {
      const forced = pinnedDefinitions(pinned, (n) => catalog.get(n));
      if (forced.length === 0) {
        log(
          `[rerank] piso de relevancia no superado (relevancia=${relevance.toFixed(3)} < min=${this.config.adaptiveMinScore}) → 0 tools`,
        );
        return {
          tools: [],
          complexity: "simple",
          modelSize,
          rankedScores: [relevance],
          calibratedScores: [CalibrationService.calibrateProbability(relevance)],
          context: {
            intent: {
              primaryAction: "unknown",
              confidence: 0,
              summary: "Piso de relevancia no superado",
            },
            constraints: {
              negations: [],
              isConfirmation: false,
              isExploratory: true,
            },
            dialogState: { phase: "discovery", topicShift: false },
            anticipation: { suggestedNextTools: [], reasoning: "Sin herramientas activas" },
          },
        };
      }
      log(
        `[rerank] piso de relevancia no superado (relevancia=${relevance.toFixed(3)} < min=${this.config.adaptiveMinScore}) ` +
          `pero ${forced.length} tools explícitas → se devuelven`,
      );
      const forcedScores = forced.map(() => relevance);
      return {
        tools: forced,
        complexity: "simple",
        modelSize,
        rankedScores: forcedScores,
        calibratedScores: CalibrationService.calibrateRankedScores(forcedScores),
        context: {
          intent: {
            primaryAction: forced[0]?.category ?? "execute",
            confidence: CalibrationService.calibrateProbability(relevance),
            summary: "Herramientas forzadas explícitas",
          },
          constraints: {
            negations: [],
            isConfirmation: false,
            isExploratory: false,
          },
          dialogState: { phase: "execution", topicShift: false },
          anticipation: { suggestedNextTools: [], reasoning: "Herramientas forzadas por planificador" },
        },
      };
    }

    const selectedNames = adaptiveThreshold(
      fused,
      this.config.adaptiveMinTools,
      this.config.adaptiveMaxTools,
      this.config.adaptiveGapThreshold,
    );

    // Tope final opcional (MAX_OUTPUT_TOOLS, 0-150): recorta lo decidido por el
    // umbral adaptativo, p. ej. 0 para no devolver herramientas nunca.
    const cappedNames = selectedNames.slice(0, this.config.maxOutputTools);

    const byName = new Map<string, ToolDefinition>();
    for (const r of pool) {
      const t = catalog.get(r.name);
      if (t) byName.set(t.name, t);
    }
    const scoreByName = new Map(fused.map((s) => [s.name, s.score]));

    // Las nombradas explícitamente encabezan la salida; si el umbral adaptativo
    // las dejó fuera se añaden igualmente.
    const finalNames = promotePinned(cappedNames, pinned).slice(0, this.config.maxOutputTools);

    const tools: ToolDefinition[] = [];
    const scores: number[] = [];
    for (const name of finalNames) {
      const t = byName.get(name) ?? catalog.get(name);
      if (t) {
        tools.push(t);
        scores.push(scoreByName.get(name) ?? 0);
      }
    }

    const complexity = estimateComplexity(fused, tools.length);
    log(`[rerank] selected=${tools.length} complexity=${complexity}`);

    const calibratedScores = CalibrationService.calibrateRankedScores(scores);
    const topTool = tools[0];
    const context: PredictedContext = {
      intent: {
        primaryAction: topTool?.category ?? "execute",
        confidence: calibratedScores[0] ?? 0.5,
        category: topTool?.group,
        summary: tools.length > 0 ? `Seleccionadas ${tools.length} herramientas` : "Sin herramientas seleccionadas",
      },
      constraints: {
        negations: [],
        isConfirmation: false,
        isExploratory: tools.length === 0,
      },
      dialogState: {
        phase: tools.length > 0 ? "execution" : "discovery",
        topicShift: false,
        activeDomain: topTool?.group,
      },
      anticipation: {
        suggestedNextTools: [],
        reasoning: "Pipeline lineal sin grafo",
      },
    };

    return {
      tools,
      complexity,
      modelSize,
      rankedScores: scores,
      calibratedScores,
      context,
    };
  }

  /**
   * Enrutador topológico (Graph Router): puntúa el catálogo completo contra el
   * estado latente de sesión z_t, refuerza con BM25, propaga pre-requisitos por
   * la matriz de adyacencia, resuelve mutexes y ordena topológicamente.
   */
  graphFilter(opts: {
    zt: Float32Array;
    graph: TenantToolGraph;
    edges: GraphEdge[];
    lexicalScores: Map<string, number>;
    learned: LearnedScores;
    catalog: Catalog;
    modelSize: string;
    exclude?: string[];
    queryTokens?: Set<string>;
    pinnedNames?: string[];
    topicShift?: boolean;
    turnType?: string;
  }): GraphRerankResult {
    const { zt, graph, edges, lexicalScores, learned, catalog, modelSize, exclude, queryTokens } = opts;
    // `total` es el tamaño COMPLETO del grafo: la matriz de adyacencia es
    // global (total × total) y se indexa con `toolIndexMap`.
    const allNames = [...graph.nodes.keys()];
    const total = allNames.length;
    // Exclusión de la segunda pasada: las tools ya ofrecidas no entran al
    // enrutador, así liberan cupo y dejan paso a otras del catálogo.
    const excluded = exclusionSet(exclude);
    const names = excluded ? allNames.filter((n) => !excluded.has(n.toLowerCase())) : allNames;
    // Herramientas nombradas EXPLÍCITAMENTE en las palabras clave delegadas: son
    // una orden del planificador, así que se fijan al frente y se garantiza su
    // presencia aunque el umbral adaptativo las deje fuera.
    const pinned = pinnedList(opts.pinnedNames, excluded);
    const pinnedSet = new Set(pinned.map((n) => n.toLowerCase()));
    const learnWeight = learned.weight;

    // 1. Similitud base s_i = cosine(z_t, E(T_i)) sobre embeddings de nodo.
    const baseScores: ScoredTool[] = names.map((name) => ({
      name,
      score: EmbeddingEngineService.cosine(zt, graph.nodes.get(name)!.embedding),
    }));

    // Etapa A: conserva solo las mejores categorías (grupos) del catálogo, sin
    // descartar las que contienen herramientas afines al nombre de la consulta
    // ni las nombradas explícitamente.
    const isAffine = affinityPredicate(queryTokens);
    const keptNames = keepTopCategories(
      baseScores,
      (name) => graph.nodes.get(name)?.definition.group,
      this.config.maxCategories,
      (name) => isAffine?.(name) === true || pinnedSet.has(name.toLowerCase()),
    ).map((s) => s.name);
    const baseByName = new Map(baseScores.map((s) => [s.name, s.score]));
    // Puerta de familia: namespaces nombrados explícitamente en las palabras
    // clave. Vacío = la consulta no nombra familia ninguna → no se penaliza.
    const families = namedFamilies(queryTokens, names);

    // 2. Refuerzo léxico BM25 + perfil aprendido del canal + afinidad por
    //    NOMBRE (única señal que separa familias dentro de un complemento).
    const fused = new Map<string, number>();
    for (const name of keptNames) {
      fused.set(
        name,
        (baseByName.get(name) ?? 0) +
          this.config.keywordBoost * (lexicalScores.get(name) ?? 0) +
          learnWeight * (learned.scores.get(name) ?? 0) +
          this.config.nameAffinityBoost * nameAffinity(queryTokens, name) -
          (isForeignFamily(families, name) ? this.config.familyGatePenalty : 0),
      );
    }

    // 3. Propagación de pre-requisitos: S_propagado = S + alpha · (Aᵀ · S).
    //    La matriz de adyacencia es global (total × total): se indexa con
    //    `toolIndexMap`, no con la posición dentro del pool filtrado.
    const propagated = new Map<string, number>();
    for (const jName of keptNames) {
      const j = graph.toolIndexMap.get(jName);
      let boost = 0;
      if (j !== undefined) {
        for (const iName of keptNames) {
          const i = graph.toolIndexMap.get(iName);
          if (i === undefined) continue;
          boost += graph.adjacencyMatrix[i * total + j] * (fused.get(iName) ?? 0);
        }
      }
      propagated.set(jName, (fused.get(jName) ?? 0) + graphPropagationAlphaDefault * boost);
    }

    // 4. Piso de relevancia sobre la señal fusionada (§6.6): el término
    //    aprendido del canal puede rescatar una tool que el coseno dejaba bajo
    //    el mínimo. Con `learnWeight = 0` es el máximo propagado de siempre.
    let relevance = 0;
    for (const name of keptNames) {
      const value =
        (propagated.get(name) ?? 0) + learnWeight * (learned.scores.get(name) ?? 0);
      if (value > relevance) relevance = value;
    }
    if (this.config.adaptiveMinScore > 0 && relevance < this.config.adaptiveMinScore) {
      // Las herramientas nombradas explícitamente son una orden del planificador,
      // no una predicción: se devuelven aunque la relevancia no supere el piso.
      const forced = pinnedDefinitions(pinned, (n) => catalog.get(n));
      if (forced.length === 0) {
        log(
          `[rerank:graph] piso de relevancia no superado (relevancia=${relevance.toFixed(3)} < min=${this.config.adaptiveMinScore}) → 0 tools`,
        );
        return {
          tools: [],
          complexity: "simple",
          modelSize,
          rankedScores: [relevance],
          graph: { nodes: [], edges: [], executionOrder: [] },
        };
      }
      const order = forced.map((t) => t.name);
      log(
        `[rerank:graph] piso de relevancia no superado (relevancia=${relevance.toFixed(3)} < min=${this.config.adaptiveMinScore}) ` +
          `pero ${forced.length} tools explícitas → se devuelven`,
      );
      return {
        tools: forced,
        complexity: "simple",
        modelSize,
        rankedScores: forced.map(() => relevance),
        graph: { nodes: order, edges: [], executionOrder: order },
      };
    }

    // 5. Umbral adaptativo + tope de salida (etapa B).
    const scored: ScoredTool[] = keptNames.map((name) => ({ name, score: propagated.get(name) ?? 0 }));
    const sortedScored = [...scored].sort((a, b) => b.score - a.score);
    if (sortedScored.length > 0) {
      log(
        `[rerank:graph] top=${sortedScored
          .slice(0, Math.min(5, sortedScored.length))
          .map((s) => `${s.name}(${s.score.toFixed(3)})`)
          .join(", ")}`,
      );
    }
    const selectedNames = adaptiveThreshold(
      scored,
      this.config.adaptiveMinTools,
      this.config.adaptiveMaxTools,
      this.config.adaptiveGapThreshold,
    ).slice(0, this.config.maxOutputTools);

    // 6. Resolución de mutexes: de dos nodos excluidos, queda el de mayor score.
    const selectedSet = new Set(selectedNames);
    // Las herramientas nombradas explícitamente entran siempre a la selección
    // (si están en el pool), aunque el umbral adaptativo no las haya elegido.
    const nodeNameByLower = new Map(names.map((n) => [n.toLowerCase(), n]));
    const keptLower = new Set(keptNames.map((n) => n.toLowerCase()));
    for (const name of pinned) {
      const actual = nodeNameByLower.get(name.toLowerCase());
      if (actual && keptLower.has(name.toLowerCase())) selectedSet.add(actual);
    }
    const removed: string[] = [];
    for (const e of edges) {
      if (e.type !== "MUTUALLY_EXCLUSIVE") continue;
      if (!selectedSet.has(e.from) || !selectedSet.has(e.to)) continue;
      // Una tool nombrada explícitamente nunca se descarta por un mutex: la
      // ordena el planificador, no el ranking.
      const fromPinned = pinnedSet.has(e.from.toLowerCase());
      const toPinned = pinnedSet.has(e.to.toLowerCase());
      if (fromPinned && toPinned) continue;
      if (fromPinned || toPinned) {
        const loser = fromPinned ? e.to : e.from;
        selectedSet.delete(loser);
        removed.push(loser);
        continue;
      }
      const fromScore = propagated.get(e.from) ?? 0;
      const toScore = propagated.get(e.to) ?? 0;
      const loser = fromScore >= toScore ? e.to : e.from;
      selectedSet.delete(loser);
      removed.push(loser);
    }

    // No dejar que los mutexes reduzcan por debajo del mínimo adaptativo: se
    // rellena con la siguiente mejor herramienta que no esté excluida.
    const minTools = Math.min(this.config.adaptiveMinTools, keptNames.length);
    if (selectedSet.size < minTools) {
      for (const s of sortedScored) {
        if (selectedSet.size >= minTools) break;
        if (selectedSet.has(s.name)) continue;
        const conflictsWithSelected = edges.some(
          (e) =>
            e.type === "MUTUALLY_EXCLUSIVE" &&
            ((e.from === s.name && selectedSet.has(e.to)) ||
              (e.to === s.name && selectedSet.has(e.from))),
        );
        if (conflictsWithSelected) continue;
        selectedSet.add(s.name);
      }
    }
    if (removed.length > 0) {
      log(`[rerank:graph] mutex removidos=${removed.join(",")} final=${selectedSet.size}`);
    }

    // 7. Orden topológico (Kahn) sobre las aristas PREREQUISITE del subgrafo. Las
    //    herramientas nombradas explícitamente desempatan al frente; sus
    //    pre-requisitos seleccionados siguen ejecutándose antes que ellas.
    const orderScores = new Map(propagated);
    for (const name of pinned) {
      const actual = nodeNameByLower.get(name.toLowerCase());
      if (actual && selectedSet.has(actual)) orderScores.set(actual, Number.MAX_SAFE_INTEGER);
    }
    const executionOrder = topologicalSort([...selectedSet], edges, orderScores);

    // 8. Subgrafo activado: solo aristas cuyos extremos quedaron seleccionados.
    const subEdges = edges.filter((e) => selectedSet.has(e.from) && selectedSet.has(e.to));

    const tools: ToolDefinition[] = [];
    const scores: number[] = [];
    for (const name of executionOrder) {
      const t = catalog.get(name);
      if (t) {
        tools.push(t);
        scores.push(propagated.get(name) ?? 0);
      }
    }

    const complexity = estimateComplexity(scored, tools.length);
    const orderWithScores = executionOrder
      .map((name) => `${name}(${(propagated.get(name) ?? 0).toFixed(3)})`)
      .join(">");
    log(
      `[rerank:graph] selected=${tools.length} order=${orderWithScores} complexity=${complexity}`,
    );
    // Anticipación de herramientas siguientes en el flujo de trabajo:
    // Identifica herramientas que tienen como PREREQUISITE alguna de las herramientas
    // seleccionadas en este turno pero que aún no han sido ejecutadas.
    const selectedSetNames = new Set(executionOrder);
    const nextToolCandidates: string[] = [];
    for (const e of edges) {
      if (e.type === "PREREQUISITE" && selectedSetNames.has(e.from) && !selectedSetNames.has(e.to)) {
        if (!nextToolCandidates.includes(e.to)) {
          nextToolCandidates.push(e.to);
        }
      }
    }

    const calibratedScores = CalibrationService.calibrateRankedScores(scores);
    const topTool = tools[0];
    const isConfirm = Boolean(opts.turnType && opts.turnType.includes("confirm"));
    const primaryAction = topTool?.category ?? "execute";
    const context: PredictedContext = {
      intent: {
        primaryAction,
        confidence: calibratedScores[0] ?? 0.5,
        category: topTool?.group,
        summary: tools.length > 0
          ? `Ejecución topológica de ${tools.length} herramientas (${executionOrder.join(" -> ")})`
          : "Sin herramientas seleccionadas",
      },
      constraints: {
        negations: [],
        isConfirmation: isConfirm,
        isExploratory: tools.length === 0,
      },
      dialogState: {
        phase: isConfirm ? "confirmation" : tools.length === 0 ? "discovery" : "execution",
        topicShift: opts.topicShift ?? false,
        activeDomain: topTool?.group,
      },
      anticipation: {
        suggestedNextTools: nextToolCandidates.slice(0, 5),
        reasoning:
          nextToolCandidates.length > 0
            ? `Próximos pasos inferidos del DAG tras ejecutar: ${executionOrder.join(" > ")}`
            : "Flujo completado sin dependencias posteriores inmediatas",
      },
    };

    return {
      tools,
      complexity,
      modelSize,
      rankedScores: scores,
      calibratedScores,
      context,
      graph: { nodes: executionOrder, edges: subEdges, executionOrder },
    };
  }
}

/**
 * Normaliza la lista de nombres a omitir (segunda pasada del mini-agente) a un
 * `Set` en minúsculas para comparar por nombre exacto insensible a
 * mayúsculas. Devuelve `undefined` cuando no hay nada que omitir para que el
 * llamador pueda saltarse el filtrado.
 */
export function exclusionSet(exclude?: string[]): Set<string> | undefined {
  if (!exclude || exclude.length === 0) return undefined;
  return new Set(exclude.map((n) => n.toLowerCase()));
}

/**
 * Herramientas nombradas EXPLÍCITAMENTE en las palabras clave delegadas,
 * normalizadas y ya sin las omitidas por la segunda pasada. Se preserva el orden
 * de mención de las palabras clave y se eliminan duplicados.
 */
export function pinnedList(pinnedNames?: string[], excluded?: Set<string>): string[] {
  if (!pinnedNames || pinnedNames.length === 0) return [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of pinnedNames) {
    const name = String(raw ?? "").trim();
    if (name === "") continue;
    const lower = name.toLowerCase();
    if (excluded?.has(lower) || seen.has(lower)) continue;
    seen.add(lower);
    out.push(name);
  }
  return out;
}

/**
 * Promueve las herramientas nombradas explícitamente al frente de la salida,
 * conservando el orden relativo del resto. Las que no estuvieran en `names` se
 * añaden igualmente: son una orden del planificador, no una predicción.
 */
export function promotePinned(names: string[], pinned: string[]): string[] {
  if (pinned.length === 0) return names;
  const pinnedLower = new Set(pinned.map((n) => n.toLowerCase()));
  const rest = names.filter((n) => !pinnedLower.has(n.toLowerCase()));
  return [...pinned, ...rest];
}

/**
 * Resuelve las definiciones de las herramientas fijadas por nombre exacto,
 * saltando las que no existan en el catálogo.
 */
export function pinnedDefinitions(
  pinned: string[],
  resolve: (name: string) => ToolDefinition | undefined,
): ToolDefinition[] {
  const out: ToolDefinition[] = [];
  for (const name of pinned) {
    const t = resolve(name);
    if (t) out.push(t);
  }
  return out;
}

/**
 * Predicado de "familia dorada": marca las herramientas cuyo NOMBRE casa por
 * completo con las palabras clave de la consulta. Se usa como pin de la etapa A
 * para que el recorte por `MAX_CATEGORIES` no descarte la categoría que
 * contiene la familia correcta —la única señal que distingue familias cuando el
 * grupo lo comparte todo el complemento.
 *
 * El umbral es exigente (afinidad 1: todos los tokens de identidad presentes) a
 * propósito: fijar una categoría de más anula el recorte, así que solo se fija
 * con certeza plena.
 */
export function affinityPredicate(
  queryTokens?: Set<string>,
): ((name: string) => boolean) | undefined {
  if (!queryTokens || queryTokens.size === 0) return undefined;
  return (name: string) => nameAffinity(queryTokens, name) >= 1;
}

/**
 * Etapa A del pipeline: agrupa las candidatas por categoría (`group`) y
 * conserva únicamente las de mayor score, hasta `limit`.
 *
 * La categoría se puntúa con el máximo de sus herramientas (una sola tool muy
 * relevante basta para que su categoría entre al decisor). Si el pool no supera
 * el tope de categorías, se devuelve intacto: la etapa A es una cota, no un
 * filtro agresivo.
 */
export function keepTopCategories(
  candidates: ScoredTool[],
  groupOf: (name: string) => string | undefined,
  limit: number,
  pinned?: (name: string) => boolean,
): ScoredTool[] {
  if (candidates.length === 0) return candidates;
  const categoryScore = new Map<string, number>();
  const pinnedGroups = new Set<string>();
  for (const c of candidates) {
    const g = groupOf(c.name) || "";
    const cur = categoryScore.get(g);
    if (cur === undefined || c.score > cur) categoryScore.set(g, c.score);
    if (pinned?.(c.name)) pinnedGroups.add(g);
  }
  if (limit <= 0 || categoryScore.size <= limit) return candidates;

  const kept = new Set(
    [...categoryScore.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([g]) => g),
  );
  // La familia dorada nunca se recorta: el grupo compartido no discrimina y el
  // coseno puede dejarla fuera del top de categorías.
  for (const g of pinnedGroups) kept.add(g);
  const filtered = candidates.filter((c) => kept.has(groupOf(c.name) || ""));
  log(
    `[rerank:categorias] etapaA categorias=${categoryScore.size}->${kept.size} ` +
      `candidatas=${candidates.length}->${filtered.length} fijadas=${pinnedGroups.size}`,
  );
  return filtered;
}

/**
 * Orden topológico por Kahn sobre las aristas PREREQUISITE del subgrafo.
 * Los pre-requisitos (origen) se ejecutan antes que sus dependientes. Ante
 * empates o ciclos se desempata por mayor score propagado.
 */
export function topologicalSort(
  selected: string[],
  edges: GraphEdge[],
  scoreByName: Map<string, number>,
): string[] {
  const set = new Set(selected);
  const adj = new Map<string, string[]>();
  const indegree = new Map<string, number>();
  for (const name of selected) {
    adj.set(name, []);
    indegree.set(name, 0);
  }

  for (const e of edges) {
    if (e.type !== "PREREQUISITE") continue;
    if (!set.has(e.from) || !set.has(e.to)) continue;
    adj.get(e.from)!.push(e.to);
    indegree.set(e.to, (indegree.get(e.to) ?? 0) + 1);
  }

  const order: string[] = [];
  const queue = selected.filter((n) => (indegree.get(n) ?? 0) === 0);
  while (queue.length > 0) {
    queue.sort((a, b) => (scoreByName.get(b) ?? 0) - (scoreByName.get(a) ?? 0));
    const node = queue.shift()!;
    order.push(node);
    for (const next of adj.get(node) ?? []) {
      indegree.set(next, (indegree.get(next) ?? 0) - 1);
      if (indegree.get(next) === 0) queue.push(next);
    }
  }

  if (order.length < selected.length) {
    const missing = selected
      .filter((n) => !order.includes(n))
      .sort((a, b) => (scoreByName.get(b) ?? 0) - (scoreByName.get(a) ?? 0));
    order.push(...missing);
  }
  return order;
}

/** Replica AdaptiveThreshold del Go (mínimo real + corte por gap natural). */
export function adaptiveThreshold(
  scored: ScoredTool[],
  minTools: number,
  maxTools: number,
  gapThreshold: number,
): string[] {
  if (scored.length === 0) return [];
  const sorted = [...scored].sort((a, b) => b.score - a.score);

  const min = Math.min(minTools, sorted.length);
  const max = Math.max(min, maxTools);
  const top = sorted[0].score;
  // Banda de relevancia: cuánto puede caer un score respecto del top y seguir
  // contando como "de la misma tanda". Se deriva del umbral de gap (×2) para
  // que sea proporcional a la escala que el operador ya configuró.
  const band = gapThreshold * 2;

  // Busca un gap natural después del mínimo. Un gap solo corta si además deja
  // fuera herramientas claramente peores (`top - score > band`): si la cola
  // sigue dentro de la banda del top, el ranking es denso y el corte es un
  // artefacto del refuerzo léxico — típico cuando un grupo entero (p. ej. las
  // tools de un complemento) comparte firma semántica y solo BM25 desempata.
  for (let i = min; i < Math.min(sorted.length, max); i++) {
    const gap = sorted[i - 1].score - sorted[i].score;
    if (gap > gapThreshold && top - sorted[i].score > band) {
      return sorted.slice(0, i).map((t) => t.name);
    }
  }

  // Sin gap natural → el ranking es denso (muchas tools igualmente relevantes,
  // típico de prompts complejos multi-paso). En lugar de colapsar al mínimo, se
  // devuelve el cluster superior (tools dentro de la banda del top), acotado al
  // máximo. Así un prompt complejo ya no queda reducido a 1-2 tools.
  let cluster = sorted.length;
  for (let i = 0; i < sorted.length; i++) {
    if (top - sorted[i].score > band) {
      cluster = i;
      break;
    }
  }
  const count = Math.max(min, Math.min(cluster, max));
  return sorted.slice(0, count).map((t) => t.name);
}

/** Replica EstimateComplexity del Go. */
export function estimateComplexity(scored: ScoredTool[], selectedCount: number): ToolComplexity {
  if (scored.length === 0) return "simple";
  const avg = scored.reduce((s, t) => s + t.score, 0) / scored.length;
  if (selectedCount > COMPLEXITY_COUNT_HIGH && avg > COMPLEXITY_AVG_HIGH) return "complex";
  if (selectedCount > COMPLEXITY_COUNT_MID && avg > COMPLEXITY_AVG_MID) return "moderate";
  return "simple";
}
