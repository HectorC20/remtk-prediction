/**
 * RerankService: Capa 3 del pipeline. Fusiona el score cross-idioma de keywords
 * (capa 2) con la confirmación léxica de Qdrant (BM25) y aplica el umbral
 * adaptativo. El resultado final se recorta al tope configurado de herramientas
 * de salida (env opcional `MAX_OUTPUT_TOOLS`, 0-50).
 * Migrado de rerank_service.go + adaptive_threshold.go del server Go.
 */
import { EmbeddingEngineService } from "../../embedding/embedding-engine";
import type { AppConfig } from "../../config";
import { log } from "../../logger";
import type { ScoredTool, ToolComplexity, ToolDefinition } from "../../shared/interfaces/domain.interface";
import type { GraphEdge, TenantToolGraph } from "../../shared/interfaces/graph.interface";
import { graphPropagationAlphaDefault } from "../../shared/constants/predict/general.predict";

export interface RerankResult {
  tools: ToolDefinition[];
  complexity: ToolComplexity;
  modelSize: string;
  rankedScores: number[];
}

export interface GraphRerankResult extends RerankResult {
  graph: {
    nodes: string[];
    edges: GraphEdge[];
    executionOrder: string[];
  };
}

/** Catálogo mínimo para resolver definiciones por nombre (evita acoplar Qdrant). */
type Catalog = { get(name: string): ToolDefinition | undefined };

export class RerankService {
  constructor(private readonly config: AppConfig) {}

  /**
   * Fusiona el ranking cross-idioma (reduced) con el score léxico normalizado
   * de Qdrant y aplica el piso de relevancia + umbral adaptativo.
   *
   * final = cosineKeywords + keywordBoost * lexicalQdrant
   */
  filter(
    reduced: ScoredTool[],
    lexicalScores: Map<string, number>,
    catalog: { get(name: string): ToolDefinition | undefined },
    modelSize: string,
  ): RerankResult {
    const fused: ScoredTool[] = reduced.map((r) => ({
      name: r.name,
      score: r.score + this.config.keywordBoost * (lexicalScores.get(r.name) ?? 0),
    }));
    fused.sort((a, b) => b.score - a.score);

    const topCos = reduced.reduce((m, r) => Math.max(m, r.score), 0);
    if (fused.length > 0) {
      log(
        `[rerank] fused=${fused.length} top1=${fused[0].name}(${fused[0].score.toFixed(3)}) ` +
          `topCos=${topCos.toFixed(3)}`,
      );
    }

    // Piso absoluto de relevancia (semántico): si ni la mejor keyword coseno
    // supera el mínimo, no se selecciona ninguna herramienta.
    if (this.config.adaptiveMinScore > 0 && topCos < this.config.adaptiveMinScore) {
      log(
        `[rerank] piso de relevancia no superado (topCos=${topCos.toFixed(3)} < min=${this.config.adaptiveMinScore}) → 0 tools`,
      );
      return { tools: [], complexity: "simple", modelSize, rankedScores: [topCos] };
    }

    const selectedNames = adaptiveThreshold(
      fused,
      this.config.adaptiveMinTools,
      this.config.adaptiveMaxTools,
      this.config.adaptiveGapThreshold,
    );

    // Tope final opcional (MAX_OUTPUT_TOOLS, 0-50): recorta lo decidido por el
    // umbral adaptativo, p. ej. 0 para no devolver herramientas nunca.
    const cappedNames = selectedNames.slice(0, this.config.maxOutputTools);

    const byName = new Map<string, ToolDefinition>();
    for (const r of reduced) {
      const t = catalog.get(r.name);
      if (t) byName.set(t.name, t);
    }
    const scoreByName = new Map(fused.map((s) => [s.name, s.score]));

    const tools: ToolDefinition[] = [];
    const scores: number[] = [];
    for (const name of cappedNames) {
      const t = byName.get(name);
      if (t) {
        tools.push(t);
        scores.push(scoreByName.get(name) ?? 0);
      }
    }

    const complexity = estimateComplexity(fused, tools.length);
    log(`[rerank] selected=${tools.length} complexity=${complexity}`);
    return { tools, complexity, modelSize, rankedScores: scores };
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
    catalog: Catalog;
    modelSize: string;
  }): GraphRerankResult {
    const { zt, graph, edges, lexicalScores, catalog, modelSize } = opts;
    const names = [...graph.nodes.keys()];

    // 1. Similitud base s_i = cosine(z_t, E(T_i)) sobre embeddings de nodo.
    const fused = new Map<string, number>();
    for (const name of names) {
      const node = graph.nodes.get(name)!;
      const base = EmbeddingEngineService.cosine(zt, node.embedding);
      fused.set(name, base + this.config.keywordBoost * (lexicalScores.get(name) ?? 0));
    }

    // 2. Propagación de pre-requisitos: S_propagado = S + alpha · (Aᵀ · S).
    const propagated = new Map<string, number>();
    const n = names.length;
    for (let j = 0; j < n; j++) {
      let boost = 0;
      for (let i = 0; i < n; i++) {
        boost += graph.adjacencyMatrix[i * n + j] * (fused.get(names[i]) ?? 0);
      }
      propagated.set(names[j], (fused.get(names[j]) ?? 0) + graphPropagationAlphaDefault * boost);
    }

    // 3. Piso de relevancia.
    const topScore = names.length > 0 ? Math.max(...propagated.values()) : 0;
    if (this.config.adaptiveMinScore > 0 && topScore < this.config.adaptiveMinScore) {
      log(
        `[rerank:graph] piso de relevancia no superado (top=${topScore.toFixed(3)} < min=${this.config.adaptiveMinScore}) → 0 tools`,
      );
      return {
        tools: [],
        complexity: "simple",
        modelSize,
        rankedScores: [topScore],
        graph: { nodes: [], edges: [], executionOrder: [] },
      };
    }

    // 4. Umbral adaptativo + tope de salida.
    const scored: ScoredTool[] = names.map((name) => ({ name, score: propagated.get(name) ?? 0 }));
    const selectedNames = adaptiveThreshold(
      scored,
      this.config.adaptiveMinTools,
      this.config.adaptiveMaxTools,
      this.config.adaptiveGapThreshold,
    ).slice(0, this.config.maxOutputTools);

    // 5. Resolución de mutexes: de dos nodos excluidos, queda el de mayor score.
    const selectedSet = new Set(selectedNames);
    for (const e of edges) {
      if (e.type !== "MUTUALLY_EXCLUSIVE") continue;
      if (selectedSet.has(e.from) && selectedSet.has(e.to)) {
        const fromScore = propagated.get(e.from) ?? 0;
        const toScore = propagated.get(e.to) ?? 0;
        selectedSet.delete(fromScore >= toScore ? e.to : e.from);
      }
    }

    // 6. Orden topológico (Kahn) sobre las aristas PREREQUISITE del subgrafo.
    const executionOrder = topologicalSort([...selectedSet], edges, propagated);

    // 7. Subgrafo activado: solo aristas cuyos extremos quedaron seleccionados.
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
    log(
      `[rerank:graph] selected=${tools.length} order=${executionOrder.join(">")} complexity=${complexity}`,
    );
    return {
      tools,
      complexity,
      modelSize,
      rankedScores: scores,
      graph: { nodes: executionOrder, edges: subEdges, executionOrder },
    };
  }
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

  // Busca un gap natural después del mínimo.
  for (let i = min; i < Math.min(sorted.length, max); i++) {
    const gap = sorted[i - 1].score - sorted[i].score;
    if (gap > gapThreshold) {
      return sorted.slice(0, i).map((t) => t.name);
    }
  }

  // Sin gap natural → devolver el mínimo.
  return sorted.slice(0, min).map((t) => t.name);
}

/** Replica EstimateComplexity del Go. */
export function estimateComplexity(scored: ScoredTool[], selectedCount: number): ToolComplexity {
  if (scored.length === 0) return "simple";
  const avg = scored.reduce((s, t) => s + t.score, 0) / scored.length;
  if (selectedCount > 20 && avg > 0.65) return "complex";
  if (selectedCount > 10 && avg > 0.55) return "moderate";
  return "simple";
}
