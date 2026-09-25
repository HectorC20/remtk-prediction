/**
 * Especificidad (juicio §6): penalización por difusión semántica + indexación
 * dual capacidad/problema.
 *
 * A. Difusión d(t): coseno medio del embedding de la herramienta contra el
 *    resto del catálogo del scope. Herramientas generalistas (web_search)
 *    colisionan con muchos temas → d alto → se exige margen más estricto:
 *
 *      score' = score / (1 + ρ·max(0, d(t) − d̄))
 *
 *    Se calcula una vez por scope a partir de los nodos del grafo (sin
 *    inferencia extra) y se invalida al re-registrar el catálogo.
 *
 * B. Indexación dual: si la herramienta declara `problemSpace` (cómo un
 *    usuario expresaría la necesidad), se embebe y puntúa en paralelo:
 *    el score final toma el máximo entre capacidad y necesidad.
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import type { AppConfig } from "src/config";
import type { ToolGraphCacheService } from "src/predict/services/graph-cache.service";
import type { ToolDefinition } from "src/shared/interfaces/domain.interface";

export class EspecificidadService {
  /** scopeKey → (nombre → difusión) + media del scope. */
  private readonly diffusion = new Map<string, { byTool: Map<string, number>; mean: number }>();
  /** scopeKey → toolName → embedding del problemSpace. */
  private readonly problemEmbs = new Map<string, Map<string, Float32Array>>();

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly graphCache: ToolGraphCacheService,
    private readonly config: AppConfig,
  ) {}

  /** Invalida los índices del scope (POST /tools). */
  clearScope(scopeKey: string): void {
    this.diffusion.delete(scopeKey);
    this.problemEmbs.delete(scopeKey);
  }

  /** Devuelve las definiciones de herramientas del grafo del scope. */
  catalogTools(scopeKey: string): ToolDefinition[] {
    const graph = this.graphCache.get(scopeKey);
    if (!graph) return [];
    return [...graph.nodes.values()].map((n) => n.definition);
  }

  /**
   * Factor multiplicativo de penalización por difusión para una herramienta
   * (1 = sin penalización; <1 = herramienta generalista castigada).
   */
  penalty(scopeKey: string, toolName: string): number {
    const idx = this.ensureDiffusion(scopeKey);
    if (!idx) return 1;
    const d = idx.byTool.get(toolName) ?? idx.mean;
    return 1 / (1 + this.config.juicioSpecificityRho * Math.max(0, d - idx.mean));
  }

  /** Similitud del espacio de necesidad (0 si la tool no declara problemSpace). */
  async problemScore(
    scopeKey: string,
    tool: ToolDefinition,
    uText: Float32Array | undefined,
  ): Promise<number> {
    if (!tool.problemSpace || !uText || !this.engine.isReady("small")) return 0;
    let scope = this.problemEmbs.get(scopeKey);
    if (!scope) {
      scope = new Map();
      this.problemEmbs.set(scopeKey, scope);
    }
    let emb = scope.get(tool.name);
    if (!emb) {
      const res = await this.engine.embedPassage(tool.problemSpace, "small", { high: true });
      if (res.model === "hash") return 0;
      emb = res.embedding;
      scope.set(tool.name, emb);
    }
    return EmbeddingEngineService.cosine(uText, emb);
  }

  /**
   * Proyecta un vector de consulta sobre la variedad (manifold) de herramientas
   * del scope (capacidad + problemSpace ponderados por difusión). Devuelve la
   * herramienta líder, su prominencia sobre la media del catálogo y el mapa de
   * activaciones.
   */
  async projectManifold(
    scopeKey: string,
    vec: Float32Array | undefined,
  ): Promise<
    | {
        topTool: string;
        topScore: number;
        secondScore: number;
        meanScore: number;
        prominence: number;
        scores: Map<string, number>;
      }
    | undefined
  > {
    if (!vec || !this.engine.isReady("small")) return undefined;
    const graph = this.graphCache.get(scopeKey);
    if (!graph || graph.nodes.size < 2) return undefined;

    const scores = new Map<string, number>();
    let topTool = "";
    let topScore = -Infinity;
    let secondScore = -Infinity;
    let sum = 0;

    for (const [name, node] of graph.nodes.entries()) {
      const cap = EmbeddingEngineService.cosine(vec, node.embedding);
      const prob = await this.problemScore(scopeKey, node.definition, vec);
      const base = prob > cap + this.config.juicioProblemMargin ? prob : cap;
      const s = base * this.penalty(scopeKey, name);
      scores.set(name, s);
      sum += s;
      if (s > topScore) {
        secondScore = topScore;
        topScore = s;
        topTool = name;
      } else if (s > secondScore) {
        secondScore = s;
      }
    }

    const meanScore = sum / graph.nodes.size;
    return {
      topTool,
      topScore,
      secondScore: Number.isFinite(secondScore) ? secondScore : topScore,
      meanScore,
      prominence: topScore - meanScore,
      scores,
    };
  }

  /**
   * Evalúa sub-ventanas deslizantes sobre una consulta extensa o multi-cláusula
   * y devuelve las herramientas que lideran con prominencia cada sub-ventana.
   */
  async subWindowPeaks(
    scopeKey: string,
    text: string,
  ): Promise<Map<string, { tool: ToolDefinition; score: number; windowIdx: number }>> {
    const out = new Map<string, { tool: ToolDefinition; score: number; windowIdx: number }>();
    if (!this.engine.isReady("small")) return out;
    const graph = this.graphCache.get(scopeKey);
    if (!graph || graph.nodes.size < 2) return out;

    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length < 7) return out;

    const windows: string[] = [];
    const seenWin = new Set<string>();
    const addWin = (w: string): void => {
      const clean = w.trim();
      const key = clean.toLowerCase();
      if (clean.length >= 10 && !seenWin.has(key)) {
        seenWin.add(key);
        windows.push(clean);
      }
    };

    // Sub-tramos por puntuación estructural de cláusula y ventanas deslizantes (W=3, W=4, W=5, W=6).
    for (const clause of text.split(/[,;:\n…]+/)) {
      const cw = clause.trim().split(/\s+/).filter(Boolean);
      if (cw.length >= 3 && cw.length <= 10) addWin(clause);
    }
    for (let i = 0; i + 3 <= words.length; i += 2) {
      addWin(words.slice(i, i + 3).join(" "));
    }
    for (let i = 0; i + 4 <= words.length; i += 2) {
      addWin(words.slice(i, i + 4).join(" "));
    }
    addWin(words.slice(Math.max(0, words.length - 4)).join(" "));
    for (let i = 0; i + 5 <= words.length; i += 2) {
      addWin(words.slice(i, i + 5).join(" "));
    }
    addWin(words.slice(Math.max(0, words.length - 5)).join(" "));
    for (let i = 0; i + 6 <= words.length; i += 3) {
      addWin(words.slice(i, i + 6).join(" "));
    }
    addWin(words.slice(Math.max(0, words.length - 6)).join(" "));

    const windowProjs: {
      r: number;
      meanScore: number;
      scores: Map<string, number>;
    }[] = [];
    const toolSum = new Map<string, number>();

    for (let r = 0; r < windows.length; r++) {
      const emb = await this.engine.embedQuery(windows[r], "small", { high: true });
      if (emb.model === "hash") continue;
      const proj = await this.projectManifold(scopeKey, emb.embedding);
      if (!proj) continue;
      windowProjs.push({ r, meanScore: proj.meanScore, scores: proj.scores });
      for (const [name, s] of proj.scores.entries()) {
        toolSum.set(name, (toolSum.get(name) ?? 0) + s);
      }
    }
    if (windowProjs.length < 2) return out;

    const R = windowProjs.length;
    const toolMean = new Map<string, number>();
    for (const [name, sum] of toolSum.entries()) {
      toolMean.set(name, sum / R);
    }

    for (const wp of windowProjs) {
      let bestTool = "";
      let bestInteraction = -Infinity;
      let bestRaw = 0;

      for (const [name, s] of wp.scores.entries()) {
        const meanT = toolMean.get(name) ?? s;
        const spatial = s - wp.meanScore;
        const temporal = s - meanT;
        if (s < 0.810 || spatial < 0.014 || temporal < 0.014 || spatial + temporal < 0.040) {
          continue;
        }
        // Interacción doblemente centrada: prioriza el pulso local de la herramienta
        // en esta ventana descontando su línea base global en el resto de la oración.
        const interaction = temporal + 0.5 * spatial;
        if (interaction > bestInteraction) {
          bestInteraction = interaction;
          bestTool = name;
          bestRaw = s;
        }
      }

      if (!bestTool) continue;
      const node = graph.nodes.get(bestTool);
      if (!node) continue;
      const prev = out.get(bestTool);
      if (!prev || bestRaw > prev.score) {
        out.set(bestTool, {
          tool: node.definition,
          score: bestRaw,
          windowIdx: prev ? prev.windowIdx : wp.r,
        });
      }
    }
    return out;
  }

  /** Difusión por scope desde los nodos del grafo (perezoso, una vez por scope). */
  private ensureDiffusion(scopeKey: string): { byTool: Map<string, number>; mean: number } | undefined {
    const cached = this.diffusion.get(scopeKey);
    if (cached) return cached;
    const graph = this.graphCache.get(scopeKey);
    if (!graph || graph.nodes.size < 2) return undefined;

    const names = [...graph.nodes.keys()];
    const byTool = new Map<string, number>();
    let total = 0;
    for (const a of names) {
      const ea = graph.nodes.get(a)!.embedding;
      let acc = 0;
      for (const b of names) {
        if (a === b) continue;
        acc += EmbeddingEngineService.cosine(ea, graph.nodes.get(b)!.embedding);
      }
      const d = acc / (names.length - 1);
      byTool.set(a, d);
      total += d;
    }
    const idx = { byTool, mean: total / names.length };
    this.diffusion.set(scopeKey, idx);
    return idx;
  }
}
