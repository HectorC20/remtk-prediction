/**
 * RerankService: Capa 3 del pipeline. Fusiona el score cross-idioma de keywords
 * (capa 2) con la confirmación léxica de Qdrant (BM25) y aplica el umbral
 * adaptativo para devolver entre 0 y 10 herramientas.
 * Migrado de rerank_service.go + adaptive_threshold.go del server Go.
 */
import type { AppConfig } from "../config";
import { log } from "../logger";
import type { ScoredTool, ToolComplexity, ToolDefinition } from "../types";

export interface RerankResult {
  tools: ToolDefinition[];
  complexity: ToolComplexity;
  modelSize: string;
  rankedScores: number[];
}

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

    const byName = new Map<string, ToolDefinition>();
    for (const r of reduced) {
      const t = catalog.get(r.name);
      if (t) byName.set(t.name, t);
    }
    const scoreByName = new Map(fused.map((s) => [s.name, s.score]));

    const tools: ToolDefinition[] = [];
    const scores: number[] = [];
    for (const name of selectedNames) {
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
