/**
 * MaxSim de interacción tardía (juicio §5, estilo ColBERT): multi-intención
 * sin segmentar por conjunciones.
 *
 * En vez de colapsar la consulta en un solo vector (mean pooling), se conservan
 * los embeddings por token {e_1..e_m} y cada herramienta puntúa por alineación
 * máxima token a token:
 *
 *   Score(tool) = (1/|J|) Σ_j max_i (e_i · t_j)
 *
 * Dos intenciones en un turno activan grupos de tokens distintos sin interferir:
 * "descarga el reporte… y envíamelo por correo" puntúa alto tanto contra
 * download_report como contra send_email.
 *
 * Los vectores de herramienta se cachean por scope+nombre (se invalidan al
 * re-registrar el catálogo).
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import type { AppConfig } from "src/config";
import type { ToolDefinition } from "src/shared/interfaces/domain.interface";

export class MaxSimService {
  /** scopeKey → toolName → vectores por token de la firma de la herramienta. */
  private readonly toolTokens = new Map<string, Map<string, Float32Array[]>>();
  private lastQuery?: { text: string; tokens: Float32Array[] };

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly config: AppConfig,
  ) {}

  /** Invalida los vectores cacheados del scope (POST /tools). */
  clearScope(scopeKey: string): void {
    this.toolTokens.delete(scopeKey);
  }

  private async getQueryTokens(text: string): Promise<Float32Array[]> {
    if (this.lastQuery && this.lastQuery.text === text) {
      return this.lastQuery.tokens;
    }
    const tokens = await this.engine.embedTokens(`query: ${text}`, "small", { high: true });
    if (tokens.length > 0) {
      this.lastQuery = { text, tokens };
    }
    return tokens;
  }

  private async ensureToolTokens(
    scopeKey: string,
    tools: ToolDefinition[],
  ): Promise<Map<string, Float32Array[]>> {
    let scope = this.toolTokens.get(scopeKey);
    if (!scope) {
      scope = new Map();
      this.toolTokens.set(scopeKey, scope);
    }
    for (const tool of tools) {
      let vecs = scope.get(tool.name);
      if (!vecs) {
        vecs = (
          await this.engine.embedTokens(`passage: ${signature(tool)}`, "small", { high: true })
        ).slice(0, this.config.juicioMaxsimMaxTokens);
        scope.set(tool.name, vecs);
      }
    }
    return scope;
  }

  /**
   * Scores MaxSim del texto actual contra las herramientas dadas.
   * Devuelve mapa nombre → score (0..1); vacío si no hay nivel token (hash).
   */
  async score(
    scopeKey: string,
    text: string,
    tools: ToolDefinition[],
  ): Promise<Map<string, number>> {
    const out = new Map<string, number>();
    if (tools.length === 0 || !this.engine.isReady("small")) return out;

    const queryTokens = await this.getQueryTokens(text);
    if (queryTokens.length === 0) return out;

    const scope = await this.ensureToolTokens(scopeKey, tools);
    for (const tool of tools) {
      const vecs = scope.get(tool.name) ?? [];
      out.set(tool.name, maxsim(queryTokens, vecs));
    }
    return out;
  }

  /**
   * Detecta herramientas ancla a nivel token mediante doble centrado (interacción
   * fila-columna) sobre la matriz ColBERT M_{i,j} = max_k (q_i · t_{j,k}).
   * Permite rescatar intenciones compactas (p. ej. complemento de idioma) cuyo
   * sustantivo objeto es compartido anafóricamente con otra cláusula.
   */
  async anchorTools(
    scopeKey: string,
    text: string,
    tools: ToolDefinition[],
  ): Promise<Map<string, { tool: ToolDefinition; score: number; tokenIdx: number }>> {
    const out = new Map<string, { tool: ToolDefinition; score: number; tokenIdx: number }>();
    if (tools.length < 2 || !this.engine.isReady("small")) return out;

    const queryTokens = await this.getQueryTokens(text);
    if (queryTokens.length < 4) return out;

    const scope = await this.ensureToolTokens(scopeKey, tools);
    const m = queryTokens.length;
    const n = tools.length;

    // Matriz M[i][j] = max_k cosine(q_i, t_{j,k})
    const mat: number[][] = [];
    const rowMeans = new Float64Array(m);
    const colSums = new Float64Array(n);

    for (let i = 0; i < m; i++) {
      const row: number[] = new Array(n);
      let rSum = 0;
      const q = queryTokens[i];
      for (let j = 0; j < n; j++) {
        const tVecs = scope.get(tools[j].name) ?? [];
        let best = 0;
        for (const tv of tVecs) {
          const s = EmbeddingEngineService.cosine(q, tv);
          if (s > best) best = s;
        }
        row[j] = best;
        rSum += best;
        colSums[j] += best;
      }
      mat.push(row);
      rowMeans[i] = rSum / n;
    }

    const colMeans = new Float64Array(n);
    let grandSum = 0;
    for (let j = 0; j < n; j++) {
      colMeans[j] = colSums[j] / m;
      grandSum += colMeans[j];
    }
    const grandMean = grandSum / n;

    for (let i = 0; i < m; i++) {
      let bestJ = -1;
      let bestI = -Infinity;
      let secondI = -Infinity;
      const rMean = rowMeans[i];

      for (let j = 0; j < n; j++) {
        const interaction = mat[i][j] - rMean - colMeans[j] + grandMean;
        if (interaction > bestI) {
          secondI = bestI;
          bestI = interaction;
          bestJ = j;
        } else if (interaction > secondI) {
          secondI = interaction;
        }
      }

      if (bestJ < 0) continue;
      const raw = mat[i][bestJ];
      const margin = bestI - secondI;
      if (bestI >= 0.030 && margin >= 0.025 && raw >= 0.770) {
        const tool = tools[bestJ];
        const prev = out.get(tool.name);
        if (!prev || raw > prev.score) {
          out.set(tool.name, {
            tool,
            score: raw,
            tokenIdx: prev ? prev.tokenIdx : i,
          });
        }
      }
    }

    return out;
  }
}

/** (1/|J|) Σ_j max_i (q_i · t_j): ambos lados normalizados L2 → coseno. */
function maxsim(queryTokens: Float32Array[], toolTokens: Float32Array[]): number {
  if (queryTokens.length === 0 || toolTokens.length === 0) return 0;
  let acc = 0;
  for (const t of toolTokens) {
    let best = 0;
    for (const q of queryTokens) {
      const s = EmbeddingEngineService.cosine(q, t);
      if (s > best) best = s;
    }
    acc += best;
  }
  return acc / toolTokens.length;
}

/** Firma compacta de la herramienta para el MaxSim (sin inputSchema). */
function signature(tool: ToolDefinition): string {
  return [tool.name, tool.intentSummary, tool.description, (tool.tags ?? []).join(" ")]
    .filter(Boolean)
    .join(" ");
}
