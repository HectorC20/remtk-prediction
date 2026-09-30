/**
 * ToolGraphCacheService: topología en memoria de las herramientas por scope
 * (scopeKey; el chat general usa scopeKey = tenant).
 *
 * En el ciclo de POST /tools reconstruye el grafo G = (V, E) del scope:
 *   - Cada nodo embebe la firma de su herramienta con prefijo `passage:` (e5-small).
 *   - Las aristas se infieren de las relaciones declaradas (`prerequisites`,
 *     `conflicts`) y del esquema `inputSchema`, más la co-ocurrencia por `group`.
 *   - La matriz de adyacencia A (N×N) guarda el peso de la arista i → j
 *     (i es pre-requisito de j) para la propagación del Graph Router.
 */
import { createHash } from "node:crypto";
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { log } from "src/logger";
import type { ToolDefinition } from "src/shared/interfaces/domain.interface";
import type {
  GraphEdge,
  TenantToolGraph,
  ToolGraphNode,
} from "src/shared/interfaces/graph.interface";
import {
  CO_OCCURRENCE_WEIGHT,
  edgeInferenceMinSimByModel,
  MUTUALLY_EXCLUSIVE_WEIGHT,
  PREREQUISITE_WEIGHT,
} from "src/shared/constants/predict";

export class ToolGraphCacheService {
  /** Grafos por scopeKey (chat general: scopeKey = tenant). */
  private readonly graphs = new Map<string, TenantToolGraph>();
  private readonly edgesByScope = new Map<string, GraphEdge[]>();

  constructor(private readonly engine: EmbeddingEngineService) {}

  get(scopeKey: string): TenantToolGraph | undefined {
    return this.graphs.get(scopeKey);
  }

  edges(scopeKey: string): GraphEdge[] {
    return this.edgesByScope.get(scopeKey) ?? [];
  }

  /** POST /tools: reconstruye el grafo del scope (embeddings passage: + aristas). */
  async buildTenantGraph(scopeKey: string, tools: ToolDefinition[]): Promise<TenantToolGraph> {
    const versionHash = computeVersionHash(tools);

    // Nodos: embedding passage: de la firma de cada herramienta (prioridad baja,
    // es recompute masivo de registro, no de predicción).
    const nodes = new Map<string, ToolGraphNode>();
    for (const tool of tools) {
      const res = await this.engine.embedPassage(toolSignature(tool), this.engine.defaultSize, { high: false });
      nodes.set(tool.name, {
        id: tool.id,
        name: tool.name,
        embedding: res.embedding,
        prerequisites: tool.prerequisites ?? [],
        conflicts: tool.conflicts ?? [],
        definition: tool,
      });
    }

    const toolIndexMap = new Map<string, number>();
    [...nodes.keys()].forEach((name, i) => toolIndexMap.set(name, i));
    const n = nodes.size;
    const adjacencyMatrix = new Float32Array(n * n);
    const edges = await this.inferEdges(tools, nodes, toolIndexMap, adjacencyMatrix);

    const graph: TenantToolGraph = { tenant: scopeKey, versionHash, nodes, adjacencyMatrix, toolIndexMap };
    this.graphs.set(scopeKey, graph);
    this.edgesByScope.set(scopeKey, edges);
    log(
      `[graph] scope=${scopeKey} nodes=${n} edges=${edges.length} hash=${versionHash.slice(0, 8)}`,
    );
    return graph;
  }

  /**
   * Inferencia latente vector del esquema a herramientas proveedoras (e5-small):
   * Embebe cada argumento del esquema (`categoryId`, `zoneId`, etc.) y calcula la similitud
   * del coseno con el nodo de cada herramienta en el espacio vectorial.
   * Cero palabras clave, diccionarios de verbos o substring matching (AGENT.md).
   */
  private async inferEdges(
    tools: ToolDefinition[],
    nodes: Map<string, ToolGraphNode>,
    toolIndexMap: Map<string, number>,
    adjacencyMatrix: Float32Array,
  ): Promise<GraphEdge[]> {
    const edges: GraphEdge[] = [];
    const byName = new Map(tools.map((t) => [t.name, t]));
    const n = tools.length;
    const paramEmbCache = new Map<string, Float32Array>();
    const intentEmbCache = new Map<string, Float32Array>();

    // Pre-embeber el propósito / resumen de intención de cada herramienta
    for (const t of tools) {
      const sig = `${t.name}: ${t.intentSummary || t.name}`;
      const res = await this.engine.embedPassage(sig, this.engine.defaultSize, { high: false });
      intentEmbCache.set(t.name, res.embedding);
    }

    for (const tool of tools) {
      const prereqs = new Set<string>(tool.prerequisites ?? []);
      const schemaObj = (tool.inputSchema?.properties ?? tool.inputSchema ?? {}) as Record<string, unknown>;
      const rawKeys = Object.keys(schemaObj).filter(
        (k) => !["type", "properties", "required", "$schema", "additionalProperties"].includes(k),
      );

      for (const rawKey of rawKeys) {
        if (rawKey.length < 2) continue;

        // Parámetros que representan referencias a entidades (e.g. categoryId, zoneId, *_id, o id propio en mutación)
        const isSelfId = rawKey.toLowerCase() === "id";
        const isRefKey = isSelfId || rawKey.endsWith("Id") || rawKey.endsWith("_id") || rawKey.endsWith("ID");
        if (!isRefKey) continue;

        let probe: string;
        if (isSelfId) {
          const parts = tool.name.split("_");
          const entity = parts.length > 2 ? parts[parts.length - 2] : tool.name;
          probe = `obtener ${entity}`;
        } else {
          const cleaned = rawKey.replace(/([A-Z])/g, " $1").replace(/_?id$/i, "").trim();
          probe = `obtener o listar ${cleaned}`;
        }

        let pEmb = paramEmbCache.get(probe);
        if (!pEmb) {
          const res = await this.engine.embedQuery(probe, this.engine.defaultSize, { high: false });
          pEmb = res.embedding;
          paramEmbCache.set(probe, pEmb);
        }

        // Selecciona la mejor herramienta proveedora/descubridora de la entidad.
        // El umbral es relativo a la escala del coseno del modelo activo.
        let bestTool: string | undefined;
        let bestSim = edgeInferenceMinSimByModel[this.engine.defaultSize];
        for (const other of tools) {
          if (other.name === tool.name) continue;
          if (!tool.group || !other.group || tool.group !== other.group) continue;

          // Una herramienta antecedente no debe ser una mutación destructiva o creación
          if (
            other.name.endsWith("_eliminar") ||
            other.name.endsWith("_actualizar") ||
            other.name.endsWith("_crear")
          ) {
            continue;
          }

          // Para referencias externas desconocidas (ej. categoryId), debe listar sin requerir id previo
          if (!isSelfId) {
            const otherReq = (other.inputSchema?.required ?? []) as string[];
            if (otherReq.includes("id") || other.name.endsWith("_obtener")) {
              continue;
            }
          }

          const oEmb = intentEmbCache.get(other.name);
          if (!oEmb) continue;

          const sim = EmbeddingEngineService.cosine(pEmb, oEmb);
          if (sim > bestSim) {
            bestSim = sim;
            bestTool = other.name;
          }
        }
        if (bestTool) {
          prereqs.add(bestTool);
        }
      }

      for (const preName of prereqs) {
        if (preName === tool.name || !byName.has(preName)) continue;
        const i = toolIndexMap.get(preName);
        const j = toolIndexMap.get(tool.name);
        if (i === undefined || j === undefined) continue;
        adjacencyMatrix[i * n + j] = Math.max(adjacencyMatrix[i * n + j], PREREQUISITE_WEIGHT);
        edges.push({ from: preName, to: tool.name, type: "PREREQUISITE", weight: PREREQUISITE_WEIGHT });
      }
    }

    for (const tool of tools) {
      for (const cName of tool.conflicts ?? []) {
        if (cName === tool.name || !byName.has(cName)) continue;
        // Dedupe: una sola dirección (from < to).
        if (tool.name >= cName) continue;
        edges.push({
          from: tool.name,
          to: cName,
          type: "MUTUALLY_EXCLUSIVE",
          weight: MUTUALLY_EXCLUSIVE_WEIGHT,
        });
      }
    }

    const seen = new Set<string>();
    for (let a = 0; a < tools.length; a++) {
      for (let b = a + 1; b < tools.length; b++) {
        if (tools[a].group && tools[a].group === tools[b].group) {
          const key = [tools[a].name, tools[b].name].sort().join("\u0000");
          if (seen.has(key)) continue;
          seen.add(key);
          edges.push({
            from: tools[a].name,
            to: tools[b].name,
            type: "CO_OCCURRENCE",
            weight: CO_OCCURRENCE_WEIGHT,
          });
        }
      }
    }

    return edges;
  }
}

/**
 * Firma canónica de la herramienta (la proyección de nodo del espacio latente).
 * Excluye el `inputSchema` y recorta los ejemplos JSON/metadatos de
 * `description`/`intentSummary`: ambos diluían la señal semántica del nodo
 * (p. ej. `hero_configurar`, con un esquema `slides` enorme, quedaba por debajo
 * de `hero_obtener`, de firma limpia).
 */
function toolSignature(tool: ToolDefinition): string {
  return [
    tool.name,
    tool.group,
    tool.category,
    shortText(tool.description),
    (tool.tags ?? []).join(" "),
    shortText(tool.intentSummary),
  ]
    .filter(Boolean)
    .join(" ");
}

/**
 * Primer párrafo de un texto libre, descartando ejemplos JSON y metadatos
 * (permisos, "Complemento…") que no describen la intención de la herramienta.
 */
function shortText(text: string | undefined, maxLen = 320): string {
  if (!text) return "";
  const cut = text.search(/\n|[{\[]/);
  const head = cut >= 0 ? text.slice(0, cut) : text;
  return head.replace(/\s+/g, " ").trim().slice(0, maxLen);
}

function computeVersionHash(tools: ToolDefinition[]): string {
  const digest = tools
    .map(
      (t) =>
        `${t.name}\u0000${toolSignature(t)}\u0000${(t.prerequisites ?? []).join(",")}\u0000${(t.conflicts ?? []).join(",")}`,
    )
    .sort()
    .join("\u0001");
  return createHash("sha1").update(digest).digest("hex");
}

