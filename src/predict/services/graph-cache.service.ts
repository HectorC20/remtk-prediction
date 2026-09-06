/**
 * ToolGraphCacheService: topología en memoria de las herramientas por tenant.
 *
 * En el ciclo de POST /tools reconstruye el grafo G = (V, E) del tenant:
 *   - Cada nodo embebe la firma de su herramienta con prefijo `passage:` (e5-small).
 *   - Las aristas se infieren de las relaciones declaradas (`prerequisites`,
 *     `conflicts`) y del esquema `inputSchema`, más la co-ocurrencia por `group`.
 *   - La matriz de adyacencia A (N×N) guarda el peso de la arista i → j
 *     (i es pre-requisito de j) para la propagación del Graph Router.
 */
import { createHash } from "node:crypto";
import { EmbeddingEngineService } from "../../embedding/embedding-engine";
import { log } from "../../logger";
import type { ToolDefinition } from "../../shared/interfaces/domain.interface";
import type {
  GraphEdge,
  TenantToolGraph,
  ToolGraphNode,
} from "../../shared/interfaces/graph.interface";
import { normalizeToken } from "../keywords";

const PREREQUISITE_WEIGHT = 0.85;
const CO_OCCURRENCE_WEIGHT = 0.4;
const MUTUALLY_EXCLUSIVE_WEIGHT = 1.0;

export class ToolGraphCacheService {
  private readonly graphs = new Map<string, TenantToolGraph>();
  private readonly edgesByTenant = new Map<string, GraphEdge[]>();

  constructor(private readonly engine: EmbeddingEngineService) {}

  get(tenant: string): TenantToolGraph | undefined {
    return this.graphs.get(tenant);
  }

  edges(tenant: string): GraphEdge[] {
    return this.edgesByTenant.get(tenant) ?? [];
  }

  /** POST /tools: reconstruye el grafo del tenant (embeddings passage: + aristas). */
  async buildTenantGraph(tenant: string, tools: ToolDefinition[]): Promise<TenantToolGraph> {
    const versionHash = computeVersionHash(tools);

    // Nodos: embedding passage: de la firma de cada herramienta (prioridad baja,
    // es recompute masivo de registro, no de predicción).
    const nodes = new Map<string, ToolGraphNode>();
    for (const tool of tools) {
      const res = await this.engine.embedPassage(toolSignature(tool), "small", { high: false });
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
    const edges = inferEdges(tools, toolIndexMap, adjacencyMatrix);

    const graph: TenantToolGraph = { tenant, versionHash, nodes, adjacencyMatrix, toolIndexMap };
    this.graphs.set(tenant, graph);
    this.edgesByTenant.set(tenant, edges);
    log(
      `[graph] tenant=${tenant} nodes=${n} edges=${edges.length} hash=${versionHash.slice(0, 8)}`,
    );
    return graph;
  }
}

/** Firma canónica de la herramienta (la proyección de nodo del espacio latente). */
function toolSignature(tool: ToolDefinition): string {
  return [
    tool.name,
    tool.group,
    tool.category,
    tool.description,
    (tool.tags ?? []).join(" "),
    tool.intentSummary,
    JSON.stringify(tool.inputSchema ?? {}),
  ]
    .filter(Boolean)
    .join(" ");
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

/**
 * Infiere las aristas del grafo:
 *  1. PREREQUISITE: relaciones declaradas + inferencia por `inputSchema`.
 *  2. MUTUALLY_EXCLUSIVE: relaciones declaradas (simétricas, una sola arista).
 *  3. CO_OCCURRENCE: herramientas del mismo `group`.
 */
function inferEdges(
  tools: ToolDefinition[],
  toolIndexMap: Map<string, number>,
  adjacencyMatrix: Float32Array,
): GraphEdge[] {
  const edges: GraphEdge[] = [];
  const byName = new Map(tools.map((t) => [t.name, t]));
  const n = tools.length;

  for (const tool of tools) {
    const prereqs = new Set<string>(tool.prerequisites ?? []);
    for (const inferred of inferPrerequisitesFromSchema(tool, tools)) prereqs.add(inferred);
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

/**
 * Inferencia ligera de pre-requisitos por esquema: si una clave del
 * `inputSchema` de la herramienta coincide con el nombre o el grupo de otra
 * herramienta, ésta es un pre-requisito (produce ese argumento).
 */
function inferPrerequisitesFromSchema(tool: ToolDefinition, tools: ToolDefinition[]): string[] {
  const keys = Object.keys(tool.inputSchema ?? {}).map((k) => normalizeToken(k));
  if (keys.length === 0) return [];
  const out: string[] = [];
  for (const other of tools) {
    if (other.name === tool.name) continue;
    const candidates = new Set([normalizeToken(other.name), normalizeToken(other.group)]);
    for (const key of keys) {
      if (candidates.has(key)) {
        out.push(other.name);
        break;
      }
    }
  }
  return out;
}
