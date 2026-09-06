/**
 * Contrato del grafo de dependencias entre herramientas MCP.
 *
 * El enrutador topológico reemplaza la reducción plana de herramientas por un
 * espacio latente directo: cada herramienta es un nodo (embedding e5-small L2
 * sobre su firma) y las aristas codifican dependencias (PREREQUISITE),
 * co-ocurrencia (CO_OCCURRENCE) y exclusión mutua (MUTUALLY_EXCLUSIVE).
 */
import type { PredictionResult, ToolDefinition } from "./domain.interface";

export type EdgeType = "PREREQUISITE" | "CO_OCCURRENCE" | "MUTUALLY_EXCLUSIVE";

export interface GraphEdge {
  from: string; // toolId origen
  to: string;   // toolId destino
  type: EdgeType;
  weight: number; // Peso de propagación [0.0 - 1.0]
}

export interface ToolGraphNode {
  id: string;
  name: string;
  embedding: Float32Array; // 384 dimensiones L2
  prerequisites: string[];
  conflicts: string[];
  definition: ToolDefinition;
}

export interface TenantToolGraph {
  tenant: string;
  versionHash: string;
  nodes: Map<string, ToolGraphNode>;
  adjacencyMatrix: Float32Array; // Matriz aplanada (N x N), A[i][j] = peso de i → j
  toolIndexMap: Map<string, number>; // Mapeo name -> Índice matricial
}

/** Resultado de predicción con el subgrafo activado (DAG + orden de ejecución). */
export interface GraphPredictionResult extends PredictionResult {
  graph: {
    nodes: string[];
    edges: GraphEdge[];
    executionOrder: string[];
  };
}
