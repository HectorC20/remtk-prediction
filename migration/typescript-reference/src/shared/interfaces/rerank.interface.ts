import type { PredictedContext, ToolComplexity, ToolDefinition } from "./domain.interface";
import type { GraphEdge } from "./graph.interface";

export interface RerankResult {
  tools: ToolDefinition[];
  complexity: ToolComplexity;
  modelSize: string;
  rankedScores: number[];
  calibratedScores?: number[];
  context?: PredictedContext;
}

export interface GraphRerankResult extends RerankResult {
  graph: {
    nodes: string[];
    edges: GraphEdge[];
    executionOrder: string[];
  };
}
