import type {
  PredictionInput,
  ToolDefinition,
} from "./domain.interface";
import type { GraphPredictionResult } from "./graph.interface";
import type {
  MemoryPredictionInput,
  MemoryPredictionResult,
} from "./memory.interface";

export interface IPredictionOrchestrator {
  predict(input: PredictionInput): Promise<GraphPredictionResult>;
  predictMemory(input: MemoryPredictionInput): Promise<MemoryPredictionResult>;
  registerTools(tenant: string, tools: ToolDefinition[], agentId?: string): Promise<{ indexed: number }>;
  countTools(tenant: string, agentId?: string): number;
}
