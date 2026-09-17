import type {
  FeedbackInput,
  FeedbackResult,
  PredictionInput,
  ToolDefinition,
} from "./domain.interface";
import type { GraphPredictionResult } from "./graph.interface";
import type {
  InterestPredictionInput,
  InterestPredictionResult,
  MemoryPredictionInput,
  MemoryPredictionResult,
} from "./memory.interface";

export interface IPredictionOrchestrator {
  predict(input: PredictionInput): Promise<GraphPredictionResult>;
  predictMemory(input: MemoryPredictionInput): Promise<MemoryPredictionResult>;
  predictInterest(input: InterestPredictionInput): Promise<InterestPredictionResult>;
  registerTools(tenant: string, tools: ToolDefinition[], agentId?: string): Promise<{ indexed: number }>;
  countTools(tenant: string, agentId?: string): number;
  /** Señal de refuerzo del perfil léxico del canal (§8.1). */
  feedback(input: FeedbackInput): FeedbackResult;
}
