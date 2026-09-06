/** Tipos de dominio, contrato 1:1 con el Tool Prediction Server original (Go). */

export type ToolComplexity = "simple" | "moderate" | "complex";
// export type ModelSize = "large" | "small";
export type ModelSize = "small";

export interface ToolDefinition {
  id: string;
  name: string;
  group: string;
  category: string;
  description: string;
  tags: string[];
  intentSummary: string;
  inputSchema: Record<string, unknown>;
}

export interface ScoredTool {
  name: string;
  score: number;
}

export interface PredictionResult {
  tools: ToolDefinition[];
  complexity: ToolComplexity;
  modelSize: string;
  rankedScores: number[];
}

export interface PredictionInput {
  sessionId: string;
  tenant: string;
  text: string;
  source: "human" | "agent";
  priorPlan?: string;
  /** Mensajes previos de la conversación (contexto para la predicción). */
  history?: ChatMessage[];
}

/** Mensaje de historial (rol + contenido plano). */
export interface ChatMessage {
  role: string;
  content: string;
}

export interface MemoryDefinition {
  id: string;
  content: string;
  chatId?: string;
  metadata?: Record<string, unknown>;
  createdAt?: string;
  score: number;
}

export interface MemoryCandidate {
  memory: MemoryDefinition;
  retrievalScore: number;
}

export interface MemoryPredictionInput {
  sessionId: string;
  tenant: string;
  text: string;
  limit: number;
}

export interface MemoryPredictionResult {
  memories: MemoryDefinition[];
  topicShift: boolean;
  topicScore: number;
  modelSize: string;
  rankedScores: number[];
}

/** Traza de una predicción (para el endpoint /debug). */
export interface Trace {
  sessionId: string;
  tenant: string;
  turnType: string;
  confirmHit: boolean;
  recall: number;
  degraded: boolean;
  inputTokens: number;
  outputTools: number;
  modelSize: string;
  embeddingsRecomputed: number;
  embeddingsCached: number;
  complexity: ToolComplexity;
  latencyMs: number;
  timestamp: string;
}
