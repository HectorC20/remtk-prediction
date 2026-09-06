

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
