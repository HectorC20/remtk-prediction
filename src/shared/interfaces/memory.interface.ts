

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
  /** Agente opcional: ausente/'' ⇒ filtro legacy por userId; 'general' ⇒ userId + agente general. */
  agentId?: string;
}

export interface MemoryPredictionResult {
  memories: MemoryDefinition[];
  topicShift: boolean;
  topicScore: number;
  modelSize: string;
  rankedScores: number[];
}

/**
 * Entrada del scorer de interés. Las `anchors` son los temas de interés
 * descubiertos por remtk-memory (subgrafos/anclas); el predictor NO conoce
 * ningún diccionario de negocio, solo compara el turno contra esas etiquetas.
 */
export interface InterestPredictionInput {
  sessionId: string;
  tenant: string;
  text: string;
  /** Agente opcional: ausente/vacío ⇒ chat general (scope = tenant). */
  agentId?: string;
  /** Anclas de tema (etiquetas de subgrafo) contra las que medir el match. */
  anchors?: string[];
  /** Tope de anclas evaluadas (default INTEREST_MAX_ANCHORS). */
  limit?: number;
}

export interface InterestMatch {
  anchor: string;
  score: number;
}

export interface InterestPredictionResult {
  /** Score global de interés del turno (0..1). */
  interestScore: number;
  /** Ancla (tema) con mejor match; null si no se pasaron anclas. */
  topic: string | null;
  /** Coseno del mejor ancla (0..1). */
  topicScore: number;
  /** Coseno máximo contra los arquetipos de intención (0..1). */
  intentScore: number;
  /** Mejores matches por ancla, ordenados desc. */
  matches: InterestMatch[];
  /** Modelo usado por el scorer: "small"/"large" (ONNX) o "hash" (degradado). */
  method: string;
}
