/**
 * Contrato de dominio compartido (1:1 con el Tool Prediction Server Go).
 *
 * Uniones de valor e interfaces del pipeline de predicción, autocontenidas.
 * `ModelSize` NO vive aquí: se deriva de la constante `onnxModelSizeSmall` en
 * `constants/models/version.models.ts` (fuente única del literal "small").
 */

import { ChatMessage } from "./index";

/** Complejidad estimada del prompt (gobierna el corte del pipeline). */
export type ToolComplexity = "simple" | "moderate" | "complex";

export interface ToolDefinition {
  id: string;
  name: string;
  group: string;
  category: string;
  description: string;
  tags: string[];
  intentSummary: string;
  inputSchema: Record<string, unknown>;
  /** Herramientas que deben ejecutarse antes (relaciones declaradas, opcional). */
  prerequisites?: string[];
  /** Herramientas mutuamente excluyentes (relaciones declaradas, opcional). */
  conflicts?: string[];
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
  /** Agente opcional: ausente/vacío ⇒ chat general (scope = tenant). */
  agentId?: string;
}


/** Traza de una predicción (para el endpoint /debug). */
export interface Trace {
  sessionId: string;
  tenant: string;
  /** Scope efectivo de la partición (tenant o `tenant::agentId`); opcional en debug. */
  scopeKey?: string;
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
