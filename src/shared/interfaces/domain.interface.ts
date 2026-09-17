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
  /**
   * Palabras clave que acompañan a la tarea (p. ej. las que delega el
   * planificador a cada mini-agente). Se añaden a la consulta para afinar el
   * recall BM25 y el match cross-idioma.
   */
  keywords?: string[];
}


/**
 * Señal de refuerzo del aprendizaje léxico (§8 del plan de entrenamiento).
 * `used` debe contener SOLO herramientas cuya ejecución confirmó éxito real;
 * `rejected` las predichas que fallaron o no se usaron.
 */
export interface FeedbackInput {
  sessionId: string;
  tenant: string;
  text: string;
  /** Agente opcional: ausente/vacío ⇒ chat general (scope = tenant). */
  agentId?: string;
  /** Herramientas cuyo uso terminó en éxito real. */
  used: string[];
  /** Herramientas predichas que no funcionaron. */
  rejected?: string[];
}

export interface FeedbackResult {
  ok: true;
  /** Señales acumuladas del canal tras aplicar este evento. */
  events: number;
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
  /** Peso efectivo de la señal aprendida del canal (0 = capa inactiva, §6.5). */
  learnWeight?: number;
  /** Herramientas del canal con score aprendido en esta predicción. */
  learnTerms?: number;
}
