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
  /**
   * Espacio de necesidad (opcional, indexación dual de la capa de juicio):
   * cómo un usuario expresaría el problema que esta herramienta resuelve
   * ("mi jefe me pide el balance", "necesito saber si lloverá"). Si está
   * presente se embebe y puntúa en paralelo a la descripción técnica.
   */
  problemSpace?: string;
}

export interface ScoredTool {
  name: string;
  score: number;
}

/** Tipo de acción semántica inferida de la intención. */
export type ActionType =
  | "create"
  | "read"
  | "update"
  | "delete"
  | "execute"
  | "confirm"
  | "query"
  | "meta"
  | "unknown";

/** Fase del diálogo deducida por el pipeline. */
export type DialogPhase = "discovery" | "execution" | "confirmation" | "clarification";

/**
 * Contexto enriquecido predicho junto a las herramientas.
 * Proporciona a los agentes el por qué y el cómo de la predicción.
 */
export interface PredictedContext {
  intent: {
    primaryAction: ActionType | string;
    confidence: number;
    category?: string;
    summary: string;
  };
  constraints: {
    negations: string[];
    isConfirmation: boolean;
    isExploratory: boolean;
  };
  dialogState: {
    phase: DialogPhase;
    topicShift: boolean;
    activeDomain?: string;
  };
  anticipation: {
    suggestedNextTools: string[];
    reasoning: string;
  };
}

export interface PredictionResult {
  tools: ToolDefinition[];
  complexity: ToolComplexity;
  modelSize: string;
  rankedScores: number[];
  context?: PredictedContext;
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
  /**
   * Nombres de herramientas a OMITIR del resultado. Es la palanca de la
   * segunda pasada del mini-agente: ya se ofrecieron esas herramientas y no
   * resolvieron la subtarea, así que se descartan antes del umbral adaptativo
   * para que entren otras del mismo catálogo. La comparación es por nombre
   * exacto, insensible a mayúsculas.
   */
  exclude?: string[];
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
  /** λ de la puerta de coherencia de juicio (undefined = capa inactiva). */
  juicioGateLambda?: number;
  /** Motivo de abstención de juicio si el turno terminó en ∅ (noop/energy). */
  juicioAbstained?: string;
  /** Veredicto NLI de estado del turno (confirm/reject/neutral). */
  juicioEstado?: "confirm" | "reject" | "neutral";
  /** Coseno máximo contra el prototipo __NOOP__ (0 = capa inactiva). */
  juicioNoopScore?: number;
  /** Energía libre de Helmholtz del ranking de herramienta. */
  juicioEnergy?: number;
  /** Señales de re-rank de juicio que intervinieron este turno. */
  juicioSignals?: { maxsim: boolean; reranker: boolean; specificity: boolean };
}
