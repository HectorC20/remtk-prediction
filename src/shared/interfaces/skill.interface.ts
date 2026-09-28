/**
 * Contratos de la Memoria de Habilidades y Grafo de Entidades Paramétricas.
 *
 * Arquitectura alternativa complementaria:
 * 1. Habilidades (Skills): Secuencias/DAGs aprendidos con su espacio vectorial
 *    de anclaje de parámetros (Parameter Grounding).
 * 2. Entidades de Sesión: Grafo de entidades activas, ciclo de vida y acciones
 *    pendientes aislado por sesión (sessionId) y tenant.
 */

export interface ParameterAnchor {
  id: string;
  name: string;
  embedding: Float32Array;
  metadata?: Record<string, unknown>;
}

export interface SkillDefinition {
  id: string;
  name: string;
  description: string;
  intentSummary: string;
  intentEmbedding?: Float32Array;
  /** Herramientas del flujo en orden topológico */
  tools: string[];
  /** Parámetros referenciales aprendidos (ej: categoryId, zoneId) */
  parameterGrounding?: Record<string, ParameterAnchor[]>;
  /** Contador de éxitos / refuerzo */
  reinforcementScore: number;
}

export interface SessionEntityState {
  id: string;
  type: string;
  name: string;
  slug?: string;
  attributes?: Record<string, unknown>;
  state: {
    lifecycle: "created" | "updated" | "queried" | "archived";
    pendingAction?: "image_upload" | "confirmation" | "parameter_completion";
    missingFields?: string[];
  };
  activityScore: number;
  updatedAt: number;
  embedding?: Float32Array;
}

export interface SkillPredictInput {
  sessionId: string;
  tenant?: string;
  agentId?: string;
  keyRemtk?: string;
  text: string;
  /** Datos adjuntos (e.g. URLs de archivos, payloads de imagen) */
  attachments?: {
    fileId?: string;
    url?: string;
    mimeType?: string;
  }[];
}

export interface GroundedParameterMatch {
  key: string;
  matchedId: string;
  matchedName: string;
  confidence: number;
}

export interface SkillPredictResult {
  matchedSkill?: {
    id: string;
    name: string;
    confidence: number;
    tools: string[];
  };
  resolvedEntity?: {
    id: string;
    name: string;
    type: string;
    matchedPendingAction?: string;
    confidence: number;
  };
  groundedParameters: GroundedParameterMatch[];
  /** Payload pre-resuelto listo para invocar directamente la herramienta sin [FALTAN_DATOS] */
  preResolvedArgs: Record<string, unknown>;
  suggestedAction: "execute_direct" | "discover_prerequisites" | "ask_user";
  contextPrompt: string;
}
