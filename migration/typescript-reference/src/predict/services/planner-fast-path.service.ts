/**
 * PlannerFastPathService: Enrutador desacoplado para optimización de tokens y latencia.
 *
 * Resuelve la redundancia arquitectónica del doble pase (Planner -> Harness):
 * Cuando remtk-prediction determina 0 herramientas y no hay planes pendientes,
 * permite a los consumidores (harnesses, key-remtk) cortocircuitar la llamada del
 * planificador LLM y pasar directo al modo conversacional, ahorrando ~2500 tokens
 * y la mitad del tiempo de respuesta.
 */

import type { ToolDefinition } from "src/shared/interfaces/domain.interface";

export type FastPathAction = "DIRECT_CHAT" | "DIRECT_EXECUTE" | "FULL_PLANNER";

export interface FastPathInput {
  /** Herramientas devueltas por remtk-prediction */
  tools: ToolDefinition[] | Array<{ name: string; prerequisites?: string[] }>;
  /** Texto del turno del usuario */
  text: string;
  /** Scores de ranking de la predicción si existen */
  rankedScores?: number[];
  /** Indica si la sesión actual tiene un plan previo pendiente de confirmación */
  hasPendingPlan?: boolean;
  /** Estimación de tokens promedio que consume el prompt del planificador */
  averagePlannerTokens?: number;
}

export interface FastPathDecision {
  /** Acción recomendada para el arnés */
  action: FastPathAction;
  /** Indica si debe saltearse la llamada al planificador LLM */
  skipPlanner: boolean;
  /** Justificación técnica de la decisión */
  reason: string;
  /** Estimación de tokens ahorrados al evitar la llamada del planificador */
  estimatedTokensSaved: number;
  /** Herramienta sugerida en caso de ejecución directa (si aplica) */
  directTool?: string;
  /** Payload sugerido para configurar la llamada directa al chat */
  suggestedChatContext?: {
    situation: "DIRECT_CHAT";
    toolCount: 0;
    promptRole: "conversational_assistant";
  };
}

export class PlannerFastPathService {
  private readonly defaultPlannerTokens: number;

  constructor(options?: { defaultPlannerTokens?: number }) {
    this.defaultPlannerTokens = options?.defaultPlannerTokens ?? 2500;
  }

  /**
   * Evalúa si el turno actual califica para el Fast-Path sin llamar al planificador.
   */
  decide(input: FastPathInput): FastPathDecision {
    const { tools, hasPendingPlan = false } = input;
    const tokensSaved = input.averagePlannerTokens ?? this.defaultPlannerTokens;

    // Caso 1: 0 herramientas y sin plan pendiente -> Salteo garantizado al chat
    if ((!tools || tools.length === 0) && !hasPendingPlan) {
      return {
        action: "DIRECT_CHAT",
        skipPlanner: true,
        reason: "remtk-prediction determinó 0 herramientas y no hay plan pendiente en sesión. Se saltea el planificador LLM.",
        estimatedTokensSaved: tokensSaved,
        suggestedChatContext: {
          situation: "DIRECT_CHAT",
          toolCount: 0,
          promptRole: "conversational_assistant",
        },
      };
    }

    // Caso 2: Plan pendiente en sesión -> Se requiere el planificador para resolver confirmación/rechazo
    if (hasPendingPlan) {
      return {
        action: "FULL_PLANNER",
        skipPlanner: false,
        reason: "Existe un plan pendiente en sesión que requiere juicio de confirmación/modificación por el planificador.",
        estimatedTokensSaved: 0,
      };
    }

    // Caso 3: Múltiples herramientas o herramientas con prerrequisitos -> Planificador completo
    const hasPrerequisites = tools.some(
      (t) => Array.isArray(t.prerequisites) && t.prerequisites.length > 0,
    );

    if (tools.length > 1 || hasPrerequisites) {
      return {
        action: "FULL_PLANNER",
        skipPlanner: false,
        reason: `Se predijeron ${tools.length} herramientas${hasPrerequisites ? " con prerrequisitos DAG" : ""}. Requiere secuenciación del planificador.`,
        estimatedTokensSaved: 0,
      };
    }

    // Caso 4: Una sola herramienta atómica de alta confianza (opcional para ejecución directa)
    const singleTool = tools[0];
    const topScore = input.rankedScores?.[0] ?? 1.0;

    if (singleTool && topScore >= 0.85) {
      return {
        action: "DIRECT_EXECUTE",
        skipPlanner: true,
        reason: `Herramienta única atómica '${singleTool.name}' con alta confianza (${topScore.toFixed(2)}). Puede ejecutarse o proponerse directamente.`,
        estimatedTokensSaved: tokensSaved,
        directTool: singleTool.name,
      };
    }

    // Fallback estándar por defecto
    return {
      action: "FULL_PLANNER",
      skipPlanner: false,
      reason: "Catálogo con herramientas ambiguas o confianza moderada. Se delega al planificador.",
      estimatedTokensSaved: 0,
    };
  }
}

/**
 * Función utilitaria estática externa para consumo directo en una sola línea.
 */
export function decidePlannerFastPath(input: FastPathInput): FastPathDecision {
  const service = new PlannerFastPathService({
    defaultPlannerTokens: input.averagePlannerTokens,
  });
  return service.decide(input);
}
