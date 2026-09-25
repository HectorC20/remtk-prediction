/**
 * Tipos de la capa de juicio (docs/juicio.md): veredictos de estado, contexto
 * de pre-pipeline y resultado de post-pipeline.
 */

/** Veredicto del NLI de estado condicionado sobre la propuesta pendiente. */
export type JuicioEstado = "confirm" | "reject" | "neutral";

/** Motivo de abstención (conjunto vacío de herramientas). */
export type JuicioAbstainReason = "noop" | "energy";

/**
 * Contexto que `pre()` produce y `post()` consume dentro del mismo turno.
 */
export interface JuicioContext {
  /** Embedding `query:` del texto actual (enfocado por saliencia si era multi-tramo). */
  uText?: Float32Array;
  /** Tramo(s) de mayor saliencia funcional cuando el turno traía relleno extenso. */
  focusedText?: string;
  /** true cuando un turno multi-tramo carece de pico funcional en todos sus tramos (divagación pura). */
  allSpansNoop?: boolean;
  /** Veredicto NLI de estado (neutral si no había propuesta pendiente). */
  estado: JuicioEstado;
  /**
   * λ de la puerta de coherencia: 0 = aísla el turno, 1 = el historial domina.
   * undefined = la capa no operó este turno (mezcla fija 70/30 histórica).
   */
  gateLambda?: number;
  /** true si el turno se aísla del historial (λ bajo el mínimo de texto). */
  historyGated: boolean;
  /** true si el veredicto resolvió el turno (confirm → plan cacheado, reject → vacío). */
  resolved: boolean;
  /** Motivo de resolución anticipada, para trazas. */
  resolvedBy?: "nli_confirm" | "nli_reject";
}

/** Diagnóstico que `post()` adjunta a la traza del turno. */
export interface JuicioPostDiag {
  abstained: boolean;
  abstainReason?: JuicioAbstainReason;
  noopScore: number;
  energy: number;
  maxsimApplied: boolean;
  rerankerApplied: boolean;
  specificityApplied: boolean;
}
