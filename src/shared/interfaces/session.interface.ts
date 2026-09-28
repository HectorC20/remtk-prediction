export interface SessionStateResult {
  zt: Float32Array;
  topicShift: boolean;
  /** Coseno del turno contra z_{t-1}; undefined si la sesión no tenía estado previo. */
  topicSim?: number;
  model: string;
}
