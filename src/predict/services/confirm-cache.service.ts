/** Caché de confirmación por sesión: reutiliza el plan previo en confirmaciones vacías. */
import type { GraphPredictionResult } from "../../shared/interfaces/graph.interface";

export class ConfirmationCache {
  private readonly store = new Map<string, GraphPredictionResult>();

  get(sessionId: string): GraphPredictionResult | undefined {
    return this.store.get(sessionId);
  }

  set(sessionId: string, result: GraphPredictionResult): void {
    this.store.set(sessionId, result);
  }
}
