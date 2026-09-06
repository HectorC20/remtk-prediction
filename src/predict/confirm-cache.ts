/** Caché de confirmación por sesión: reutiliza el plan previo en confirmaciones vacías. */
import type { PredictionResult } from "../types";

export class ConfirmationCache {
  private readonly store = new Map<string, PredictionResult>();

  get(sessionId: string): PredictionResult | undefined {
    return this.store.get(sessionId);
  }

  set(sessionId: string, result: PredictionResult): void {
    this.store.set(sessionId, result);
  }
}
