/** Debugger: trazas en anillo y estadísticas para el endpoint /debug. */
import type { Trace } from "../types";

const MAX_TRACES = 200;

export interface DebugStats {
  tenants: number;
  confirmGets: number;
  confirmHits: number;
  confirmSets: number;
  embeddingsRecomputed: number;
  embeddingsCached: number;
}

export class Debugger {
  private readonly traces: Trace[] = [];
  private stats: DebugStats = {
    tenants: 0,
    confirmGets: 0,
    confirmHits: 0,
    confirmSets: 0,
    embeddingsRecomputed: 0,
    embeddingsCached: 0,
  };

  setTenants(n: number): void {
    this.stats.tenants = n;
  }

  bump(part: Partial<DebugStats>): void {
    for (const k of Object.keys(part) as (keyof DebugStats)[]) {
      const v = part[k];
      if (typeof v === "number") {
        (this.stats as unknown as Record<string, number>)[k] += v;
      }
    }
  }

  addTrace(trace: Trace): void {
    this.traces.push(trace);
    if (this.traces.length > MAX_TRACES) this.traces.shift();
  }

  snapshot(): { stats: DebugStats; traces: Trace[] } {
    return {
      stats: { ...this.stats },
      traces: [...this.traces].slice(-20).reverse(),
    };
  }
}
