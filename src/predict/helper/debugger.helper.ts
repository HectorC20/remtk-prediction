/** Debugger: trazas en anillo y estadísticas para el endpoint /debug. */
import { Injectable } from "@nestjs/common";
import type { Trace } from "../../shared/interfaces/domain.interface";
import { DebugStats } from "src/shared/interfaces";
import { MAX_TRACES } from "src/shared/constants/predict";

@Injectable()
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
