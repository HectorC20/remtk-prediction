/**
 * Cliente local de remtk-prediction para el benchmark.
 *
 * Levanta el listener de predicción en puerto efímero (igual que tests/e2e),
 * indexa el catálogo del bench y espera el warm-up de embeddings (grafo) antes
 * de medir. ONNX queda habilitado (modelos reales); Qdrant se desactiva salvo
 * que se pida con --qdrant, y el modo usado queda registrado en el reporte.
 */
import { startPredictServer, type ServerHandle } from "../src/app.module";
import type { ToolDefinition, Trace } from "../src/shared/interfaces/domain.interface";
import type { BenchTurn } from "./cases";

export interface RemtkTurnResult {
  tools: string[];
  latencyMs: number;
  modelSize: string;
  turnType?: string;
  viaGraph: boolean;
}

export class RemtkBenchClient {
  private server?: ServerHandle;
  private base = "";
  mode = "";

  async start(tools: ToolDefinition[], opts: { qdrant: boolean }): Promise<void> {
    if (!opts.qdrant) process.env.QDRANT_ENABLED = "0";
    this.server = await startPredictServer(0);
    if (!opts.qdrant) delete process.env.QDRANT_ENABLED;
    this.base = `http://localhost:${this.server.port}`;

    const res = await fetch(`${this.base}/tools`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant: "bench", tools }),
    });
    if (!res.ok) throw new Error(`registro de tools falló: HTTP ${res.status}`);

    // El warm-up (embeddings + grafo) corre en background tras POST /tools:
    // sondear hasta que la ruta topológica responda o agotar la espera.
    const deadline = Date.now() + 180_000;
    let viaGraph = false;
    let modelSize = "hash";
    while (Date.now() < deadline) {
      const probe = await this.predict("bench-warmup", "envía un correo al equipo", []);
      viaGraph = probe.viaGraph;
      modelSize = probe.modelSize;
      if (viaGraph) break;
      await new Promise((r) => setTimeout(r, 1_000));
    }
    const juicioEnv = (process.env.JUICIO_ENABLED ?? "").toLowerCase();
    const juicioOn = juicioEnv === "" || ["1", "true", "yes", "on"].includes(juicioEnv);
    this.mode =
      `${modelSize}${viaGraph ? "+graph" : "+flat"}${opts.qdrant ? "+qdrant" : ""}` +
      `${juicioOn ? "+juicio" : "-juicio"}`;
  }

  /** Juega un turno contra /predict con historial explícito. */
  async predict(sessionId: string, text: string, history: BenchTurn[]): Promise<RemtkTurnResult> {
    const start = Date.now();
    const res = await fetch(`${this.base}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, tenant: "bench", text, source: "human", history }),
    });
    const latencyMs = Date.now() - start;
    if (!res.ok) throw new Error(`/predict falló: HTTP ${res.status}`);
    const data = (await res.json()) as {
      tools?: { name: string }[];
      modelSize?: string;
      graph?: { nodes?: string[] };
    };
    return {
      tools: (data.tools ?? []).map((t) => t.name),
      latencyMs,
      modelSize: data.modelSize ?? "?",
      viaGraph: (data.graph?.nodes?.length ?? 0) > 0,
    };
  }

  async close(): Promise<void> {
    await this.server?.close();
  }

  /** Traza más reciente de /debug para el turno evaluado. */
  async lastTrace(sessionId: string): Promise<Trace | undefined> {
    const res = await fetch(`${this.base}/debug`);
    if (!res.ok) return undefined;
    const data = (await res.json()) as { traces?: Trace[] };
    return (data.traces ?? []).find((t) => t.sessionId === sessionId);
  }
}
