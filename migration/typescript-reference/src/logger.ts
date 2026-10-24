/** Logger compacto con timestamp, equivalente al log estándar del server Go. */
import { AsyncLocalStorage } from "node:async_hooks";
import { appendFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

/** Carpeta de logs: env `LOGS_DIR` o `<cwd>/logs`. */
const logsDir = (process.env.LOGS_DIR ?? "").trim() || resolve(process.cwd(), "logs");

function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return String(value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function logFileName(): string {
  const d = new Date();
  const ymd = [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, "0"),
    String(d.getDate()).padStart(2, "0"),
  ].join("-");
  return resolve(logsDir, `prediction-${ymd}.log`);
}

/** Escribe una línea en el archivo de logs diario (mejor esfuerzo, nunca lanza). */
function persist(line: string): void {
  try {
    mkdirSync(logsDir, { recursive: true });
    appendFileSync(logFileName(), line + "\n");
  } catch {
    /* noop: si no se puede escribir en disco, seguimos solo en consola. */
  }
}

function write(level: "info" | "warn", tag: string, args: unknown[]): void {
  const iso = new Date().toISOString();
  const suffix = args.length > 0 ? " " + args.map(stringify).join(" ") : "";
  persist(`${iso} [${tag}]${suffix}`);
  // Espejo en consola (formato original).
  if (level === "warn") {
    console.warn(iso, `[${tag}]`, ...args);
  } else {
    console.log(iso, `[${tag}]`, ...args);
  }
}

export function log(tag: string, ...args: unknown[]): void {
  write("info", tag, args);
}

export function warn(tag: string, ...args: unknown[]): void {
  write("warn", tag, args);
}

// ── Stage profiler ───────────────────────────────────────────────────────────
//
// Generic per-request timing: instead of logging specific failures, every
// profiled operation (predict, memory, interest, warm-up) reports how long each
// stage took and how much of that time was spent WAITING in the ONNX inference
// queue versus actually RUNNING inference. The context is propagated with
// AsyncLocalStorage, so nested services (juicio, embedding) record into the
// active request without changing their signatures.

/** Total latency (ms) above which a profile report is emitted as a warning. */
const slowThresholdMs = Number(process.env.PROFILE_SLOW_MS ?? 1500) || 1500;
/** Set PROFILE_ENABLED=0 to silence timing reports. */
const profileEnabled = (process.env.PROFILE_ENABLED ?? "1") !== "0";

export class StageProfiler {
  private readonly stages = new Map<string, number>();
  private readonly startedAt = performance.now();
  private context = "";
  onnxCalls = 0;
  onnxWaitMs = 0;
  onnxRunMs = 0;
  onnxMaxQueueDepth = 0;

  constructor(readonly tag: string) {}

  /** Adds `ms` to a stage. Nested stages use `parent>child` naming. */
  add(name: string, ms: number): void {
    this.stages.set(name, (this.stages.get(name) ?? 0) + ms);
  }

  /** Free-form context appended to the report (session, output size, ...). */
  setContext(context: string): void {
    this.context = context;
  }

  /** Records one ONNX inference: time queued before start and time running. */
  recordOnnx(waitMs: number, runMs: number, queueDepth: number): void {
    this.onnxCalls++;
    this.onnxWaitMs += waitMs;
    this.onnxRunMs += runMs;
    if (queueDepth > this.onnxMaxQueueDepth) this.onnxMaxQueueDepth = queueDepth;
  }

  report(): void {
    if (!profileEnabled) return;
    const total = performance.now() - this.startedAt;
    const fmt = (ms: number) => `${Math.round(ms)}ms`;
    const entries = [...this.stages.entries()];
    const stageText = entries.map(([n, ms]) => `${n}=${fmt(ms)}`).join(" ");

    // Slowest top-level stage (nested `a>b` entries are detail, not candidates).
    const topLevel = entries.filter(([n]) => !n.includes(">"));
    const slowest = topLevel.sort((a, b) => b[1] - a[1])[0];
    const share = slowest && total > 0 ? Math.round((slowest[1] / total) * 100) : 0;
    const untracked = total - topLevel.reduce((acc, [, ms]) => acc + ms, 0);

    // Diagnosis is structural (where the time went), not a specific fault.
    let diagnosis = "ok";
    if (total >= slowThresholdMs) {
      if (this.onnxWaitMs > this.onnxRunMs && this.onnxWaitMs > total * 0.3) {
        diagnosis = "onnx-queue-contention (another job is holding the inference queue)";
      } else if (this.onnxRunMs > total * 0.5) {
        diagnosis = `onnx-compute-bound (${this.onnxCalls} inferences)`;
      } else if (untracked > total * 0.4) {
        diagnosis = "untracked-time (I/O, Qdrant or event-loop blocking outside spans)";
      } else if (slowest) {
        diagnosis = `stage-bound (${slowest[0]})`;
      }
    }

    const line =
      `${this.tag} total=${fmt(total)}` +
      ` onnx(calls=${this.onnxCalls} wait=${fmt(this.onnxWaitMs)} run=${fmt(this.onnxRunMs)} maxQueue=${this.onnxMaxQueueDepth})` +
      (stageText ? ` | ${stageText}` : "") +
      ` | untracked=${fmt(Math.max(0, untracked))}` +
      (slowest ? ` slowest=${slowest[0]}(${share}%)` : "") +
      ` diagnosis=${diagnosis}` +
      (this.context ? ` | ${this.context}` : "");
    if (total >= slowThresholdMs) warn("timing", line);
    else log("timing", line);
  }
}

const profilerStore = new AsyncLocalStorage<StageProfiler>();

/** Runs `fn` under a new profiler and emits its report when it settles. */
export async function runProfiled<T>(tag: string, fn: () => Promise<T>): Promise<T> {
  const profiler = new StageProfiler(tag);
  try {
    return await profilerStore.run(profiler, fn);
  } finally {
    profiler.report();
  }
}

/** Active profiler of the current async context, if any. */
export function currentProfiler(): StageProfiler | undefined {
  return profilerStore.getStore();
}

/** Times `fn` as stage `name` of the active profiler (no-op outside one). */
export async function span<T>(name: string, fn: () => Promise<T>): Promise<T> {
  const profiler = profilerStore.getStore();
  if (!profiler) return fn();
  const t0 = performance.now();
  try {
    return await fn();
  } finally {
    profiler.add(name, performance.now() - t0);
  }
}
