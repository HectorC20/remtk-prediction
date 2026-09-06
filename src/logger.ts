/** Logger compacto con timestamp, equivalente al log estándar del server Go. */
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
