/**
 * Runner del benchmark de contexto: remtk-prediction vs Jev (System One).
 *
 *   pnpm bench              → ambos lados (Jev solo si hay EXPERIENTIAL_API_KEY)
 *   pnpm bench -- --remtk   → solo remtk (gratis, local)
 *   pnpm bench -- --jev     → solo Jev (cobra por request: 1 por caso)
 *   pnpm bench -- --qdrant  → remtk con Qdrant habilitado
 *
 * Escribe el detalle en bench/results/results-<timestamp>.json.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { BENCH_TOOLS } from "./catalog";
import { BENCH_CASES, type BenchCase, type BenchTurn } from "./cases";
import { jevChoice, NONE_KEY } from "./jev.client";
import { RemtkBenchClient } from "./remtk.client";

const FLAGS = new Set(process.argv.slice(2));
const WANT_REMTK = FLAGS.has("--remtk") || !FLAGS.has("--jev");
const WANT_JEV = FLAGS.has("--jev") || !FLAGS.has("--remtk");
const WITH_QDRANT = FLAGS.has("--qdrant");

interface SideResult {
  picked: string[];
  pass: boolean;
  partial?: boolean;
  latencyMs: number;
  detail: string;
}

interface CaseReport {
  id: string;
  why: string;
  expected: string[];
  remtk?: SideResult;
  jev?: SideResult;
}

function scoreRemtk(c: BenchCase, tools: string[], latencyMs: number, detail: string): SideResult {
  const picked = tools.slice(0, 3);
  let pass: boolean;
  if (c.expected.length === 0) {
    pass = tools.length === 0;
  } else if (c.multi) {
    pass = c.expected.every((e) => picked.includes(e));
  } else {
    pass = tools[0] === c.expected[0] || (c.expected.includes(tools[0]) && c.expected.length > 1);
  }
  return { picked, pass, latencyMs, detail };
}

function scoreJev(c: BenchCase, choice: string | undefined, latencyMs: number, detail: string): SideResult {
  const picked = choice ? [choice] : [];
  let pass = false;
  let partial = false;
  if (c.expected.length === 0) {
    pass = choice === NONE_KEY;
  } else if (choice && c.expected.includes(choice)) {
    pass = true;
    partial = c.multi === true && c.expected.length > 1;
  }
  return { picked, pass, partial, latencyMs, detail };
}

async function runRemtkCase(client: RemtkBenchClient, c: BenchCase): Promise<SideResult> {
  const sessionId = `bench-${c.id}`;
  const history: BenchTurn[] = [];
  let last: Awaited<ReturnType<RemtkBenchClient["predict"]>> | undefined;
  for (const turn of c.turns) {
    if (turn.role === "user") {
      last = await client.predict(sessionId, turn.content, [...history]);
    }
    history.push(turn);
  }
  const tools = last?.tools ?? [];
  return scoreRemtk(c, tools, last?.latencyMs ?? 0, tools.join(",") || "(vacío)");
}

async function runJevCase(c: BenchCase): Promise<SideResult> {
  const lastUser = [...c.turns].reverse().find((t) => t.role === "user");
  const idx = c.turns.lastIndexOf(lastUser!);
  const out = await jevChoice({ history: c.turns.slice(0, idx), text: lastUser!.content, tools: BENCH_TOOLS });
  if (!out.ok) {
    return { picked: [], pass: false, latencyMs: out.latencyMs, detail: `ERROR: ${out.error}` };
  }
  return scoreJev(c, out.choice, out.latencyMs, `${out.choice} (conf ${out.confidence ?? "?"})`);
}

function mark(r: SideResult | undefined): string {
  if (!r) return "  - ";
  if (r.detail.startsWith("ERROR")) return " ERR";
  return r.pass ? (r.partial ? " ~  " : " ✓ ") : " ✗ ";
}

async function main(): Promise<void> {
  const reports: CaseReport[] = [];
  const remtk = new RemtkBenchClient();
  let remtkMode = "off";

  if (WANT_REMTK) {
    console.log(`[bench] levantando remtk-prediction (qdrant=${WITH_QDRANT})…`);
    await remtk.start(BENCH_TOOLS, { qdrant: WITH_QDRANT });
    remtkMode = remtk.mode;
    console.log(`[bench] remtk listo, modo=${remtkMode}`);
  }
  if (WANT_JEV && !process.env.EXPERIENTIAL_API_KEY) {
    console.log("[bench] EXPERIENTIAL_API_KEY ausente: lado Jev omitido (usa --remtk o configura .env)");
  }
  const jevOn = WANT_JEV && !!process.env.EXPERIENTIAL_API_KEY;

  for (const c of BENCH_CASES) {
    const report: CaseReport = { id: c.id, why: c.why, expected: c.expected };
    if (WANT_REMTK) report.remtk = await runRemtkCase(remtk, c);
    if (jevOn) report.jev = await runJevCase(c);
    reports.push(report);
    console.log(
      `${mark(report.remtk)} remtk  ${mark(report.jev)} jev  ${c.id.padEnd(22)} ` +
        `esperado=[${c.expected.join(",") || "∅"}]  remtk=${report.remtk?.detail ?? "-"}  ` +
        `jev=${report.jev?.detail ?? "-"}`,
    );
  }

  await remtk.close();

  const summarize = (side: "remtk" | "jev") => {
    const rows = reports.map((r) => r[side]).filter((r): r is SideResult => !!r);
    const ok = rows.filter((r) => r.pass).length;
    const err = rows.filter((r) => r.detail.startsWith("ERROR")).length;
    const ms = rows.length ? Math.round(rows.reduce((a, r) => a + r.latencyMs, 0) / rows.length) : 0;
    return { total: rows.length, pass: ok, errors: err, avgLatencyMs: ms };
  };
  const summary = { remtkMode, remtk: summarize("remtk"), jev: summarize("jev") };

  console.log("\n── resumen ─────────────────────────────────────────────");
  if (WANT_REMTK) {
    console.log(
      `remtk (${remtkMode}): ${summary.remtk.pass}/${summary.remtk.total} casos, ` +
        `latencia media ${summary.remtk.avgLatencyMs}ms`,
    );
  }
  if (jevOn) {
    console.log(
      `jev: ${summary.jev.pass}/${summary.jev.total} casos (${summary.jev.errors} errores), ` +
        `latencia media ${summary.jev.avgLatencyMs}ms`,
    );
  }

  const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "results");
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `results-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
  fs.writeFileSync(file, JSON.stringify({ summary, cases: reports }, null, 2));
  console.log(`\n[bench] reporte → ${file}`);
}

main().catch((err) => {
  console.error("[bench] fallo fatal:", err);
  process.exit(1);
});
