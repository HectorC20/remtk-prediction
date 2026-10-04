/**
 * Mejora de arquitectura (EXPERIMENTAL, no toca producción): capa de
 * DESCOMPOSICIÓN DE INTENCIÓN que envuelve `/predict` de remtk-prediction.
 *
 * Ataca las carencias medidas en el bench (10/19):
 *
 *   [turno]
 *     │ E0 · Enmascarado de código/citas técnicas  → `secuestro-lexico-codigo`
 *     │      (los nombres de tools dentro de `...` no se fijan como explícitos)
 *     ▼
 *   E1 · Segmentación en cláusulas (puntuación + marcadores de discurso)
 *     ▼
 *   E2 · Rol por cláusula: AFIRMA | NIEGA | RETRACTA | IRONIA | CONDICION
 *        (pista de discurso determinística → si no hay, arquetipos por coseno)
 *     ▼
 *   E3 · Recencia: RETRACTA anula lo anterior; IRONIA cuenta como negación
 *     ▼
 *   E4 · /predict POR CLÁUSULA accionable (sesión aislada; reintento con las
 *        cláusulas previas como historial si la cláusula es anafórica)
 *     ▼
 *   E5 · Lista de EXCLUSIÓN: lo que piden las cláusulas negadas/retractadas
 *        se resta del resultado
 *     ▼
 *   E6 · Abstención tri-estado: ∅ solo si no queda intención afirmada;
 *        si hay intención afirmada pero /predict se abstuvo → "ambiguo":
 *        top-1 por coseno contra la firma del catálogo (con margen)
 *     ▼
 *   [tools]
 *
 * Turnos de UNA sola cláusula afirmativa se delegan intactos al pipeline
 * actual (no se re-arbitra lo que ya funciona: anáfora, confirmación,
 * cambio de tema, meta-pregunta, small-talk).
 *
 *   pnpm tsx bench/mejora-arquitectura.ts      → baseline vs mejora sobre BENCH_CASES
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";
import { BENCH_TOOLS } from "./catalog";
import { BENCH_CASES, type BenchCase, type BenchTurn } from "./cases";
import { RemtkBenchClient } from "./remtk.client";

// ── Tipos ────────────────────────────────────────────────────────────────────

export type RolClausula = "AFIRMA" | "NIEGA" | "RETRACTA" | "IRONIA" | "CONDICION";

export interface Clausula {
  texto: string;
  rol: RolClausula;
  /** Origen de la decisión de rol (para depurar). */
  via: "pista" | "coseno" | "defecto";
  tools: string[];
  estado?: "resuelto" | "ambiguo" | "vacio";
}

export interface ResultadoMejora {
  tools: string[];
  clausulas: Clausula[];
  excluidas: string[];
  modo: "delegado" | "descompuesto";
}

// ── E0 · Enmascarado ─────────────────────────────────────────────────────────

const CODIGO_RE = /```[\s\S]*?```|`[^`\n]*`/g;

/** Sustituye bloques de código por un marcador neutro. */
export function enmascarar(texto: string): string {
  return texto.replace(CODIGO_RE, "[código]");
}

// ── E1 · Segmentación ────────────────────────────────────────────────────────

/**
 * Marcadores que abren una cláusula nueva. Se corta ANTES del marcador y el
 * marcador queda al inicio de la cláusula siguiente (es la pista de su rol).
 */
const MARCADORES_CORTE = [
  ", pero ",
  " pero ",
  ", y luego ",
  " y luego ",
  ", luego ",
  ", solo ",
  ", mejor ",
  ", lo que realmente ",
  ", lo que necesito ",
  ", obviamente ",
  "... ",
];

export function segmentar(texto: string): string[] {
  let partes = texto
    .split(/(?<=[.!?;])\s+|\.\.\.\s*/)
    .map((s) => s.trim())
    .filter(Boolean);

  for (const m of MARCADORES_CORTE) {
    partes = partes.flatMap((p) => {
      const out: string[] = [];
      let resto = p;
      let idx = resto.toLowerCase().indexOf(m);
      while (idx > 0) {
        out.push(resto.slice(0, idx));
        resto = resto.slice(idx + (m.startsWith("...") ? m.length : 1));
        idx = resto.toLowerCase().indexOf(m, 1);
      }
      out.push(resto);
      return out;
    });
  }

  // " y " + verbo: solo corta si el lado derecho tiene ≥3 palabras (evita
  // partir sintagmas nominales tipo "ventas y compras").
  partes = partes.flatMap((p) => {
    const i = p.toLowerCase().lastIndexOf(" y ");
    if (i <= 0) return [p];
    const der = p.slice(i + 3).trim();
    return der.split(/\s+/).length >= 3 ? [p.slice(0, i), der] : [p];
  });

  return partes
    .map((s) => s.replace(/^[,;:\s]+|[,;:\s]+$/g, "").trim())
    .filter((s) => s.replace(/[^\p{L}]/gu, "").length >= 3);
}

// ── E2 · Rol por cláusula ────────────────────────────────────────────────────

const RELLENO_INICIAL = /^(pero|y|luego|ahora|por favor|bueno|entonces|,|\s)+/i;

/** Pistas de discurso (determinísticas, documentadas). */
function rolPorPista(clausula: string, posicion: number, total: number): RolClausula | undefined {
  const t = clausula.toLowerCase().replace(RELLENO_INICIAL, "").trim();
  if (/^(no dejes de|no olvides|no te olvides de)\b/.test(t)) return "AFIRMA";
  if (/^(espera|ignora|olvida|olvídalo|pensándolo bien|pensandolo bien|cancela|retiro)\b/.test(t)) {
    // "espera, no, no generes nada" / "ignora ese código" → anula lo previo
    return "RETRACTA";
  }
  if (/^mejor\s+(déjalo|dejalo|no)\b/.test(t) || /^(déjalo|dejalo) (ahí|ahi|así|asi|para después)/.test(t)) {
    return "NIEGA";
  }
  if (total > 1 && posicion < total - 1 && /^(sí|si),?\s+claro\b/.test(t)) return "IRONIA";
  if (/^(no|ni|nunca|tampoco)\b/.test(t)) return "NIEGA";
  if (/^si\s/.test(t)) return /\bno hagas (nada|eso)\b/.test(t) ? "NIEGA" : "CONDICION";
  return undefined;
}

const ARQUETIPOS: Record<"AFIRMA" | "NIEGA", string[]> = {
  AFIRMA: [
    "hazlo, ejecuta la acción ahora",
    "quiero que realices esta tarea",
    "guárdalo, envíalo, descárgalo, búscalo",
  ],
  NIEGA: [
    "no lo hagas, no ejecutes esa acción",
    "no toques eso, déjalo como está",
    "prohibido realizar esa tarea",
  ],
};

// ── Utilidades vectoriales ───────────────────────────────────────────────────

function cos(a: Float32Array, b: Float32Array): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

// ── Capa de mejora ───────────────────────────────────────────────────────────

export class MejoraArquitectura {
  private arquetipos?: Record<"AFIRMA" | "NIEGA", Float32Array[]>;
  private firmas?: { name: string; v: Float32Array }[];
  private seq = 0;

  /** Margen mínimo (coseno) para decidir rol por arquetipos. */
  static readonly MARGEN_ROL = 0.02;
  /** Margen top−media para emitir "ambiguo" en vez de ∅. */
  static readonly MARGEN_AMBIGUO = 0.04;

  constructor(
    private readonly remtk: RemtkBenchClient,
    private readonly engine: EmbeddingEngineService,
    private readonly catalogo: ToolDefinition[],
  ) {}

  private async vq(t: string) {
    return (await this.engine.embedQuery(t, undefined, { high: true })).embedding;
  }

  private async rolPorCoseno(clausula: string): Promise<RolClausula | undefined> {
    if (!this.arquetipos) {
      this.arquetipos = { AFIRMA: [], NIEGA: [] };
      for (const k of ["AFIRMA", "NIEGA"] as const) {
        for (const a of ARQUETIPOS[k]) this.arquetipos[k].push(await this.vq(a));
      }
    }
    const v = await this.vq(clausula);
    const best = (k: "AFIRMA" | "NIEGA") => Math.max(...this.arquetipos![k].map((a) => cos(v, a)));
    const af = best("AFIRMA");
    const ng = best("NIEGA");
    if (ng - af > MejoraArquitectura.MARGEN_ROL) return "NIEGA";
    return undefined;
  }

  /** E6 · ranking propio contra la firma del catálogo (solo para "ambiguo"). */
  private async rankingAmbiguo(clausula: string): Promise<string | undefined> {
    if (!this.firmas) {
      this.firmas = [];
      for (const t of this.catalogo) {
        const firma = [t.intentSummary, t.description, (t.tags ?? []).join(" ")].filter(Boolean).join(". ");
        const v = (await this.engine.embedPassage(firma, undefined, { high: true })).embedding;
        this.firmas.push({ name: t.name, v });
      }
    }
    const q = await this.vq(clausula);
    const s = this.firmas.map((f) => ({ name: f.name, s: cos(q, f.v) })).sort((a, b) => b.s - a.s);
    const media = s.reduce((a, x) => a + x.s, 0) / s.length;
    return s[0].s - media >= MejoraArquitectura.MARGEN_AMBIGUO ? s[0].name : undefined;
  }

  private async predecirClausula(
    caseId: string,
    texto: string,
    previas: string[],
    historial: BenchTurn[],
  ): Promise<string[]> {
    const id = () => `mejora-${caseId}-${++this.seq}`;
    const sola = await this.remtk.predict(id(), texto, historial);
    if (sola.tools.length > 0 || previas.length === 0) return sola.tools;
    // Cláusula anafórica ("guárdalo para mañana"): se le da el resto del
    // mensaje como contexto; la exclusión (E5) limpia lo que arrastre.
    const ctx: BenchTurn[] = [...historial, { role: "user", content: previas.join(". ") }];
    return (await this.remtk.predict(id(), texto, ctx)).tools;
  }

  async resolver(caseId: string, texto: string, historial: BenchTurn[]): Promise<ResultadoMejora> {
    const limpio = enmascarar(texto);
    const crudas = segmentar(limpio);

    // E2 · rol
    const clausulas: Clausula[] = [];
    for (let i = 0; i < crudas.length; i++) {
      const pista = rolPorPista(crudas[i], i, crudas.length);
      if (pista) {
        clausulas.push({ texto: crudas[i], rol: pista, via: "pista", tools: [] });
        continue;
      }
      const c = crudas.length > 1 ? await this.rolPorCoseno(crudas[i]) : undefined;
      clausulas.push({ texto: crudas[i], rol: c ?? "AFIRMA", via: c ? "coseno" : "defecto", tools: [] });
    }

    // Turno simple afirmativo → pipeline actual intacto (sin re-arbitrar).
    if (clausulas.length <= 1 && (clausulas[0]?.rol ?? "AFIRMA") === "AFIRMA") {
      const r = await this.remtk.predict(`mejora-${caseId}-${++this.seq}`, limpio, historial);
      return {
        tools: r.tools,
        clausulas: clausulas.map((c) => ({ ...c, tools: r.tools, estado: r.tools.length ? "resuelto" : "vacio" })),
        excluidas: [],
        modo: "delegado",
      };
    }

    // E3 · recencia: lo anterior a la última RETRACTA queda anulado.
    const ultRetracta = clausulas.map((c) => c.rol).lastIndexOf("RETRACTA");

    // E4 · predicción por cláusula
    const previas: string[] = [];
    for (let i = 0; i < clausulas.length; i++) {
      const c = clausulas[i];
      if (c.rol === "RETRACTA" && c.texto.split(/\s+/).length <= 3) {
        previas.push(c.texto);
        continue; // "espera, no" / "ignora ese código": sin contenido accionable
      }
      c.tools = await this.predecirClausula(caseId, c.texto, previas, historial);
      c.estado = c.tools.length ? "resuelto" : "vacio";
      previas.push(c.texto);
    }

    const anulada = (i: number) => i < ultRetracta;
    const afirmada = (c: Clausula, i: number) =>
      !anulada(i) && (c.rol === "AFIRMA" || c.rol === "CONDICION");

    // E5 · exclusión: top-2 de lo negado/ironizado/retractado/anulado, salvo
    // lo que una cláusula afirmada eligió como top-1 (la afirmación manda).
    const protegidas = new Set(
      clausulas.filter((c, i) => afirmada(c, i) && c.tools[0]).map((c) => c.tools[0]),
    );
    const excl = new Set<string>();
    clausulas.forEach((c, i) => {
      if (!afirmada(c, i)) c.tools.slice(0, 2).forEach((t) => !protegidas.has(t) && excl.add(t));
    });

    // E6 · abstención tri-estado
    for (let i = 0; i < clausulas.length; i++) {
      const c = clausulas[i];
      if (c.rol === "AFIRMA" && afirmada(c, i) && c.tools.filter((t) => !excl.has(t)).length === 0) {
        const amb = await this.rankingAmbiguo(c.texto);
        if (amb && !excl.has(amb)) {
          c.tools = [amb];
          c.estado = "ambiguo";
        }
      }
    }

    // Ensamblado: top-1 de cada cláusula afirmada primero (en orden), luego
    // los segundos (solo AFIRMA; una CONDICION aporta su inspección, top-1).
    const out: string[] = [];
    const push = (t?: string) => {
      if (t && !excl.has(t) && !out.includes(t)) out.push(t);
    };
    const vivas = clausulas.filter(afirmada);
    for (const c of vivas) push(c.tools.find((t) => !excl.has(t) && !out.includes(t)));
    for (const c of vivas) {
      if (c.rol !== "AFIRMA") continue;
      push(c.tools.filter((t) => !excl.has(t) && !out.includes(t))[0]);
    }

    return { tools: out, clausulas, excluidas: [...excl], modo: "descompuesto" };
  }
}

// ── Puntuación (idéntica a bench/run.ts › scoreRemtk) ────────────────────────

export function aprueba(c: BenchCase, tools: string[]): boolean {
  const picked = tools.slice(0, 3);
  if (c.expected.length === 0) return tools.length === 0;
  if (c.multi) return c.expected.every((e) => picked.includes(e));
  return tools[0] === c.expected[0] || (c.expected.includes(tools[0]) && c.expected.length > 1);
}

/** Juega los turnos previos del caso y devuelve (historial, último turno de usuario). */
export function partirCaso(c: BenchCase): { historial: BenchTurn[]; texto: string } {
  const idx = c.turns.map((t) => t.role).lastIndexOf("user");
  return { historial: c.turns.slice(0, idx), texto: c.turns[idx].content };
}

export async function crearMejora(): Promise<{ mejora: MejoraArquitectura; remtk: RemtkBenchClient }> {
  const remtk = new RemtkBenchClient();
  await remtk.start(BENCH_TOOLS, { qdrant: false });
  const engine = new EmbeddingEngineService(loadConfig());
  return { mejora: new MejoraArquitectura(remtk, engine, BENCH_TOOLS), remtk };
}

// ── CLI: baseline vs mejora ──────────────────────────────────────────────────

async function main(): Promise<void> {
  const { mejora, remtk } = await crearMejora();
  console.log(`[mejora] remtk listo, modo=${remtk.mode}\n`);

  const filas: { id: string; esperado: string[]; base: string[]; mejora: string[]; okB: boolean; okM: boolean; r: ResultadoMejora }[] = [];
  for (const c of BENCH_CASES) {
    const { historial, texto } = partirCaso(c);
    // Baseline: mismo flujo que bench/run.ts (turnos completos, sesión propia).
    const hist: BenchTurn[] = [];
    let base: string[] = [];
    for (const t of c.turns) {
      if (t.role === "user") base = (await remtk.predict(`base-${c.id}`, t.content, [...hist])).tools;
      hist.push(t);
    }
    const r = await mejora.resolver(c.id, texto, historial);
    const fila = { id: c.id, esperado: c.expected, base, mejora: r.tools, okB: aprueba(c, base), okM: aprueba(c, r.tools), r };
    filas.push(fila);
    const m = (ok: boolean) => (ok ? "✓" : "✗");
    console.log(
      `${m(fila.okB)} base  ${m(fila.okM)} mejora  ${c.id.padEnd(28)} esperado=[${c.expected.join(",") || "∅"}] ` +
        `base=[${base.join(",")}] mejora=[${r.tools.join(",")}] (${r.modo})`,
    );
    if (r.modo === "descompuesto") {
      for (const cl of r.clausulas) {
        console.log(`      · ${cl.rol.padEnd(9)} (${cl.via}) "${cl.texto}" → [${cl.tools.join(",")}]${cl.estado === "ambiguo" ? " ambiguo" : ""}`);
      }
      if (r.excluidas.length) console.log(`      · excluidas: ${r.excluidas.join(",")}`);
    }
  }
  await remtk.close();

  const okB = filas.filter((f) => f.okB).length;
  const okM = filas.filter((f) => f.okM).length;
  console.log(`\n── resumen ──\nbaseline: ${okB}/${filas.length}   mejora: ${okM}/${filas.length}`);
  const reg = filas.filter((f) => f.okB && !f.okM).map((f) => f.id);
  if (reg.length) console.log(`regresiones: ${reg.join(", ")}`);

  const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "results");
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `mejora-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
  fs.writeFileSync(file, JSON.stringify({ baseline: okB, mejora: okM, total: filas.length, casos: filas }, null, 2));
  console.log(`[mejora] reporte → ${file}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error("[mejora] fallo fatal:", e);
    process.exit(1);
  });
}
