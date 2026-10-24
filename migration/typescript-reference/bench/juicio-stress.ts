/**
 * Banco de esfuerzo de la capa de Juicio: escala la complejidad del texto por
 * ejes hasta encontrar el PUNTO DE QUIEBRE de cada componente. No compite con
 * Jev; probea SOLO remtk-prediction (con juicio activo) y reporta las señales
 * internas por turno (estado NLI, λ de puerta, noop, energía, señales de
 * re-rank) junto al RANK del tool esperado, para ver dónde degrada.
 *
 *   pnpm bench:stress
 *
 * Cada eje es una escalera: el nivel donde el rank pasa de 1 a >1, o donde una
 * abstención correcta/incorrecta se dispara, es el quiebre medido.
 */
import { BENCH_TOOLS } from "./catalog";
import type { BenchTurn } from "./cases";
import { RemtkBenchClient } from "./remtk.client";

/** Relleno neutro: párrafos sin señal de tool que diluyen la consulta real. */
const FILLER = [
  "He estado revisando algunas notas sueltas del proyecto esta mañana mientras tomaba café.",
  "La verdad es que el equipo viene comentando mucho sobre la carga de trabajo de las últimas semanas.",
  "Ayer por la tarde tuve una reunión larga que se extendió bastante más de lo previsto.",
  "No sé si importa, pero el clima de la oficina estaba algo pesado y todos andábamos distraídos.",
  "Mi compañero de al lado me contaba una anécdota curiosa sobre un cliente antiguo.",
  "En fin, son cosas menores que uno va acumulando a lo largo del día sin darle demasiada vuelta.",
  "También estaba pensando en reorganizar un par de carpetas que tengo desordenadas desde hace tiempo.",
  "El caso es que la jornada se me está yendo entre pequeñas distracciones sin casi darme cuenta.",
];

function ramble(n: number): string {
  const parts: string[] = [];
  for (let i = 0; i < n; i++) parts.push(FILLER[i % FILLER.length]);
  return parts.join(" ");
}

type Expect = { tools?: string[]; empty?: boolean; all?: boolean };

interface StressCase {
  axis: string;
  id: string;
  note: string;
  turns: BenchTurn[];
  expect: Expect;
}

const U = (content: string): BenchTurn => ({ role: "user", content });
const A = (content: string): BenchTurn => ({ role: "assistant", content });

const CASES: StressCase[] = [
  // ── EJE A · LONGITUD (intención única sepultada en texto extenso) ──────────
  {
    axis: "A·longitud",
    id: "A1-corto",
    note: "referencia: petición directa",
    turns: [U("descarga el reporte de ventas de ayer")],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A2-medio",
    note: "la petición al final tras ~4 frases",
    turns: [U(`${ramble(4)} En resumen, descarga el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A3-largo-10",
    note: "la petición al final tras ~10 frases (~150 palabras)",
    turns: [U(`${ramble(10)} Lo que necesito realmente es que descargues el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A4-largo-13",
    note: "la petición al final tras ~13 frases",
    turns: [U(`${ramble(13)} Lo que necesito realmente es que descargues el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A5-largo-16",
    note: "la petición al final tras ~16 frases",
    turns: [U(`${ramble(16)} Lo que necesito realmente es que descargues el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A6-largo-19",
    note: "la petición al final tras ~19 frases",
    turns: [U(`${ramble(19)} Lo que necesito realmente es que descargues el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A7-muylargo-22",
    note: "la petición al final tras ~22 frases (~330 palabras)",
    turns: [U(`${ramble(22)} Al final, lo único que necesito es descargar el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A8-largo-26",
    note: "la petición al final tras ~26 frases",
    turns: [U(`${ramble(26)} Al final, lo único que necesito es descargar el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "A·longitud",
    id: "A9-extremo",
    note: "la petición SEPULTADA en el centro tras ~35 frases",
    turns: [U(`Antes de nada, un poco de contexto. ${ramble(18)} Necesito descargar el reporte de ventas de ayer cuanto antes. ${ramble(17)} Eso es todo por ahora.`)],
    expect: { tools: ["download_report"] },
  },

  // ── EJE B · MULTI-INTENCIÓN (nº de acciones encadenadas) ───────────────────
  {
    axis: "B·multi-intención",
    id: "B1-una",
    note: "1 acción",
    turns: [U("descarga el reporte de ventas")],
    expect: { tools: ["download_report"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B2-dos",
    note: "2 acciones",
    turns: [U("descarga el reporte de ventas y envíamelo por correo")],
    expect: { tools: ["download_report", "send_email"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B3-tres",
    note: "3 acciones",
    turns: [U("descarga el reporte de ventas, consulta a cuánto está el dólar y me lo envías por correo")],
    expect: { tools: ["download_report", "get_exchange_rate", "send_email"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B4-cuatro",
    note: "4 acciones",
    turns: [U(
      "descarga el reporte de ventas, mira el clima de mañana en Lima, consulta el tipo de cambio del dólar " +
        "y mándame todo eso resumen por correo",
    )],
    expect: { tools: ["download_report", "get_weather", "get_exchange_rate", "send_email"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B4b-cuatro-puntuado",
    note: "4 acciones con delimitadores claros (uno/dos/tres/cuatro)",
    turns: [U(
      "Cuatro cosas, una por una: uno, descarga el reporte de ventas; dos, mira el clima de mañana en Lima; " +
        "tres, consulta el tipo de cambio del dólar; cuatro, mándame el resumen por correo.",
    )],
    expect: { tools: ["download_report", "get_weather", "get_exchange_rate", "send_email"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B5-cinco",
    note: "5 acciones (límite teórico)",
    turns: [U(
      "descarga el reporte de ventas, busca en la web el precio del petróleo, mira el clima de Madrid, " +
        "pasa el informe al inglés y guárdame el resumen como borrador de correo",
    )],
    expect: { tools: ["download_report", "web_search", "get_weather", "translate_text", "save_draft"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B5b-cinco-encadenado",
    note: "5 acciones encadenadas con 'y luego'",
    turns: [U(
      "descarga el reporte de ventas, y luego busca en la web el precio del petróleo, y luego mira el clima " +
        "de Madrid, y luego pasa el informe al inglés, y luego guárdame el resumen como borrador de correo",
    )],
    expect: { tools: ["download_report", "web_search", "get_weather", "translate_text", "save_draft"], all: true },
  },
  {
    axis: "B·multi-intención",
    id: "B6-seis",
    note: "6 acciones (por encima del límite)",
    turns: [U(
      "descarga el reporte de ventas, y luego busca en la web el precio del petróleo, y luego mira el clima de " +
        "Madrid, y luego consulta el dólar, y luego pasa el informe al inglés, y luego guárdame todo como borrador",
    )],
    expect: { tools: ["download_report", "web_search", "get_weather", "get_exchange_rate", "translate_text", "save_draft"], all: true },
  },

  // ── EJE C · NEGACIÓN A DISTANCIA (cross-encoder OFF: probea el límite) ─────
  {
    axis: "C·negación",
    id: "C1-cerca",
    note: "negación junto al verbo",
    turns: [U("No envíes el correo todavía, mejor guárdalo como borrador")],
    expect: { tools: ["save_draft"] },
  },
  {
    axis: "C·negación",
    id: "C2-media",
    note: "la negación separada por una cláusula",
    turns: [U(
      "He pensado que el asunto puede esperar a mañana. Así que NO lo envíes ya; en su lugar guárdalo como borrador.",
    )],
    expect: { tools: ["save_draft"] },
  },
  {
    axis: "C·negación",
    id: "C3-lejos",
    note: "negación y acción separadas por ~8 frases",
    turns: [U(
      `Un correo con el balance trimestral para el equipo. ${ramble(8)} ` +
        "Eso sí, por ahora NO lo envíes: déjalo solo guardado como borrador.",
    )],
    expect: { tools: ["save_draft"] },
  },
  {
    axis: "C·negación",
    id: "C4-contradicción",
    note: "auto-corrección: 'envíalo, no, mejor guárdalo'",
    turns: [U("Prepara el correo y envíalo… no, espera, mejor no lo envíes, guárdalo únicamente como borrador")],
    expect: { tools: ["save_draft"] },
  },

  // ── EJE D · CAMBIO DE TEMA TRAS CONTEXTO LARGO (puerta λ) ──────────────────
  {
    axis: "D·cambio-tema",
    id: "D1-corto",
    note: "1 mensaje de contexto y pivot",
    turns: [
      U("envía la factura al cliente por correo"),
      A("Correo enviado al cliente."),
      U("ahora, ¿a cuánto está el dólar hoy?"),
    ],
    expect: { tools: ["get_exchange_rate"] },
  },
  {
    axis: "D·cambio-tema",
    id: "D1b-den2",
    note: "pivot tras contexto con 2 frases de relleno",
    turns: [
      U("envía la factura al cliente por correo"),
      A("Correo enviado. ¿Quieres que prepare la siguiente?"),
      U(`Sobre los correos: ${ramble(2)} Recuérdame pasarlas a PDF.`),
      A("Anotado, las paso a PDF."),
      U("Genial. Cambiando de tema por completo, ¿qué clima hace mañana en Bilbao?"),
    ],
    expect: { tools: ["get_weather"] },
  },
  {
    axis: "D·cambio-tema",
    id: "D1c-den4",
    note: "pivot tras contexto con 4 frases de relleno",
    turns: [
      U("envía la factura al cliente por correo"),
      A("Correo enviado. ¿Quieres que prepare la siguiente?"),
      U(`Sobre los correos: ${ramble(4)} Recuérdame pasarlas a PDF.`),
      A("Anotado, las paso a PDF."),
      U("Genial. Cambiando de tema por completo, ¿qué clima hace mañana en Bilbao?"),
    ],
    expect: { tools: ["get_weather"] },
  },
  {
    axis: "D·cambio-tema",
    id: "D2-largo-6",
    note: "contexto extenso de correo (~6 frases) y pivot a clima",
    turns: [
      U("envía la factura al cliente por correo"),
      A("Correo enviado. ¿Quieres que prepare la siguiente?"),
      U(`Sobre los correos: ${ramble(6)} Recuérdame pasarlas a PDF.`),
      A("Anotado, las paso a PDF."),
      U("Genial. Cambiando de tema por completo, ¿qué clima hace mañana en Bilbao?"),
    ],
    expect: { tools: ["get_weather"] },
  },
  {
    axis: "D·cambio-tema",
    id: "D3-extremo",
    note: "contexto muy denso (3 turnos largos) y pivot breve a tipo de cambio",
    turns: [
      U(`Revisemos los envíos por correo de esta semana. ${ramble(10)}`),
      A(`${ramble(6)} Queda un envío pendiente al proveedor.`),
      U(`Vale, y respecto a ese proveedor ${ramble(8)} confirma que le llegó el último correo.`),
      A("Confirmado, recibido."),
      U("Oye, ¿y el dólar?"),
    ],
    expect: { tools: ["get_exchange_rate"] },
  },

  // ── EJE E · CODE-SWITCHING + LONGITUD ──────────────────────────────────────
  {
    axis: "E·idioma",
    id: "E1-es",
    note: "español puro (referencia)",
    turns: [U("envía el reporte mensual al equipo por correo")],
    expect: { tools: ["send_email"] },
  },
  {
    axis: "E·idioma",
    id: "E2-spanglish",
    note: "spanglish con la acción en inglés",
    turns: [U("Mira, el resumen ya está listo, so please send it to my team by email cuando puedas")],
    expect: { tools: ["send_email"] },
  },
  {
    axis: "E·idioma",
    id: "E3-mix-largo",
    note: "mezcla es/en/fr + relleno largo",
    turns: [U(
      `Contexto del día: ${ramble(8)} Anyway, il faudrait générer le document, et por favor ` +
        "download the sales report and send it by email. Merci!",
    )],
    expect: { tools: ["download_report"] },
  },

  // ── EJE F · LÍMITE DE ABSTENCIÓN (¿cuándo NO devolver tools?) ──────────────
  {
    axis: "F·abstención",
    id: "F0b-div2",
    note: "divagación corta sin acción → debe abstener",
    turns: [U(`${ramble(2)} En fin, solo quería desahogarme un rato, no necesito nada concreto.`)],
    expect: { empty: true },
  },
  {
    axis: "F·abstención",
    id: "F0c-div6",
    note: "divagación media sin acción → debe abstener",
    turns: [U(`${ramble(6)} En fin, solo quería desahogarme un rato, no necesito nada concreto.`)],
    expect: { empty: true },
  },
  {
    axis: "F·abstención",
    id: "F1-sin-herramienta-12",
    note: "divagación larga sin acción → debe abstener",
    turns: [U(`${ramble(12)} En fin, solo quería desahogarme un rato, no necesito nada concreto.`)],
    expect: { empty: true },
  },
  {
    axis: "F·abstención",
    id: "F2-largo-pero-con-intencion",
    note: "muy largo PERO con acción real al final → NO debe abstener",
    turns: [U(`${ramble(14)} Dicho esto, sí necesito que descargues el reporte de ventas de ayer.`)],
    expect: { tools: ["download_report"] },
  },
  {
    axis: "F·abstención",
    id: "F3-pregunta-meta-larga",
    note: "pregunta sobre capacidades en texto largo → abstener",
    turns: [U(
      `Llevo un rato probando cosas. ${ramble(6)} ` +
        "En general, ¿qué herramientas y funciones tienes disponibles para ayudarme?",
    )],
    expect: { empty: true },
  },

  // ── EJE G · COREFERENCIA EN CADENA (estado de sesión) ──────────────────────
  {
    axis: "G·coreferencia",
    id: "G1-un-salto",
    note: "referencia a 1 turno atrás",
    turns: [
      U("lista mis facturas de junio"),
      A("Aquí están tus 8 facturas de junio."),
      U("muéstrame la última otra vez"),
    ],
    expect: { tools: ["list_invoices"] },
  },
  {
    axis: "G·coreferencia",
    id: "G2-dos-saltos",
    note: "referencia enterrada 2 turnos atrás",
    turns: [
      U("lista mis facturas de junio"),
      A("Aquí están tus 8 facturas de junio."),
      U(`Gracias. ${ramble(5)} Ahora coméntame un poco el contexto de ese mes.`),
      A(`Fue un mes movido. ${ramble(4)}`),
      U("¿me repites ese total de facturas de las que hablamos al principio?"),
    ],
    expect: { tools: ["list_invoices"] },
  },
];

interface Row {
  axis: string;
  id: string;
  expect: string;
  got: string;
  rank: number; // 1-based rank del primer tool esperado; 0 = ausente; -1 = esperaba ∅
  verdict: string;
  signals: string;
  pass: boolean;
  latencyMs: number;
}

function rankOf(returned: string[], target: string): number {
  const i = returned.indexOf(target);
  return i < 0 ? 0 : i + 1;
}

async function play(client: RemtkBenchClient, c: StressCase): Promise<Row> {
  const sessionId = `stress-${c.id}`;
  const history: BenchTurn[] = [];
  let last: Awaited<ReturnType<RemtkBenchClient["predict"]>> | undefined;
  for (const turn of c.turns) {
    if (turn.role === "user") last = await client.predict(sessionId, turn.content, [...history]);
    history.push(turn);
  }
  const got = last?.tools ?? [];
  const expEmpty = c.expect.empty === true;
  const expected = c.expect.tools ?? [];

  let rank = -1;
  let pass: boolean;
  if (expEmpty) {
    pass = got.length === 0;
  } else if (c.expect.all) {
    // multi: todos los esperados presentes en el conjunto devuelto
    pass = expected.every((e) => got.includes(e));
    rank = expected.length ? Math.max(...expected.map((e) => rankOf(got, e) || 99)) : 0;
  } else {
    rank = rankOf(got, expected[0]);
    pass = rank >= 1 && rank <= 3;
  }

  const trace = await client.lastTrace(sessionId);
  const sig: string[] = [];
  if (trace?.juicioEstado && trace.juicioEstado !== "neutral") sig.push(`nli=${trace.juicioEstado}`);
  if (typeof trace?.juicioGateLambda === "number") sig.push(`λ=${trace.juicioGateLambda.toFixed(2)}`);
  if (typeof trace?.juicioNoopScore === "number" && trace.juicioNoopScore > 0) {
    sig.push(`noop=${trace.juicioNoopScore.toFixed(2)}`);
  }
  if (trace?.juicioAbstained) sig.push(`abst=${trace.juicioAbstained}`);
  if (trace?.juicioSignals?.maxsim) sig.push("maxsim");
  if (trace?.juicioSignals?.specificity) sig.push("espec");

  return {
    axis: c.axis,
    id: c.id,
    expect: expEmpty ? "∅" : expected.join("+"),
    got: got.slice(0, 5).join(",") || "∅",
    rank,
    verdict: trace?.juicioAbstained ? "∅" : String(rank),
    signals: sig.join(" ") || "—",
    pass,
    latencyMs: last?.latencyMs ?? 0,
  };
}

function mark(r: Row): string {
  return r.pass ? " ✓ " : " ✗ ";
}

function renderAxis(rows: Row[]): void {
  const axis = rows[0].axis;
  console.log(`\n── ${axis} ${"─".repeat(60 - axis.length)}`);
  console.log("   caso                 esperado          →  obtuvo                                  rank  señales");
  for (const r of rows) {
    const rankTxt = r.rank === -1 ? (r.pass ? "∅ ok" : "∅?") : r.rank === 0 ? "fuera" : `#${r.rank}`;
    console.log(
      `   ${mark(r)} ${r.id.padEnd(28)} ${r.expect.padEnd(34)}→ ${r.got.padEnd(40)} ${rankTxt.padStart(5)}  ${r.signals}`,
    );
  }
  // punto de quiebre: primer caso NO superado dentro de la escalera
  const breakIdx = rows.findIndex((r) => !r.pass);
  if (breakIdx < 0) {
    console.log(`   → sin quiebre en ${rows.length} niveles.`);
  } else {
    const passed = breakIdx; // niveles superados antes de fallar
    console.log(
      `   → PUNTO DE QUIEBRE en «${rows[breakIdx].id}» (nivel ${breakIdx + 1}/${rows.length}); ` +
        `aguanta ${passed} nivel${passed === 1 ? "" : "es"}.`,
    );
  }
}

async function main(): Promise<void> {
  const client = new RemtkBenchClient();
  console.log("[stress] levantando remtk-prediction con juicio activo…");
  await client.start(BENCH_TOOLS, { qdrant: false });
  console.log(`[stress] listo, modo=${client.mode}`);

  const rows: Row[] = [];
  for (const c of CASES) {
    const r = await play(client, c);
    rows.push(r);
    process.stdout.write(`   · ${mark(r)} ${r.id} (${r.latencyMs}ms)\n`);
  }
  await client.close();

  const byAxis = new Map<string, Row[]>();
  for (const r of rows) (byAxis.get(r.axis) ?? byAxis.set(r.axis, []).get(r.axis)!).push(r);
  for (const group of byAxis.values()) renderAxis(group);

  const ok = rows.filter((r) => r.pass).length;
  console.log(`\n══ TOTAL: ${ok}/${rows.length} casos de esfuerzo superados.`);
}

main().catch((err) => {
  console.error("[stress] fallo fatal:", err);
  process.exit(1);
});
