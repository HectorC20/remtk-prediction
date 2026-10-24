/**
 * Benchmarks Avanzados de Intención (Intent Benchmarks Suite):
 * Prueba la capacidad del sistema para deducir intención en escenarios complejos:
 * 
 * 1. Intención Implícita Compleja (sin verbos ni nombres directos del catálogo)
 * 2. Autocorrección y Cambios de Opinión dentro del mismo turno
 * 3. Expresiones Coloquiales y Variaciones Dialectales (Perú, Argentina, Colombia, España, México)
 * 4. Resistencia a Ruido Tipográfico / Typos (OCR / Escritura rápida)
 * 5. Peticiones Hipotéticas vs Ejecutables
 */
import { BENCH_TOOLS } from "./catalog";
import type { BenchTurn } from "./cases";
import { RemtkBenchClient } from "./remtk.client";

interface ComplexIntentCase {
  category: string;
  id: string;
  description: string;
  turns: BenchTurn[];
  expected: string[];
  expectEmpty?: boolean;
}

const U = (content: string): BenchTurn => ({ role: "user", content });
const A = (content: string): BenchTurn => ({ role: "assistant", content });

const COMPLEX_CASES: ComplexIntentCase[] = [
  // ── 1. INTENCIÓN IMPLÍCITA COMPLEJA ──────────────────────────────────────
  {
    category: "1·implícita-compleja",
    id: "imp-1-clima",
    description: "Expresa necesidad sin mencionar clima/pronóstico/tiempo",
    turns: [U("Voy a salir a almorzar en un rato y no sé si debo llevar abrigo o sombrilla")],
    expected: ["get_weather"],
  },
  {
    category: "1·implícita-compleja",
    id: "imp-2-tipo-cambio",
    description: "Expresa necesidad financiera sin mencionar dólares/moneda/tipo de cambio",
    turns: [U("Tengo una cotización en moneda extranjera y necesito saber cuántos soles peruanos representa hoy")],
    expected: ["get_exchange_rate"],
  },
  {
    category: "1·implícita-compleja",
    id: "imp-3-resumen",
    description: "Expresa sobrecarga de lectura sin mencionar resumen/sintetizar",
    turns: [U("Este informe tiene más de 50 páginas y mi reunión empieza en 5 minutos, dame solo los puntos clave")],
    expected: ["summarize_document"],
  },
  {
    category: "1·implícita-compleja",
    id: "imp-4-facturas",
    description: "Expresa control contable sin mencionar listar/billing",
    turns: [U("Necesito revisar cuáles son los cobros que le hemos emitido a los clientes este mes")],
    expected: ["list_invoices"],
  },

  // ── 2. AUTOCORRECCIÓN Y CAMBIO DE OPINIÓN EN EL MISMO TURNO ──────────────
  {
    category: "2·autocorrección-intraturno",
    id: "auto-1-cancelar-accion",
    description: "El usuario inicia pidiendo enviar correo pero rectifica a borrador a mitad de frase",
    turns: [U("Quiero enviar el reporte por correo a la directiva... bueno no, espérate, mejor déjalo guardado como borrador")],
    expected: ["save_draft"],
  },
  {
    category: "2·autocorrección-intraturno",
    id: "auto-2-cambio-herramienta",
    description: "Pide buscar en la web pero cambia a consultar tipo de cambio oficial",
    turns: [U("Busca en google a cuánto cotiza el dólar... no, mejor usa la consulta oficial de tipo de cambio")],
    expected: ["get_exchange_rate"],
  },
  {
    category: "2·autocorrección-intraturno",
    id: "auto-3-descarte-total",
    description: "Pide agendar una reunión pero al final decide no hacer nada",
    turns: [U("Agenda una reunión con el equipo para mañana a las 3pm... ah no olvígalo, ya no será necesaria")],
    expected: [],
    expectEmpty: true,
  },

  // ── 3. COLOQUIALISMOS Y VARIACIONES REGIONALES / DIALECTALES ──────────────
  {
    category: "3·coloquial-regional",
    id: "dial-1-rioplatense",
    description: "Rioplatense (Argentina/Uruguay): vos/che/pásame",
    turns: [U("Che vos, pásame este texto al inglés que no entiendo un pito")],
    expected: ["translate_text"],
  },
  {
    category: "3·coloquial-regional",
    id: "dial-2-mexicano",
    description: "Mexicano: checa/porfa/paro",
    turns: [U("Hazme el paro y chécame cómo va a estar el tiempo mañana por aca")],
    expected: ["get_weather"],
  },
  {
    category: "3·coloquial-regional",
    id: "dial-3-peruano",
    description: "Peruano: mano/sácame/al toque",
    turns: [U("Mano, sácame al toque el balance de lo que hemos vendido ayer")],
    expected: ["download_report"],
  },
  {
    category: "3·coloquial-regional",
    id: "dial-4-colombiano",
    description: "Colombiano: parcero/regálame",
    turns: [U("Parcero, regálame una búsqueda en internet sobre las noticias de hoy")],
    expected: ["web_search"],
  },
  {
    category: "3·coloquial-regional",
    id: "dial-5-espanol",
    description: "Español peninsular: tío/déjame/curro",
    turns: [U("Tío, déjame este correo guardado en borradores que ahora tengo mazo de curro")],
    expected: ["save_draft"],
  },

  // ── 4. RESISTENCIA A RUIDO TIPOGRÁFICO Y TYPOS ─────────────────────────────
  {
    category: "4·ruido-tipografico",
    id: "typo-1-reporte",
    description: "Errores ortográficos severos en descarga de reporte",
    turns: [U("descarra el repolte de bentas de ayer porfa")],
    expected: ["download_report"],
  },
  {
    category: "4·ruido-tipografico",
    id: "typo-2-correo",
    description: "Typos en envío de email",
    turns: [U("emvia er meil kon el rezumen al kliemte")],
    expected: ["send_email"],
  },
  {
    category: "4·ruido-tipografico",
    id: "typo-3-traduccion",
    description: "Typos en traducción",
    turns: [U("traduse este tekstoo al yngles")],
    expected: ["translate_text"],
  },

  // ── 5. PREGUNTAS CONDICIONALES E HIPOTÉTICAS ─────────────────────────────
  {
    category: "5·hipotéticas-condicionales",
    id: "hip-1-exploratoria",
    description: "Pregunta si es posible ejecutar algo sin pedir ejecutarlo",
    turns: [U("¿Sería posible en algún momento consultar el pronóstico del tiempo para Madrid?")],
    expected: ["get_weather"],
  },
  {
    category: "5·hipotéticas-condicionales",
    id: "hip-2-duda-capacidad",
    description: "Pregunta sobre capacidad de búsqueda",
    turns: [U("¿Tienes forma de buscar información en la web sobre precios de tecnología?")],
    expected: ["web_search"],
  },
];

async function main(): Promise<void> {
  const client = new RemtkBenchClient();
  console.log("=========================================================");
  console.log("   REMTK-PREDICTION: SUITE AVANZADA DE INTENCIÓN (TESTS) ");
  console.log("=========================================================\n");
  console.log("[intent-suite] Iniciando servicio con juicio activo y ONNX...");
  await client.start(BENCH_TOOLS, { qdrant: false });

  let totalPassed = 0;
  const categories = new Map<string, { total: number; passed: number }>();

  for (const testCase of COMPLEX_CASES) {
    const history: BenchTurn[] = [];
    let lastResult: Awaited<ReturnType<RemtkBenchClient["predict"]>> | undefined;

    for (const turn of testCase.turns) {
      if (turn.role === "user") {
        lastResult = await client.predict(`intent-${testCase.id}`, turn.content, [...history]);
      }
      history.push(turn);
    }

    const gotTools = lastResult?.tools ?? [];
    const expectEmpty = testCase.expectEmpty === true;

    let pass = false;
    if (expectEmpty) {
      pass = gotTools.length === 0;
    } else {
      pass = testCase.expected.every((exp) => gotTools.includes(exp));
    }

    if (pass) totalPassed++;

    const catStats = categories.get(testCase.category) ?? { total: 0, passed: 0 };
    catStats.total++;
    if (pass) catStats.passed++;
    categories.set(testCase.category, catStats);

    const mark = pass ? "✓ [PASS]" : "✗ [FAIL]";
    const expTxt = expectEmpty ? "∅ (Vacío)" : testCase.expected.join(", ");
    const gotTxt = gotTools.slice(0, 5).join(", ") || "∅ (Vacío)";

    console.log(`${mark} ${testCase.id.padEnd(26)} | Esperado: ${expTxt.padEnd(22)} | Obtenido: ${gotTxt}`);
    console.log(`   └─ Nota: ${testCase.description}`);
  }

  await client.close();

  console.log("\n=========================================================");
  console.log("   RESUMEN POR CATEGORÍA DE INTENCIÓN");
  console.log("=========================================================");
  for (const [cat, stats] of categories.entries()) {
    const pct = ((stats.passed / stats.total) * 100).toFixed(1);
    console.log(` • ${cat.padEnd(30)}: ${stats.passed}/${stats.total} (${pct}%)`);
  }

  console.log("\n=========================================================");
  console.log(` RESULTADO FINAL DE INTENCIÓN: ${totalPassed}/${COMPLEX_CASES.length} (${((totalPassed / COMPLEX_CASES.length) * 100).toFixed(1)}%)`);
  console.log("=========================================================\n");

  if (totalPassed < COMPLEX_CASES.length) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fallo fatal en la suite de intención:", err);
  process.exit(1);
});
