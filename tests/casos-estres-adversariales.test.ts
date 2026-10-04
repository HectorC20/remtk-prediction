/**
 * Suite de casos de estrés límite y situaciones adversariales para el sistema de predicción:
 * 1. Ruido ortográfico extremo / Typos fonéticos / Jerga / Spanglish.
 * 2. Distracción adversarial con orden anidada entre comillas (evitar secuestro semántico).
 * 3. Doble negación afirmativa ("No es que no quiera X, hazlo").
 * 4. Condicional contrafáctico excluyente ("Si fuera A haría B, pero solo haz C").
 * 5. Polisemia / Falsos amigos contextuales ("carta", "banco", "hoja").
 * 6. Metáforas e intención implícita de negocio.
 * 7. Comentario conversacional casual sin intención de acción (Small-talk camuflado vs orden real).
 * 8. Rechazo explícito de una herramienta con selección de otra.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { VarianteCService } from "../src/predict/services/variante-c.service";
import { BENCH_TOOLS } from "../bench/catalog";

interface AdversarialTestCase {
  id: string;
  category: string;
  query: string;
  contextText?: string;
  expected: string[];
  forbidden?: string[];
  why: string;
}

const ADVERSARIAL_CASES: AdversarialTestCase[] = [
  {
    id: "typos-foneticos-extremos",
    category: "Ruido Ortográfico y Fonético",
    query: "nezesito konsultar el tipoo de kanbio del dolarr a zoles xfa",
    expected: ["get_exchange_rate"],
    why: "Múltiples faltas ortográficas fonéticas (k, z, doble letra, xfa); el bi-encoder debe capturar la semántica latente.",
  },
  {
    id: "spanglish-abreviaturas",
    category: "Mezcla de Idiomas / Spanglish",
    query: "plz search en internet info sobre la nueva version de react",
    expected: ["web_search"],
    why: "Mezcla inglés-español con abreviatura 'plz'; debe resolver a búsqueda web.",
  },
  {
    id: "distraccion-orden-anidada-comillas",
    category: "Distracción Adversarial / Cita",
    query: 'El cliente me dijo en su mensaje: "Por favor envíame un whatsapp urgente con la factura", pero tú solo agenda la reunión en el calendario para el viernes.',
    expected: ["create_calendar_event"],
    forbidden: ["send_whatsapp", "send_email"],
    why: "La orden real es agendar en calendario; el mensaje citado de WhatsApp debe ser ignorado como distractor.",
  },
  {
    id: "doble-negacion-afirmativa",
    category: "Doble Negación Pragmática",
    query: "No es que no quiera el reporte de ventas del mes, simplemente descárgalo de una vez.",
    expected: ["download_report"],
    why: "Doble negación ('No es que no quiera...') equivale a una orden afirmativa de descarga de reporte.",
  },
  {
    id: "condicional-contrafactico-descarte",
    category: "Razonamiento Contrafáctico",
    query: "Si tuviéramos problemas en producción te pediría ejecutar una consulta SQL, pero como todo va bien solo traduce este texto al inglés.",
    expected: ["translate_text"],
    forbidden: ["database_query"],
    why: "La condición de SQL es hipotética/contrafáctica no cumplida; la única acción real a ejecutar es la traducción.",
  },
  {
    id: "polisemia-carta-despacho",
    category: "Polisemia y Falsos Amigos",
    query: "Mándale la carta formal al proveedor por correo electrónico.",
    expected: ["send_email"],
    why: "'Carta' aquí significa misiva formal electrónica enviada por correo, no un menú de restaurante ni juego de azar.",
  },
  {
    id: "intencion-metaforica-implicita",
    category: "Metáfora / Intención Implícita",
    query: "El gerente quiere ver los números consolidados en su bandeja de entrada hoy mismo.",
    expected: ["send_email", "download_report"],
    why: "'Ver los números en su bandeja de entrada' implica preparar/descargar cifras y enviarlas por correo.",
  },
  {
    id: "rechazo-enfatico-con-alternativa",
    category: "Rechazo de Herramienta con Alternativa",
    query: "Ni se te ocurra tocar la base de datos de producción; mejor busca información en la web sobre el error 500.",
    expected: ["web_search"],
    forbidden: ["database_query"],
    why: "Prohibición explícita de database_query y orden directa de web_search.",
  },
  {
    id: "small-talk-clima-sin-orden",
    category: "Small-Talk Camuflado vs Consulta Real",
    query: "Vaya qué calor insoportable que hace hoy por aquí en la oficina, no se puede ni trabajar.",
    expected: [],
    why: "Es una queja/conversación casual sobre el ambiente físico, NO una solicitud de pronóstico meteorológico (get_weather). Debe abstenerse.",
  },
  {
    id: "consulta-clima-real-directa",
    category: "Consulta Meteorológica Real",
    query: "¿Qué clima y temperatura vamos a tener hoy en Arequipa?",
    expected: ["get_weather"],
    why: "Petición directa de información meteorológica.",
  },
];

test("Batería de Pruebas de Estrés Límite y Casos Adversariales", async () => {
  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  const service = new VarianteCService(engine);

  const scopeKey = "adversarial-stress-scope";
  console.log(`\nIndexando catálogo de ${BENCH_TOOLS.length} herramientas...`);
  await service.indexScope(scopeKey, BENCH_TOOLS);

  let passed = 0;

  for (const c of ADVERSARIAL_CASES) {
    const t0 = Date.now();
    const rank = await service.rank(scopeKey, c.query, c.contextText);
    const latency = Date.now() - t0;

    assert.ok(rank, `El ranking no debe ser undefined para ${c.id}`);
    const decision = service.arbitrate(rank);

    // Verificación de expectativas
    let isPass: boolean;
    if (c.expected.length === 0) {
      isPass = decision.tools.length === 0;
    } else {
      isPass = c.expected.some((e) => decision.tools.includes(e));
    }

    // Verificación de prohibidos
    if (c.forbidden && c.forbidden.length > 0) {
      const containsForbidden = c.forbidden.some((f) => decision.tools.includes(f));
      if (containsForbidden) isPass = false;
    }

    if (isPass) passed++;

    console.log(`\n─────────────────────────────────────────────────────────────`);
    console.log(`[${c.category}] ID: ${c.id}`);
    console.log(`Query:     "${c.query}"`);
    console.log(`Esperado:  [${c.expected.join(", ") || "∅ (Abstención)"}] ${c.forbidden ? `(Prohibido: [${c.forbidden.join(", ")}])` : ""}`);
    console.log(`Obtenido:  [${decision.tools.join(", ") || "∅"}] (Decisión: ${decision.decision}, Margen: ${rank.margin.toFixed(3)}, Régimen: ${rank.autonomous ? "AUTO" : "DEP"})`);
    console.log(`Latencia:  ${latency}ms | Resultado: ${isPass ? "✓ PASS" : "✗ FAIL"}`);
    console.log(`Por qué:   ${c.why}`);
  }

  console.log(`\n=============================================================`);
  console.log(`RESUMEN CASOS ADVERSARIALES Y ESTRÉS LÍMITE: ${passed}/${ADVERSARIAL_CASES.length} superados (${Math.round((passed / ADVERSARIAL_CASES.length) * 100)}%)`);
  console.log(`=============================================================\n`);
});
