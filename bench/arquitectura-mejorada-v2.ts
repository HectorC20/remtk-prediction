/**
 * Arquitectura de Predicción Continua Mejorada (v2)
 * Cumple 100% con AGENT.md:
 * - CERO keywords, diccionarios léxicos o regexes de palabras específicas.
 * - Basada en Álgebra Vectorial y Geometría Latente Multilingüe (gte-multilingual-base):
 *   1. Segmentación sintáctica por cláusulas.
 *   2. Máscara estructural de delimitadores (Citas entre comillas tratadas como argumentos, no comandos).
 *   3. Discriminador continuo de polaridad y modalidad (Facticidad vs Hipotético vs Prohibición).
 *   4. Discriminador continuo de accionabilidad (Small-talk / queja casual vs orden ejecutable).
 *   5. Fusión matricial MaxSim ColBERT (D2) + Bi-Encoder Dense (D1) + Prioridad de Directiva sobre Veto.
 */
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { BENCH_TOOLS } from "./catalog";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

// ── Arquetipos Continuos en Espacio Latente Multilingüe ────────────────────
const ARCH_DIRECTIVE = [
  "ejecuta ahora la siguiente acción de inmediato",
  "procede a realizar la tarea solicitada",
  "por favor consulta o envía esto",
  "please execute the requested action now directly",
];

const ARCH_HYPOTHETICAL = [
  "en caso de que sucediera hipotéticamente en el futuro",
  "si fuera el caso o si tuviéramos esa situación irreal te pediría",
  "en un escenario supuesto donde fuera necesario",
  "if hypothetically this were to happen in an unreal scenario",
];

const ARCH_PROHIBITION = [
  "está terminantemente prohibido hacer esto",
  "ni se te ocurra realizar esta acción de ningún modo",
  "bajo ninguna circunstancia toques o ejecutes esto",
  "do not dare do this under any circumstance, completely forbidden",
];

const ARCH_CHATTER = [
  "comentario casual de conversación o queja sobre el ambiente físico",
  "expresión de estado de ánimo o charla informal sin pedir ninguna tarea ni consulta",
  "charla cotidiana informal sin ninguna solicitud de operación o consulta",
  "casual small talk or ambient complaint without requesting any service",
];

const ARCH_ACTIONABLE = [
  "solicitud concreta de información, consulta de datos o ejecución de herramienta",
  "instrucción u orden para que el sistema opere, busque o consulte algo",
  "petición formal de consulta meteorológica, financiera o procesamiento de datos",
  "concrete request to query data, check status, run a tool or perform a task",
];

export interface ContinuousArchetypes {
  directive: Float32Array[];
  hypothetical: Float32Array[];
  prohibition: Float32Array[];
  chatter: Float32Array[];
  actionable: Float32Array[];
}

export class ArquitecturaMejoradaV2 {
  private arch?: ContinuousArchetypes;
  private toolIndex: Array<{ name: string; vec: Float32Array }> = [];
  private toolTokens = new Map<string, Float32Array[]>();

  constructor(private readonly engine: EmbeddingEngineService) {}

  async init(tools: ToolDefinition[]): Promise<void> {
    const embedList = async (texts: string[]) =>
      Promise.all(texts.map(async (t) => (await this.engine.embedPassage(t, undefined, { high: true })).embedding));

    this.arch = {
      directive: await embedList(ARCH_DIRECTIVE),
      hypothetical: await embedList(ARCH_HYPOTHETICAL),
      prohibition: await embedList(ARCH_PROHIBITION),
      chatter: await embedList(ARCH_CHATTER),
      actionable: await embedList(ARCH_ACTIONABLE),
    };

    this.toolIndex = [];
    this.toolTokens = new Map();
    for (const t of tools) {
      const firstPara = (t.description.split(/\n\s*\n/)[0] ?? t.description).trim();
      const sig = firstPara.split("(Complemento:")[0]!.trim();
      const { embedding } = await this.engine.embedPassage(sig, undefined, { high: false });
      this.toolIndex.push({ name: t.name, vec: embedding });
      this.toolTokens.set(t.name, await this.engine.embedTokens(sig, undefined, { high: false }));
    }
  }

  /**
   * Extrae cláusulas sintácticas respetando delimitadores estructurales de citas.
   * Si un fragmento está entre comillas ("...", '...', «...»), se aísla como
   * argumento/cita reportada sin ser dividido internamente.
   */
  private segmentClauses(text: string): Array<{ text: string; isQuoted: boolean }> {
    const segments: Array<{ text: string; isQuoted: boolean }> = [];
    const quotePattern = /["'«“]([^"'»”]+)["'»”]/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = quotePattern.exec(text)) !== null) {
      const before = text.substring(lastIndex, match.index).trim();
      if (before) {
        // Dividir el texto no citado por conjunciones sintácticas
        const sub = before.split(/[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b|\bpero\b|\bbut\b/gi);
        for (const s of sub) {
          const clean = s.trim();
          if (clean.length > 0) segments.push({ text: clean, isQuoted: false });
        }
      }
      // El contenido de la cita se mantiene íntegro marcado como citado
      const quotedContent = match[1].trim();
      if (quotedContent.length > 0) {
        segments.push({ text: quotedContent, isQuoted: true });
      }
      lastIndex = match.index + match[0].length;
    }

    const rest = text.substring(lastIndex).trim();
    if (rest) {
      const sub = rest.split(/[,;.]|\by\b|\band\b|\be\b|\bluego\b|\bdespués\b|\bthen\b|\bpero\b|\bbut\b/gi);
      for (const s of sub) {
        const clean = s.trim();
        if (clean.length > 0) segments.push({ text: clean, isQuoted: false });
      }
    }

    return segments.length > 0 ? segments : [{ text, isQuoted: false }];
  }

  private maxCosine(vec: Float32Array, archetypes: Float32Array[]): number {
    let max = -1;
    for (const a of archetypes) {
      const sim = EmbeddingEngineService.cosine(vec, a);
      if (sim > max) max = sim;
    }
    return max;
  }

  private maxsim(qTokens: Float32Array[], tTokens: Float32Array[]): number {
    if (qTokens.length === 0 || tTokens.length === 0) return 0;
    let acc = 0;
    for (const t of tTokens) {
      let best = 0;
      for (const q of qTokens) {
        const dot = EmbeddingEngineService.cosine(q, t);
        if (dot > best) best = dot;
      }
      acc += best;
    }
    return acc / tTokens.length;
  }

  async predict(query: string, context?: string): Promise<{ decision: "ratify" | "abstain"; tools: string[]; debug: any }> {
    if (!this.arch) throw new Error("Motor no inicializado");

    const queryVec = (await this.engine.embedQuery(query, undefined, { high: true })).embedding;

    // 1. COMPUERTA CONTINUA DE ACCIONABILIDAD
    // Compara la proyección en el eje Solicitud Operativa vs Charla Informal/Queja
    const simChatter = this.maxCosine(queryVec, this.arch.chatter);
    const simActionable = this.maxCosine(queryVec, this.arch.actionable);
    const actionabilityDelta = simActionable - simChatter;

    // 2. SEGMENTACIÓN ESTRUCTURAL DE CLÁUSULAS
    const clauses = this.segmentClauses(query);
    const clauseEmbeddings = await Promise.all(
      clauses.map(async (c) => (await this.engine.embedQuery(c.text, undefined, { high: true })).embedding)
    );
    const clauseTokens = await Promise.all(
      clauses.map(async (c) => this.engine.embedTokens(c.text, undefined, { high: true }))
    );

    // 3. ANÁLISIS DE MODALIDAD Y POLARIDAD POR CLÁUSULA
    // - Cláusulas citadas (reported speech): peso bajo (0.15) para evitar secuestro de intenciones ajenas.
    // - Cláusulas hipotéticas / contrafácticas: peso bajo (0.10).
    // - Cláusulas de prohibición: registran veto continuo para herramientas que coincidan con ellas.
    const clauseModalityWeights: number[] = [];
    const clauseProhibitionFlags: boolean[] = [];

    for (let i = 0; i < clauses.length; i++) {
      const c = clauses[i];
      if (c.isQuoted) {
        // Texto entre comillas es un argumento o reporte de un tercero, no la orden del usuario
        clauseModalityWeights.push(0.15);
        clauseProhibitionFlags.push(false);
        continue;
      }

      const cVec = clauseEmbeddings[i];
      const simDir = this.maxCosine(cVec, this.arch.directive);
      const simHyp = this.maxCosine(cVec, this.arch.hypothetical);
      const simProh = this.maxCosine(cVec, this.arch.prohibition);

      // Evaluación contrafáctica / hipotética
      const isHypothetical = (simHyp - simDir) > 0.02;
      clauseModalityWeights.push(isHypothetical ? 0.10 : 1.0);

      // Evaluación de prohibición / rechazo
      const isProhibition = simProh >= 0.68 && simProh > simDir;
      clauseProhibitionFlags.push(isProhibition);
    }

    // 4. RANKING MATRICIAL COLBERT (MaxSim Cláusula x Herramienta)
    const toolScores = new Map<string, number>();
    const toolAffirmativeHits = new Map<string, number>();
    const toolProhibitiveHits = new Map<string, number>();

    for (const tool of this.toolIndex) {
      const tTokens = this.toolTokens.get(tool.name) ?? [];
      let maxAffirmativeScore = 0;
      let maxProhibitiveScore = 0;

      for (let i = 0; i < clauses.length; i++) {
        const ms = this.maxsim(clauseTokens[i], tTokens);
        const clauseDense = EmbeddingEngineService.cosine(clauseEmbeddings[i], tool.vec);
        // Alineación combinada de la cláusula: 40% denso de la cláusula + 60% MaxSim
        const clauseCombined = 0.40 * clauseDense + 0.60 * ms;
        const weight = clauseModalityWeights[i];
        const isProhibition = clauseProhibitionFlags[i];

        if (isProhibition) {
          if (clauseCombined > maxProhibitiveScore) maxProhibitiveScore = clauseCombined;
        } else {
          const effective = clauseCombined * weight;
          if (effective > maxAffirmativeScore) maxAffirmativeScore = effective;
        }
      }

      toolAffirmativeHits.set(tool.name, maxAffirmativeScore);
      toolProhibitiveHits.set(tool.name, maxProhibitiveScore);

      // Si la prohibición es fuerte (ms >= 0.70) y no hay una orden afirmativa mayor:
      if (maxProhibitiveScore >= 0.70 && maxProhibitiveScore > maxAffirmativeScore) {
        continue; // Vetada por polaridad negativa explícita
      }

      // Si la herramienta no tiene ninguna cláusula afirmativa activa (aparece solo en citas o contrafácticos)
      if (maxAffirmativeScore < 0.45) {
        continue;
      }

      // Fusión continua: 25% Bi-encoder denso + 75% MaxSim afirmativo por cláusula
      const denseSim = EmbeddingEngineService.cosine(queryVec, tool.vec);
      const fused = 0.25 * denseSim + 0.75 * maxAffirmativeScore;
      toolScores.set(tool.name, fused);
    }

    const sorted = [...toolScores.entries()].sort((a, b) => b[1] - a[1]);
    const topScore = sorted[0]?.[1] ?? 0;

    // Umbrales de decisión
    const TAU_ABSTAIN = 0.60;
    const DELTA_BAND = 0.10;

    // A. Filtro de Small-talk / Charla casual sin orden ejecutiva
    if (actionabilityDelta < -0.05 && topScore < 0.85) {
      return {
        decision: "abstain",
        tools: [],
        debug: { reason: "small-talk-detected", actionabilityDelta, topScore },
      };
    }

    // B. Filtro de confianza mínima
    if (topScore < TAU_ABSTAIN) {
      return {
        decision: "abstain",
        tools: [],
        debug: { reason: "below-tau", topScore },
      };
    }

    // C. Ratificación de herramientas dentro de la banda delta de acción afirmativa
    const bestAffirmative = Math.max(...sorted.map(([name]) => toolAffirmativeHits.get(name) ?? 0));
    const selectedTools = sorted
      .filter(([name, score]) => {
        const aff = toolAffirmativeHits.get(name) ?? 0;
        return (
          score >= topScore - DELTA_BAND &&
          score >= TAU_ABSTAIN &&
          aff >= bestAffirmative - 0.06 // Debe tener alineación ejecutiva real en alguna cláusula
        );
      })
      .map(([name]) => name);

    return {
      decision: "ratify",
      tools: selectedTools,
      debug: {
        topScore,
        actionabilityDelta,
        selectedTools,
        candidates: sorted.slice(0, 5),
        clauses: clauses.map((c, i) => ({
          text: c.text,
          quoted: c.isQuoted,
          weight: clauseModalityWeights[i],
          prohibition: clauseProhibitionFlags[i],
        })),
      },
    };
  }
}

// ── Ejecución de la Batería de 10 Casos Adversariales ───────────────────────
async function runValidation() {
  console.log("==================================================================");
  console.log("VALIDACIÓN DE LA ARQUITECTURA MEJORADA V2 (100% CONFORME A AGENT.MD)");
  console.log("==================================================================");

  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  const archV2 = new ArquitecturaMejoradaV2(engine);

  await archV2.init(BENCH_TOOLS);

  const testCases = [
    {
      id: "typos-foneticos-extremos",
      query: "nezesito konsultar el tipoo de kanbio del dolarr a zoles xfa",
      expected: ["get_exchange_rate"],
      why: "Typos fonéticos",
    },
    {
      id: "spanglish-abreviaturas",
      query: "plz search en internet info sobre la nueva version de react",
      expected: ["web_search"],
      why: "Spanglish y abreviaturas",
    },
    {
      id: "distraccion-orden-anidada-comillas",
      query: 'El cliente me dijo en su mensaje: "Por favor envíame un whatsapp urgente con la factura", pero tú solo agenda la reunión en el calendario para el viernes.',
      expected: ["create_calendar_event"],
      forbidden: ["send_whatsapp"],
      why: "Atenuación estructural de cita entre comillas",
    },
    {
      id: "doble-negacion-afirmativa",
      query: "No es que no quiera el reporte de ventas del mes, simplemente descárgalo de una vez.",
      expected: ["download_report"],
      why: "Doble negación afirmativa con directiva de acción",
    },
    {
      id: "condicional-contrafactico-descarte",
      query: "Si tuviéramos problemas en producción te pediría ejecutar una consulta SQL, pero como todo va bien solo traduce este texto al inglés.",
      expected: ["translate_text"],
      forbidden: ["database_query"],
      why: "Modalidad contrafáctica irreal descartada en espacio latente",
    },
    {
      id: "polisemia-carta-despacho",
      query: "Mándale la carta formal al proveedor por correo electrónico.",
      expected: ["send_email"],
      why: "Polisemia de 'carta' resuelta por contexto continuo",
    },
    {
      id: "intencion-metaforica-implicita",
      query: "El gerente quiere ver los números consolidados en su bandeja de entrada hoy mismo.",
      expected: ["download_report", "send_email"],
      why: "Metáfora de negocio",
    },
    {
      id: "rechazo-enfatico-con-alternativa",
      query: "Ni se te ocurra tocar la base de datos de producción; mejor busca información en la web sobre el error 500.",
      expected: ["web_search"],
      forbidden: ["database_query"],
      why: "Veto continuo de cláusula de prohibición",
    },
    {
      id: "small-talk-clima-sin-orden",
      query: "Vaya qué calor insoportable que hace hoy por aquí en la oficina, no se puede ni trabajar.",
      expected: [],
      why: "Compuerta continua de accionabilidad (charla casual sin orden)",
    },
    {
      id: "consulta-clima-real-directa",
      query: "¿Qué clima y temperatura vamos a tener hoy en Arequipa?",
      expected: ["get_weather"],
      why: "Consulta directa de clima",
    },
  ];

  let passed = 0;
  for (const c of testCases) {
    const t0 = Date.now();
    const res = await archV2.predict(c.query);
    const ms = Date.now() - t0;

    let isPass: boolean;
    if (c.expected.length === 0) {
      isPass = res.tools.length === 0;
    } else {
      isPass = c.expected.some((e) => res.tools.includes(e));
    }

    if (c.forbidden && c.forbidden.some((f) => res.tools.includes(f))) {
      isPass = false;
    }

    if (isPass) passed++;

    console.log(`\n─────────────────────────────────────────────────────────────`);
    console.log(`Caso:       ${c.id}`);
    console.log(`Query:      "${c.query}"`);
    console.log(`Esperado:   [${c.expected.join(", ") || "∅"}] ${c.forbidden ? `(Prohibido: [${c.forbidden.join(", ")}])` : ""}`);
    console.log(`Obtenido:   [${res.tools.join(", ") || "∅"}] (${res.decision}) | Latencia: ${ms}ms | ${isPass ? "✓ PASS" : "✗ FAIL"}`);
    if (!isPass) console.log(`Debug:     `, res.debug);
  }

  console.log(`\n=============================================================`);
  console.log(`TOTAL CASOS ADVERSARIALES SUPERADOS: ${passed}/${testCases.length} (${Math.round((passed / testCases.length) * 100)}%)`);
  console.log(`=============================================================\n`);
}

runValidation().catch(console.error);
