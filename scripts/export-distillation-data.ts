import { writeFileSync, readFileSync, existsSync } from "node:fs";
import * as path from "node:path";

interface RawTool {
  name: string;
  description: string;
}

interface DistilledSample {
  sample_id: string;
  query_text: string;
  keywords: string[];
  intent_context: string;
  history_text: string;
  candidate_tools: string[]; // tool names
  teacher_targets: {
    estado_probs: number[]; // [TRANSICION, CONTINUACION, CORRECCION, ABANDONO]
    estado_label: number;
    topic_shift: number;
    positive_tool_indices: number[]; // indices in candidate_tools
    abstencion: number; // 0.0 or 1.0
    expert_hint: number; // 0..5 (Expert 1..6)
  };
}

const catalogPath = path.resolve("catalog-232.json");
const catalog: RawTool[] = JSON.parse(readFileSync(catalogPath, "utf-8"));
const toolMap = new Map<string, RawTool>(catalog.map((t) => [t.name, t]));
const allToolNames = catalog.map((t) => t.name);

const samples: DistilledSample[] = [];
let sampleCounter = 0;

function addSample(
  query: string,
  posTools: string[],
  options: {
    history?: string;
    intentContext?: string;
    keywords?: string[];
    estado?: "TRANSICION" | "CONTINUACION" | "CORRECCION" | "ABANDONO";
    abstencion?: number;
    expertHint?: number;
    numNegatives?: number;
  } = {}
) {
  sampleCounter++;
  const id = `sample_${String(sampleCounter).padStart(5, "0")}`;
  const history = options.history ?? "";
  const intentCtx = options.intentContext ?? "";
  const keywords = options.keywords ?? [];
  const abstencion = options.abstencion ?? (posTools.length === 0 ? 1.0 : 0.0);
  const estado = options.estado ?? (history ? "CONTINUACION" : "TRANSICION");

  // Estado probs
  let estado_probs = [0.8, 0.1, 0.05, 0.05];
  let estado_label = 0;
  if (estado === "TRANSICION") {
    estado_probs = [0.85, 0.05, 0.05, 0.05];
    estado_label = 0;
  } else if (estado === "CONTINUACION") {
    estado_probs = [0.05, 0.85, 0.05, 0.05];
    estado_label = 1;
  } else if (estado === "CORRECCION") {
    estado_probs = [0.05, 0.1, 0.8, 0.05];
    estado_label = 2;
  } else if (estado === "ABANDONO") {
    estado_probs = [0.05, 0.05, 0.05, 0.85];
    estado_label = 3;
  }

  const topic_shift = (estado_label === 0 || estado_label === 3) ? 1.0 : 0.0;
  const expert_hint = options.expertHint ?? 0;

  // Filter valid positive tools from catalog
  const validPos = posTools.filter((t) => toolMap.has(t));
  
  // Pick negative distractors from catalog
  const numNeg = options.numNegatives ?? Math.min(25, allToolNames.length - validPos.length);
  const posSet = new Set(validPos);
  const negPool = allToolNames.filter((t) => !posSet.has(t));
  // Shuffle negatives pseudo-randomly
  const chosenNeg: string[] = [];
  for (let i = 0; i < numNeg; i++) {
    const idx = (sampleCounter * 17 + i * 31) % negPool.length;
    chosenNeg.push(negPool[idx]);
  }

  // Combine and shuffle candidates
  const candidates = [...validPos, ...chosenNeg];
  // Deterministic shuffle
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = (sampleCounter * 7 + i * 13) % (i + 1);
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }

  const posIndices: number[] = [];
  for (let i = 0; i < candidates.length; i++) {
    if (posSet.has(candidates[i])) posIndices.push(i);
  }

  samples.push({
    sample_id: id,
    query_text: query,
    keywords,
    intent_context: intentCtx,
    history_text: history,
    candidate_tools: candidates,
    teacher_targets: {
      estado_probs,
      estado_label,
      topic_shift,
      positive_tool_indices: posIndices,
      abstencion,
      expert_hint,
    },
  });
}

// 1. Casos de Catálogo MiTumbes y resto del catálogo
for (const tool of catalog) {
  const cleanName = tool.name.replace(/_/g, " ");
  // Formatos comunes de prompt
  addSample(`quiero usar ${cleanName}`, [tool.name], { expertHint: 1 });
  addSample(`ejecuta la acción de ${cleanName}`, [tool.name], { expertHint: 0 });
  addSample(`consultar ${cleanName}`, [tool.name], { expertHint: 1 });
  addSample(`ayúdame con ${cleanName} por favor`, [tool.name], { expertHint: 2 });
  addSample(`necesito ${cleanName}`, [tool.name], { expertHint: 0 });
  addSample(`por favor realiza ${cleanName}`, [tool.name], { expertHint: 0 });

  if (tool.description) {
    const snippet = tool.description.slice(0, 80).replace(/[.,;:()]/g, "");
    addSample(`acción para ${snippet}`, [tool.name], { expertHint: 1 });
  }
}

// 2. Casos de Programación y Workflows
const devCases: [string, string[], number][] = [
  ["crea un nuevo archivo index.ts con express", ["workspace_write_file"], 0],
  ["guarda el archivo con la configuración del servidor", ["workspace_write_file"], 0],
  ["escribe el código del componente Button.tsx", ["workspace_write_file"], 0],
  ["lee el archivo package.json para ver las dependencias", ["workspace_read_file"], 1],
  ["inspecciona el contenido de tsconfig.json", ["workspace_read_file"], 1],
  ["revisa qué hay dentro de src/app.module.ts", ["workspace_read_file"], 1],
  ["lista los archivos del directorio src", ["workspace_list"], 1],
  ["muéstrame qué carpetas y archivos hay en el proyecto", ["workspace_list"], 1],
  ["busca en internet la documentación de nestjs 12", ["web_search"], 1],
  ["busca en la web las últimas canciones de joji", ["web_search"], 1],
  ["busca información en google sobre el clima en tumbes", ["web_search"], 1],
];
for (const [q, tools, exp] of devCases) {
  addSample(q, tools, { expertHint: exp });
  addSample(`por favor ${q}`, tools, { expertHint: exp });
  addSample(`necesito que ${q}`, tools, { expertHint: exp });
}

// 3. Casos Multi-Turno y Referenciales (CONTINUACION / CORRECCION)
const multiTurnCases = [
  {
    q: "léeme el archivo que creaste",
    h: "user: crea un script index.js con suma | assistant: Archivo index.js creado exitosamente",
    tools: ["workspace_read_file"],
    estado: "CONTINUACION" as const,
    exp: 2,
  },
  {
    q: "guárdalo en un archivo",
    h: "user: busca canciones de joji | assistant: Aquí tienes el tracklist de Piss In The Wind",
    tools: ["workspace_write_file"],
    estado: "CONTINUACION" as const,
    exp: 2,
  },
  {
    q: "el primero",
    h: "assistant: Encontré 3 opciones: 1. Hotel Punta Sal 2. Hotel Zorritos | user: el primero",
    tools: ["mitumbes_lugar_obtener", "mitumbes_item_obtener"],
    estado: "CONTINUACION" as const,
    exp: 2,
  },
  {
    q: "no hagas eso, mejor busca restaurantes en la plaza",
    h: "assistant: ¿Deseas reservar un hotel en Tumbes?",
    tools: ["mitumbes_buscar", "mitumbes_lugar_listar"],
    estado: "CORRECCION" as const,
    exp: 0,
  },
  {
    q: "cancela lo anterior y búscame las actividades turísticas",
    h: "user: muéstrame pasajes en bus",
    tools: ["mitumbes_lugar_listar", "mitumbes_buscar"],
    estado: "CORRECCION" as const,
    exp: 0,
  },
];
for (const c of multiTurnCases) {
  addSample(c.q, c.tools, { history: c.h, estado: c.estado, expertHint: c.exp });
}

// 4. Casos Adversariales y Párrafos Largos
const advCases = [
  {
    q: 'El cliente me dijo en su mensaje: "Por favor envíame un correo con la factura", pero tú solo agenda la reunión en el calendario.',
    tools: ["calendar_scheduler_op_1", "calendar_scheduler_op_2"],
    exp: 5,
  },
  {
    q: "No es que no quiera el reporte, simplemente descárgalo de una vez.",
    tools: ["reporting_analytics_op_1"],
    exp: 0,
  },
  {
    q: "nezesito konsultar el tipoo de kanbio del dolarr a zoles xfa",
    tools: ["web_search", "finance_accounting_op_1"],
    exp: 2,
  },
  {
    q: "plz search en internet info sobre la nueva version de react",
    tools: ["web_search"],
    exp: 1,
  },
];
for (const a of advCases) {
  addSample(a.q, a.tools, { expertHint: a.exp });
}

// 5. Casos de Abstención y Retorno Nulo (ABANDONO / NOOP)
const nullCases = [
  "hola buenos días cómo estás",
  "muchas gracias por tu ayuda, eso era todo",
  "adiós, nos vemos mañana",
  "qué opinas sobre la filosofía existencialista de Sartre?",
  "El Perú es un país situado en la parte occidental de América del Sur. Limita con Ecuador y Colombia al norte.",
  "La fotosíntesis es el proceso bioquímico mediante el cual las plantas convierten la energía solar en glucosa.",
  "qué herramientas propias se tiene ?",
  "no sé qué hacer hoy día jaja",
  "test de prueba de conexión sin ninguna acción operativa",
  "ok entendido",
  "perfecto, muchas gracias",
  "excelente servicio",
  "cuál es la capital de Francia?",
  "el agua hierve a 100 grados centígrados al nivel del mar",
  "podrías explicarme la teoría de la relatividad general?",
  "buen día asistente, qué puedes hacer?",
  "no gracias, hasta luego",
  "saludos cordiales a todo el equipo",
  "buenas tardes",
  "hola que tal",
  "la física cuántica describe la naturaleza a escalas atómicas y subatómicas",
  "el teorema de pitágoras establece que a^2 + b^2 = c^2",
  "me gusta mucho programar en typescript y python",
  "hoy es un día soleado en lima",
  "me parece muy interesante lo que dices"
];
for (const n of nullCases) {
  addSample(n, [], { abstencion: 1.0, estado: "ABANDONO", expertHint: 5 });
  addSample(`usuario dice: ${n}`, [], { abstencion: 1.0, estado: "ABANDONO", expertHint: 5 });
  addSample(`solo quería comentar que ${n}`, [], { abstencion: 1.0, estado: "ABANDONO", expertHint: 5 });
}

const outPath = path.resolve("training/distilled_samples.json");
writeFileSync(outPath, JSON.stringify(samples, null, 2), "utf-8");
console.log(`Generados ${samples.length} ejemplos destilados en ${outPath}`);
