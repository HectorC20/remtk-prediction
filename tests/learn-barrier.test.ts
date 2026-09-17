/**
 * F5 — Validación de la capa de aprendizaje léxico por canal
 * (docs/entrenamiento-prediccion.md §12).
 *
 * Caso real del log `key-remtk/tests/docs/respuesta2.md`: ante el prompt
 * "podrás crear una tarea de cada 5 minutos buscando noticias en tumbes" el
 * pipeline acabó con 0 herramientas ("Predicción por tarea no disponible",
 * "ejecuta en modo degradado") y el modelo respondió que no tenía
 * `schedule_task`, aun estando registrada. La causa medida es el PISO DE
 * RELEVANCIA (§6.6): sobre coseno puro, la mejor herramienta se queda por
 * debajo de `ONNX_ADAPTIVE_MIN_SCORE` y la respuesta se vacía.
 *
 * Este script responde a una sola pregunta: ¿la cuarta señal (perfil léxico
 * aprendido del canal) rompe esa barrera?
 *
 *   §1 CORE — determinista, sin ONNX ni Qdrant: se fijan `z_t` y el coseno de
 *      cada nodo a mano y se ejercen `RerankService` + `LexicalProfileService`.
 *      A (baseline) = 0 tools · B (tras N eventos) = `schedule_task` en el top.
 *   §2 E2E — HTTP 6776: contrato completo `/tools` → `/predict` →
 *      `/predict/feedback` → `/predict`, más el aislamiento entre canales.
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { startPredictServer } from "../src/app.module";
import { LexicalProfileService } from "../src/predict/services/lexical-profile.service";
import { RerankService } from "../src/predict/services/rerank.service";
import type { ScoredTool, ToolDefinition } from "../src/shared/interfaces/domain.interface";
import type { TenantToolGraph, ToolGraphNode } from "../src/shared/interfaces/graph.interface";

// ───────────────────────────────────────────────────────────────────────────
// Escenario: el caso del log
// ───────────────────────────────────────────────────────────────────────────

/** Petición real del usuario (log L23). */
const LOG_PROMPT = "podrás crear una tarea de cada 5 minutos buscando noticias en tumbes";

/**
 * Texto del evento de refuerzo: el ejecutor envía la subtarea serializada, no
 * el prompt literal. Comparte vocabulario con él pero no es idéntico, así que
 * §1-B cubre también la generalización por términos (§6.2).
 */
const LOG_OBSERVED =
  "usar schedule_task para crear una tarea recurrente cada 5 minutos buscando noticias en tumbes";

/** Piso de relevancia de producción (`minScoreDefault`), el que vació el log. */
const MIN_SCORE = 0.8;
/** Peso del término aprendido (`learnWeightDefault`). */
const LEARN_WEIGHT = 0.25;
/** Señales mínimas del canal antes de puntuar (`learnMinEventsDefault`). */
const LEARN_MIN_EVENTS = 3;

/** Canal entrenado y canal ajeno (comparación de aislamiento, §9.4). */
const TRAINED_SCOPE = "4f4e8014-0000-0000-0000-000000000000";
const OTHER_SCOPE = "otro-usuario-0000-0000-0000-000000000000";

/**
 * Cosenos fijados a mano contra `z_t`. `schedule_task` queda en 0.70, POR
 * DEBAJO del piso de 0.80: es exactamente la situación del log (la herramienta
 * correcta está en el catálogo pero el coseno no alcanza el mínimo).
 */
const COSINES: Record<string, number> = {
  schedule_task: 0.7,
  web_search: 0.62,
  read_news_feed: 0.55,
  send_whatsapp: 0.5,
  summarize_document: 0.45,
  create_calendar_event: 0.4,
};

const ZT = new Float32Array([1, 0]);

/** Vector unitario en 2D cuyo coseno respecto de `ZT` es exactamente `cos`. */
function unit(cos: number): Float32Array {
  return new Float32Array([cos, Math.sqrt(Math.max(0, 1 - cos * cos))]);
}

function def(name: string, group: string, tags: string[]): ToolDefinition {
  return {
    id: name,
    name,
    group,
    category: "runtime",
    description: "",
    tags,
    intentSummary: "",
    inputSchema: {},
  };
}

/** Réplica fiel de `key-remtk/src/shared/constants/mcp/scheduler.mcp.ts`. */
const SCHEDULE_TASK: ToolDefinition = {
  id: "schedule_task",
  name: "schedule_task",
  group: "scheduler",
  category: "runtime",
  description:
    "Programa una tarea del agente para el futuro: única (executeAt) o periódica (cron).",
  tags: [
    "scheduler",
    "cron",
    "recordatorio",
    "reminder",
    "agendar",
    "programar",
    "tarea",
    "evento",
    "delegar",
    "autonomo",
    "tools",
    "canales",
    "agente",
  ],
  intentSummary:
    "Programa una tarea del agente (única o recurrente con cron): recordatorios, seguimientos, reportes y delegaciones autónomas",
  inputSchema: { type: "object", properties: { cron: {}, executeAt: {} } },
};

/** Catálogo reducido con `schedule_task` y sus competidoras del mismo turno. */
function scenario(): {
  tools: ToolDefinition[];
  graph: TenantToolGraph;
  catalog: { get(name: string): ToolDefinition | undefined };
  reduced: ScoredTool[];
} {
  const tools: ToolDefinition[] = [
    SCHEDULE_TASK,
    def("web_search", "search", ["buscar", "web", "internet", "consulta"]),
    def("read_news_feed", "search", ["noticias", "feed", "rss", "actualidad"]),
    def("send_whatsapp", "messaging", ["whatsapp", "mensaje", "enviar", "chat"]),
    def("summarize_document", "text", ["resumir", "documento", "sintesis"]),
    def("create_calendar_event", "calendar", ["calendario", "evento", "agenda", "reunion"]),
  ];

  const nodes = new Map<string, ToolGraphNode>();
  const toolIndexMap = new Map<string, number>();
  tools.forEach((tool, i) => {
    nodes.set(tool.name, {
      id: tool.name,
      name: tool.name,
      embedding: unit(COSINES[tool.name] ?? 0),
      prerequisites: [],
      conflicts: [],
      definition: tool,
    });
    toolIndexMap.set(tool.name, i);
  });

  const graph: TenantToolGraph = {
    tenant: TRAINED_SCOPE,
    versionHash: "hash",
    nodes,
    // Sin aristas declaradas: la propagación es 0 y la relevancia aislada es
    // "coseno máximo + término aprendido", que es justo lo que se quiere medir.
    adjacencyMatrix: new Float32Array(tools.length * tools.length),
    toolIndexMap,
  };

  const byName = new Map(tools.map((t) => [t.name, t]));
  return {
    tools,
    graph,
    catalog: { get: (name: string) => byName.get(name) },
    reduced: tools.map((t) => ({ name: t.name, score: COSINES[t.name] ?? 0 })),
  };
}

/** Config hermética: sin ONNX ni Qdrant, con el piso de relevancia del log. */
function makeConfig(overrides: Record<string, string> = {}) {
  return loadConfig({
    ...process.env,
    ONNX_ENABLED: "0",
    QDRANT_ENABLED: "0",
    ONNX_ADAPTIVE_MIN_SCORE: String(MIN_SCORE),
    LEARN_ENABLED: "true",
    ...overrides,
  } as NodeJS.ProcessEnv);
}

/** Predice por la ruta topológica (la del log) con el perfil del canal dado. */
function predictGraph(cfg: ReturnType<typeof makeConfig>, learned: ReturnType<LexicalProfileService["score"]>) {
  const { graph, catalog } = scenario();
  return new RerankService(cfg).graphFilter({
    zt: ZT,
    graph,
    edges: [],
    lexicalScores: new Map(),
    learned,
    catalog,
    modelSize: "hash",
  });
}

/** Entrena un canal: semilla del catálogo + `n` eventos con éxito real. */
function train(cfg: ReturnType<typeof makeConfig>, scopeKey: string, n: number): LexicalProfileService {
  const lexical = new LexicalProfileService(cfg);
  lexical.seedScope(scopeKey, scenario().tools);
  for (let i = 0; i < n; i++) {
    lexical.observe(scopeKey, LOG_OBSERVED, ["schedule_task"]);
  }
  return lexical;
}

// ───────────────────────────────────────────────────────────────────────────
// §1 CORE — la barrera y su ruptura, de forma determinista
// ───────────────────────────────────────────────────────────────────────────

test("§1-A baseline (LEARN_ENABLED=false): el piso de relevancia vacía la respuesta", () => {
  const cfg = makeConfig({ LEARN_ENABLED: "false" });
  const lexical = new LexicalProfileService(cfg);

  // Sin capa de aprendizaje la semilla y los eventos son inertes (§6.5).
  assert.deepEqual(lexical.seedScope(TRAINED_SCOPE, scenario().tools), {
    seeded: 0,
    cached: 0,
    pruned: 0,
  });
  lexical.observe(TRAINED_SCOPE, LOG_OBSERVED, ["schedule_task"]);
  const learned = lexical.score(TRAINED_SCOPE, LOG_PROMPT);
  assert.equal(learned.weight, 0);

  const result = predictGraph(cfg, learned);
  assert.equal(result.tools.length, 0, "el baseline debe reproducir la barrera (0 tools)");
  // La relevancia se queda por debajo del piso: es la causa, no un timeout.
  assert.ok(result.rankedScores[0] < MIN_SCORE, `relevancia ${result.rankedScores[0]} >= ${MIN_SCORE}`);
});

test("§1-B guarda de cold start: por debajo de LEARN_MIN_EVENTS la barrera sigue en pie", () => {
  const cfg = makeConfig();
  const lexical = train(cfg, TRAINED_SCOPE, LEARN_MIN_EVENTS - 1);

  const learned = lexical.score(TRAINED_SCOPE, LOG_PROMPT);
  assert.equal(learned.weight, 0, "sin eventos suficientes la capa no puntúa (§6.5)");

  const result = predictGraph(cfg, learned);
  assert.equal(result.tools.length, 0);
});

test("§1-C canal entrenado (N=LEARN_MIN_EVENTS): el término aprendido ROMPE la barrera", () => {
  const cfg = makeConfig();
  const lexical = train(cfg, TRAINED_SCOPE, LEARN_MIN_EVENTS);

  const learned = lexical.score(TRAINED_SCOPE, LOG_PROMPT);
  assert.equal(learned.weight, LEARN_WEIGHT);
  assert.ok(
    (learned.scores.get("schedule_task") ?? 0) > 0.9,
    "schedule_task debe ser el máximo de la tanda",
  );

  const result = predictGraph(cfg, learned);
  assert.ok(result.tools.length > 0, "la barrera debe romperse");
  assert.equal(result.tools[0].name, "schedule_task");
  // El rescate es exactamente el término aprendido: 0.70 + 0.25 · 1.0 = 0.95.
  assert.ok(
    result.rankedScores[0] >= MIN_SCORE,
    `relevancia ${result.rankedScores[0]} < ${MIN_SCORE}`,
  );
  assert.ok(Math.abs(result.rankedScores[0] - (COSINES.schedule_task + LEARN_WEIGHT)) < 1e-6);
});

test("§1-D la ruta plana (sin grafo) se rompe igual que la topológica", () => {
  const { reduced, catalog } = scenario();

  const before = new RerankService(makeConfig({ LEARN_ENABLED: "false" })).filter(
    reduced,
    new Map(),
    { scores: new Map(), weight: 0, terms: [] },
    catalog,
    "hash",
  );
  assert.equal(before.tools.length, 0);

  const cfg = makeConfig();
  const learned = train(cfg, TRAINED_SCOPE, LEARN_MIN_EVENTS).score(TRAINED_SCOPE, LOG_PROMPT);
  const result = new RerankService(cfg).filter(reduced, new Map(), learned, catalog, "hash");

  assert.ok(result.tools.length > 0);
  assert.equal(result.tools[0].name, "schedule_task");
});

test("§1-E aislamiento (§9.4): el canal entrenado no altera la predicción de otro", () => {
  const cfg = makeConfig();
  const lexical = train(cfg, TRAINED_SCOPE, LEARN_MIN_EVENTS);

  const other = lexical.score(OTHER_SCOPE, LOG_PROMPT);
  assert.equal(other.weight, 0, "un canal sin eventos no puede puntuar");
  assert.equal(other.scores.size, 0, "el perfil ajeno no debe filtrarse");
  assert.equal(predictGraph(cfg, other).tools.length, 0, "el otro canal queda idéntico al baseline");

  // El canal entrenado sí rompe la barrera: la diferencia es sólo el scope.
  assert.equal(predictGraph(cfg, lexical.score(TRAINED_SCOPE, LOG_PROMPT)).tools[0].name, "schedule_task");
});

test("§1-F sin regresión: con la capa inactiva el ranking es idéntico al de hoy", () => {
  const { reduced, catalog } = scenario();
  const coldStart = { scores: new Map(), weight: 0, terms: [] };
  const disabled = new RerankService(makeConfig({ LEARN_ENABLED: "false" })).filter(
    reduced,
    new Map(),
    coldStart,
    catalog,
    "hash",
  );
  const enabled = new RerankService(makeConfig()).filter(
    reduced,
    new Map(),
    coldStart,
    catalog,
    "hash",
  );
  assert.deepEqual(
    enabled.tools.map((t) => t.name),
    disabled.tools.map((t) => t.name),
  );
});

// ───────────────────────────────────────────────────────────────────────────
// §2 E2E — el contrato completo sobre el puerto 6776
// ───────────────────────────────────────────────────────────────────────────

const TENANT = "4f4e8014-1111-1111-1111-111111111111";
const OTHER_TENANT = "otro-usuario-2222-2222-2222-222222222222";

let predictUrl = "";
let closeServer: () => Promise<void> = async () => {};

before(async () => {
  // Hermético: los módulos leen `loadConfig()` del entorno al arrancar.
  process.env.ONNX_ENABLED = "0";
  process.env.QDRANT_ENABLED = "0";
  // El piso se desactiva aquí a propósito: con embeddings hash (0.2-0.4) el
  // escenario A volvería a dar 0 tools en ambos casos y no se podría observar
  // el efecto de la señal aprendida sobre el RANKING. La barrera ya está
  // demostrada en §1, que es hermético y determinista.
  process.env.ONNX_ADAPTIVE_MIN_SCORE = "0";
  process.env.LEARN_ENABLED = "1";
  const server = await startPredictServer(0);
  delete process.env.ONNX_ENABLED;
  delete process.env.QDRANT_ENABLED;
  delete process.env.ONNX_ADAPTIVE_MIN_SCORE;
  delete process.env.LEARN_ENABLED;
  predictUrl = `http://localhost:${server.port}`;
  closeServer = server.close;
});

after(async () => {
  await closeServer();
});

async function post(path: string, body: unknown): Promise<{ status: number; data: any }> {
  const res = await fetch(`${predictUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, data: await res.json() };
}

async function get(path: string): Promise<{ status: number; data: any }> {
  const res = await fetch(`${predictUrl}${path}`);
  return { status: res.status, data: await res.json() };
}

/** Nombres de las herramientas devueltas por /predict. */
function names(body: any): string[] {
  return Array.isArray(body?.tools) ? body.tools.map((t: { name: string }) => t.name) : [];
}

test("§2-A POST /predict/feedback eleva schedule_task en el turno siguiente", async () => {
  const reg = await post("/tools", { tenant: TENANT, tools: scenario().tools });
  assert.equal(reg.status, 200);
  assert.equal(reg.data.indexed, scenario().tools.length);

  const before = await post("/predict", {
    sessionId: "s-before",
    tenant: TENANT,
    text: LOG_PROMPT,
    source: "human",
  });
  assert.equal(before.status, 200);
  const beforeNames = names(before.data);

  for (let i = 0; i < LEARN_MIN_EVENTS; i++) {
    const fb = await post("/predict/feedback", {
      tenant: TENANT,
      sessionId: "s-before",
      text: LOG_OBSERVED,
      used: ["schedule_task"],
    });
    assert.equal(fb.status, 200);
    assert.equal(fb.data.ok, true);
  }

  const debug = (await get(`/debug?tenant=${TENANT}`)).data;
  assert.ok((debug.lexical?.events ?? 0) >= LEARN_MIN_EVENTS, "el canal debe acumular eventos");

  const after = await post("/predict", {
    sessionId: "s-after",
    tenant: TENANT,
    text: LOG_PROMPT,
    source: "human",
  });
  assert.equal(after.status, 200);
  const afterNames = names(after.data);

  assert.ok(
    afterNames.includes("schedule_task"),
    `esperaba schedule_task tras el feedback, obtuve [${afterNames.join(", ")}]`,
  );
  // Si ya venía en la lista, el refuerzo no puede empeorar su posición.
  if (beforeNames.includes("schedule_task")) {
    assert.ok(afterNames.indexOf("schedule_task") <= beforeNames.indexOf("schedule_task"));
  }
});

test("§2-B aislamiento: el feedback de un canal no toca a otro", async () => {
  const reg = await post("/tools", { tenant: OTHER_TENANT, tools: scenario().tools });
  assert.equal(reg.status, 200);

  const other = (await get(`/debug?tenant=${OTHER_TENANT}`)).data;
  assert.equal(other.lexical?.events ?? 0, 0, "el canal ajeno no debe tener eventos");

  const trained = (await get(`/debug?tenant=${TENANT}`)).data;
  assert.ok((trained.lexical?.events ?? 0) >= LEARN_MIN_EVENTS);
});

test("§2-C /predict/feedback valida el cuerpo (400 sin tenant/sessionId/text)", async () => {
  const res = await post("/predict/feedback", { used: ["schedule_task"] });
  assert.equal(res.status, 400);
});
