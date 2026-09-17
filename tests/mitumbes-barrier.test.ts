/**
 * Fase 3b — ¿Predice el pipeline las herramientas del log
 * `key-remtk/tests/docs/respuesta.md`?
 *
 * Caso real: el usuario pidió *"cambia de categoría "Lugares turísticos JRCM
 * Abogados" a servicios"* y el mini-agente quemó 3 ciclos (9 procesos,
 * 12k-41k tokens) porque la predicción no le devolvió el flujo correcto:
 * resolver el ítem (`item_obtener`, acepta slug o uuid) + resolver el uuid de
 * la categoría (`categoria_listar`, que NO recibe parámetros) + escribir
 * (`item_actualizar`, que exige `categoryId` uuid, no `categorySlug`).
 *
 * Aquí se recrea ese turno con las **45 herramientas reales** de MiTumbes
 * (`c:\dev\projects\mitumbes-server`) tal como llegan al servidor de
 * predicción después de `plugin-tools-adapter.service.ts`, y se responde una
 * sola pregunta: ¿la predicción devuelve las herramientas requeridas?
 *
 *   §1 CORE — determinista, sin ONNX ni Qdrant: catálogo reducido al flujo del
 *      log con cosenos fijados a mano (la herramienta correcta queda POR
 *      DEBAJO del piso de relevancia, igual que en el log). A (baseline) = 0
 *      tools · C (tras feedback) = flujo completo en el top.
 *   §2 E2E — HTTP 6776 con el catálogo real de 45 tools y el modelo e5-small
 *      real: `/tools` → `/predict` → `/predict/feedback` → `/predict`.
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { startPredictServer } from "../src/app.module";
import { LexicalProfileService } from "../src/predict/services/lexical-profile.service";
import { RerankService } from "../src/predict/services/rerank.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";
import type { TenantToolGraph, ToolGraphNode } from "../src/shared/interfaces/graph.interface";

// ───────────────────────────────────────────────────────────────────────────
// Catálogo real de MiTumbes tal como entra a la predicción
// ───────────────────────────────────────────────────────────────────────────
//
// Cadena de conversión (mi-tumbes → servidor de predicción):
//   mitumbes-server  register-tool.ts   name = `mitumbes_${name}`
//   key-remtk        plugin-tools-adapter.service.ts
//                    · name VERBATIM (sin prefijo extra)
//                    · group "plugin_mitumbes", category "plugin_tool"
//                    · tags [displayName] → ["MiTumbes"]
//                    · sin intentSummary → ""
//                    · description = describeTool(): original + sufijo del complemento
//   key-remtk        PredictionToolsSyncService.toPredictionDefinition() → POST /tools

const PLUGIN_GROUP = "plugin_mitumbes";
const PLUGIN_CATEGORY = "plugin_tool";
const PLUGIN_TAG = "MiTumbes";

/** Réplica de `plugin-tools-adapter.service.ts#describeTool()`. */
function describeTool(name: string, description: string): string {
  return (
    `${description || name}\n(Complemento: ${PLUGIN_TAG} — ` +
    `ejecuta la tool "${name}" del servidor MCP externo)`
  );
}

/**
 * JSON Schema como lo entrega el SDK MCP (el `inputSchema` de Zod llega al
 * adapter ya convertido): las claves de primer nivel son palabras del estándar
 * (`type`, `properties`, `required`), NO los nombres de los parámetros.
 */
function jsonSchema(params: Record<string, string>, required: string[] = []): Record<string, unknown> {
  return {
    type: "object",
    properties: Object.fromEntries(Object.entries(params).map(([k, t]) => [k, { type: t }])),
    ...(required.length > 0 ? { required } : {}),
    additionalProperties: false,
    $schema: "http://json-schema.org/draft-07/schema#",
  };
}

function mitumbesTool(
  name: string,
  description: string,
  params: Record<string, string> = {},
  required: string[] = [],
): ToolDefinition {
  const full = `mitumbes_${name}`;
  return {
    id: full,
    name: full,
    group: PLUGIN_GROUP,
    category: PLUGIN_CATEGORY,
    description: describeTool(full, description),
    tags: [PLUGIN_TAG],
    intentSummary: "",
    inputSchema: jsonSchema(params, required),
  };
}

/** Entidad CRUD con su descripción EXACTA y los parámetros de cada operación. */
interface CrudSpec {
  entity: string;
  /** `port.title` (solo documental: la predicción no recibe el título). */
  title: string;
  /** `port.description` verbatim. */
  description: string;
  list?: Record<string, string>;
  get?: Record<string, string>;
  create?: Record<string, string>;
  update?: Record<string, string>;
  /** `operations.remove === false` (usuario). */
  noRemove?: boolean;
}

/** Réplica de `crud.factory.ts#registerCrudTools()` (plantillas de descripción). */
function crudTools(spec: CrudSpec): ToolDefinition[] {
  const d = spec.description.toLowerCase();
  const out: ToolDefinition[] = [];
  if (spec.list) out.push(mitumbesTool(`${spec.entity}_listar`, `Lista ${d}, con filtros opcionales y paginación.`, spec.list));
  if (spec.get) out.push(mitumbesTool(`${spec.entity}_obtener`, `Obtiene ${d} (por id o slug).`, spec.get));
  if (spec.create) out.push(mitumbesTool(`${spec.entity}_crear`, `Crea ${d} en el sistema.`, spec.create));
  if (spec.update) out.push(mitumbesTool(`${spec.entity}_actualizar`, `Actualiza los campos indicados de ${d} existente.`, spec.update));
  if (!spec.noRemove && spec.get) out.push(mitumbesTool(`${spec.entity}_eliminar`, `Elimina permanentemente ${d} por su id.`, spec.get));
  return out;
}

const PAGINATION = { limit: "integer", offset: "integer" };
const LOCALIZED = { title: "object", description: "object", body: "object" };

const ITEM_DESC =
  "contenido del catálogo unificado (lugares, actividades, eventos, rutas y servicios). " +
  "Filtros por slugs legibles: type (place|activity|event|route|service), categorySlug (nivel 1, ej. activities), " +
  "subcategorySlug (ej. surf), zoneSlug (ej. punta-sal), parentSlug y search.";

const CATEGORIA_DESC =
  "categorías de contenido del sistema (title, description y body en es/en/pt; parentId para crear subcategorías)";

const LUGAR_DESC =
  "lugares turísticos del sistema (title, description, excerpt, address, hours, price, services, howToGet, " +
  "activities, source y body en es/en/pt; slug legible para la URL; image o imageUrl para la portada). " +
  "Filtros: categorySlug (ej. activities), subcategorySlug (ej. surf), zoneSlug (ej. punta-sal) y categoryId";

const EVENTO_DESC =
  "eventos del sistema (title, description, excerpt, hours, price, services, howToGet, activities, source y " +
  "body en es/en/pt; slug legible para la URL; image o imageUrl para la portada). " +
  "Filtros: zoneSlug (ej. punta-sal), subcategorySlug y zone";

const MITUMBES_TOOLS: ToolDefinition[] = [
  ...crudTools({
    entity: "item",
    title: "Ítem",
    description: ITEM_DESC,
    list: {
      type: "string",
      categorySlug: "string",
      subcategorySlug: "string",
      zoneSlug: "string",
      parentSlug: "string",
      search: "string",
      dateFrom: "string",
      dateTo: "string",
      isActive: "boolean",
      ...PAGINATION,
    },
    get: { id: "string" },
    create: { slug: "string", type: "string", ...LOCALIZED, categoryId: "string", zoneId: "string" },
    update: {
      id: "string",
      type: "string",
      ...LOCALIZED,
      excerpt: "object",
      address: "string",
      hours: "string",
      price: "string",
      services: "array",
      howToGet: "string",
      activities: "array",
      source: "string",
      categoryId: "string",
      subcategoryId: "string",
      zoneId: "string",
      parentItemId: "string",
      attributes: "object",
      links: "array",
      imageUrl: "string",
      image: "string",
      slug: "string",
      latitude: "number",
      longitude: "number",
      dateFrom: "string",
      dateTo: "string",
      isActive: "boolean",
    },
  }),
  ...crudTools({
    entity: "categoria",
    title: "Categoría",
    description: CATEGORIA_DESC,
    // `schemas.list = {}`: la tool NO recibe ningún parámetro.
    list: {},
    get: { id: "string" },
    create: { slug: "string", title: "object", description: "object", body: "object", icon: "string", parentId: "string", sortOrder: "integer" },
    update: { id: "string", slug: "string", title: "object", description: "object", body: "object", icon: "string", parentId: "string", sortOrder: "integer" },
  }),
  ...crudTools({
    entity: "lugar",
    title: "Lugar",
    description: LUGAR_DESC,
    list: { categoryId: "string", categorySlug: "string", subcategorySlug: "string", zoneSlug: "string", isActive: "boolean", ...PAGINATION },
    get: { id: "string" },
    create: { slug: "string", ...LOCALIZED, excerpt: "object", address: "string", hours: "string", price: "string", services: "array", howToGet: "string", activities: "array", source: "string", image: "string", imageUrl: "string", categoryId: "string", subcategoryId: "string", zoneId: "string", latitude: "number", longitude: "number", isActive: "boolean" },
    update: { id: "string", slug: "string", ...LOCALIZED, excerpt: "object", address: "string", hours: "string", price: "string", services: "array", howToGet: "string", activities: "array", source: "string", image: "string", imageUrl: "string", categoryId: "string", subcategoryId: "string", zoneId: "string", latitude: "number", longitude: "number", isActive: "boolean" },
  }),
  ...crudTools({
    entity: "evento",
    title: "Evento",
    description: EVENTO_DESC,
    list: { zone: "string", subcategorySlug: "string", zoneSlug: "string", isActive: "boolean", ...PAGINATION },
    get: { id: "string" },
    create: { slug: "string", ...LOCALIZED, excerpt: "object", hours: "string", price: "string", services: "array", howToGet: "string", activities: "array", source: "string", image: "string", imageUrl: "string", zoneId: "string", subcategoryId: "string", dateFrom: "string", dateTo: "string", isActive: "boolean" },
    update: { id: "string", slug: "string", ...LOCALIZED, excerpt: "object", hours: "string", price: "string", services: "array", howToGet: "string", activities: "array", source: "string", image: "string", imageUrl: "string", zoneId: "string", subcategoryId: "string", dateFrom: "string", dateTo: "string", isActive: "boolean" },
  }),
  ...crudTools({
    entity: "ruta",
    title: "Ruta",
    description: "rutas turísticas del sistema",
    list: { isActive: "boolean", ...PAGINATION },
    get: { id: "string" },
    create: { slug: "string", ...LOCALIZED, image: "string", imageUrl: "string", isActive: "boolean" },
    update: { id: "string", slug: "string", ...LOCALIZED, image: "string", imageUrl: "string", isActive: "boolean" },
  }),
  ...crudTools({
    entity: "zona",
    title: "Zona",
    description: "zonas geográficas (playas, distritos, ciudades) de la región",
    list: {},
    get: { id: "string" },
    create: { slug: "string", name: "string", type: "string", description: "string" },
    update: { id: "string", slug: "string", name: "string", type: "string", description: "string" },
  }),
  ...crudTools({
    entity: "usuario",
    title: "Usuario",
    description: "cuentas del sistema de administración",
    list: {},
    get: { id: "string" },
    create: { email: "string", name: "string", roleId: "string", isActive: "boolean" },
    update: { id: "string", name: "string", roleId: "string", isActive: "boolean" },
    noRemove: true,
  }),
  ...crudTools({
    entity: "rol",
    title: "Rol",
    description: "roles del sistema",
    list: {},
    get: { id: "string" },
    create: { name: "string", description: "string", permissionCodes: "array" },
    update: { id: "string", name: "string", description: "string", permissionCodes: "array" },
  }),
  // Extras fuera del CRUD (`tools/index.ts` + `analiticas.tools.ts`).
  mitumbesTool(
    "item_relacionar",
    "Crea (o reordena) una relación dirigida entre dos ítems del catálogo unificado: " +
      "located_in (source está dentro de target), nearby (source es cercano a target), " +
      "related (relacionados) o part_of (source es parte de target).",
    { source: "string", target: "string", relationType: "string", sortOrder: "integer" },
  ),
  mitumbesTool(
    "item_desrelacionar",
    "Elimina una relación dirigida entre dos ítems del catálogo unificado.",
    { source: "string", target: "string", relationType: "string" },
  ),
  mitumbesTool("rol_permisos", "Reemplaza los permisos de un rol por los códigos indicados.", {
    id: "string",
    permissionCodes: "array",
  }),
  mitumbesTool(
    "usuario_desactivar",
    "Desactiva una cuenta (borrado lógico). La cuenta owner no se desactiva.",
    { id: "string" },
  ),
  mitumbesTool(
    "analitica_resumen",
    "Resumen analítico de la actividad del sitio en los últimos días.",
    { days: "integer", limit: "integer" },
  ),
  mitumbesTool(
    "auditoria_listar",
    "Lista los registros de auditoría del sistema con filtros opcionales.",
    { userId: "string", action: "string", limit: "integer", offset: "integer" },
  ),
];

function mitumbesDef(name: string): ToolDefinition {
  const found = MITUMBES_TOOLS.find((t) => t.name === name);
  assert.ok(found, `falta la tool de MiTumbes ${name}`);
  return found;
}

// ───────────────────────────────────────────────────────────────────────────
// Escenario del log
// ───────────────────────────────────────────────────────────────────────────

/** Petición real (log `respuesta.md`). */
const P_CHANGE = 'cambia de categoría "Lugares turísticos JRCM Abogados" a servicios';
const P_ID = "quiero saber el id de jrcm abogados";
const P_PUT = "ponlo en la categoría de servicios";

/**
 * Flujo requerido por el turno. No existe una tool de "cambiar categoría" y
 * `item_actualizar` exige `categoryId` (uuid), no `categorySlug`: hay que
 * resolver el uuid con `categoria_listar` antes de escribir.
 */
const NEED_CHANGE = ["mitumbes_item_obtener", "mitumbes_categoria_listar", "mitumbes_item_actualizar"];
const NEED_ID = ["mitumbes_item_obtener"];
const NEED_PUT = ["mitumbes_categoria_listar", "mitumbes_item_actualizar"];

/**
 * Texto del evento de refuerzo: lo que el ejecutor reporta como subtarea
 * cumplida (no el prompt literal). Comparte léxico con la consulta, que es lo
 * que hace generalizar al perfil del canal (§6.2 del plan).
 */
const OBS_CHANGE =
  "usar mitumbes item obtener para leer el item jrcm abogados y mitumbes item actualizar " +
  "para cambiar su categoria a servicios con el id de mitumbes categoria listar";
const OBS_ID = "usar mitumbes item obtener para saber el id del item jrcm abogados";
const OBS_PUT =
  "usar mitumbes item actualizar para poner el item jrcm abogados en la categoria servicios " +
  "con el id de mitumbes categoria listar";

/** Piso de relevancia de producción (`minScoreDefault`), el del log. */
const MIN_SCORE = 0.8;
/** Peso del término aprendido (`learnWeightDefault`). */
const LEARN_WEIGHT = 0.25;
/** Señales mínimas del canal antes de puntuar (`learnMinEventsDefault`). */
const LEARN_MIN_EVENTS = 3;

const TRAINED_SCOPE = "mitumbes-canal-0000-0000-0000-000000000000";
const OTHER_SCOPE = "mitumbes-ajeno-0000-0000-0000-000000000000";

// ───────────────────────────────────────────────────────────────────────────
// §1 CORE — determinista: catálogo reducido al flujo del log
// ───────────────────────────────────────────────────────────────────────────

/**
 * Cosenos fijados a mano contra `z_t`. El flujo correcto
 * (`item_obtener` 0.74 / `categoria_listar` 0.71 / `item_actualizar` 0.69)
 * queda POR DEBAJO del piso de 0.80: es la situación medida en el log (las
 * tools existen en el catálogo pero el coseno no alcanza el mínimo, y el
 * agente termina iterando sin plan).
 */
const CORE_COSINES: Record<string, number> = {
  mitumbes_item_obtener: 0.74,
  mitumbes_categoria_listar: 0.71,
  mitumbes_item_actualizar: 0.69,
  mitumbes_item_listar: 0.66,
  mitumbes_lugar_listar: 0.58,
  mitumbes_evento_listar: 0.52,
  mitumbes_usuario_listar: 0.44,
};

const ZT = new Float32Array([1, 0]);

/** Vector unitario en 2D cuyo coseno respecto de `ZT` es exactamente `cos`. */
function unit(cos: number): Float32Array {
  return new Float32Array([cos, Math.sqrt(Math.max(0, 1 - cos * cos))]);
}

function coreScenario(): {
  tools: ToolDefinition[];
  graph: TenantToolGraph;
  catalog: { get(name: string): ToolDefinition | undefined };
} {
  const tools = Object.keys(CORE_COSINES).map(mitumbesDef);

  const nodes = new Map<string, ToolGraphNode>();
  const toolIndexMap = new Map<string, number>();
  tools.forEach((tool, i) => {
    nodes.set(tool.name, {
      id: tool.name,
      name: tool.name,
      embedding: unit(CORE_COSINES[tool.name]),
      // Sin aristas: la relevancia aislada mide solo coseno + término aprendido
      // (+ la semilla léxica del catálogo).
      prerequisites: [],
      conflicts: [],
      definition: tool,
    });
    toolIndexMap.set(tool.name, i);
  });

  const byName = new Map(tools.map((t) => [t.name, t]));
  return {
    tools,
    graph: {
      tenant: TRAINED_SCOPE,
      versionHash: "hash",
      nodes,
      adjacencyMatrix: new Float32Array(tools.length * tools.length),
      toolIndexMap,
    },
    catalog: { get: (name: string) => byName.get(name) },
  };
}

/** Config hermética: sin ONNX ni Qdrant, con el piso de relevancia del log. */
function coreConfig(overrides: Record<string, string> = {}) {
  return loadConfig({
    ...process.env,
    ONNX_ENABLED: "0",
    QDRANT_ENABLED: "0",
    ONNX_ADAPTIVE_MIN_SCORE: String(MIN_SCORE),
    ONNX_ADAPTIVE_MIN_TOOLS: "2",
    ONNX_ADAPTIVE_MAX_TOOLS: "5",
    ONNX_ADAPTIVE_GAP_THRESHOLD: "0.15",
    LEARN_ENABLED: "true",
    ...overrides,
  } as NodeJS.ProcessEnv);
}

function predictCore(
  cfg: ReturnType<typeof coreConfig>,
  learned: ReturnType<LexicalProfileService["score"]>,
) {
  const { graph, catalog } = coreScenario();
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

/** Semilla del catálogo + un evento de refuerzo de `n` herramientas con éxito. */
function trainCore(
  cfg: ReturnType<typeof coreConfig>,
  scopeKey: string,
  used: string[],
): LexicalProfileService {
  const lexical = new LexicalProfileService(cfg);
  lexical.seedScope(scopeKey, coreScenario().tools);
  lexical.observe(scopeKey, OBS_CHANGE, used);
  return lexical;
}

function namesOf(result: { tools: ToolDefinition[] }): string[] {
  return result.tools.map((t) => t.name);
}

test("§1-A baseline: el piso de relevancia vacía el plan (0 tools), igual que en el log", () => {
  const cfg = coreConfig({ LEARN_ENABLED: "false" });
  const lexical = new LexicalProfileService(cfg);

  // Sin capa de aprendizaje la semilla y los eventos son inertes (§6.5).
  assert.deepEqual(lexical.seedScope(TRAINED_SCOPE, coreScenario().tools), {
    seeded: 0,
    cached: 0,
    pruned: 0,
  });
  const learned = lexical.score(TRAINED_SCOPE, P_CHANGE);
  assert.equal(learned.weight, 0);

  const result = predictCore(cfg, learned);
  assert.equal(result.tools.length, 0, "el baseline reproduce la barrera (0 tools)");
  assert.ok(
    result.rankedScores[0] < MIN_SCORE,
    `relevancia ${result.rankedScores[0]} >= ${MIN_SCORE}`,
  );
});

test("§1-B guarda de cold start: por debajo de LEARN_MIN_EVENTS la barrera sigue en pie", () => {
  const cfg = coreConfig();
  const lexical = trainCore(cfg, TRAINED_SCOPE, NEED_CHANGE.slice(0, 2)); // 2 eventos < 3

  const learned = lexical.score(TRAINED_SCOPE, P_CHANGE);
  assert.equal(learned.weight, 0, "sin eventos suficientes la capa no puntúa (§6.5)");
  assert.equal(predictCore(cfg, learned).tools.length, 0);
});

test("§1-C canal entrenado: el flujo requerido del log entra completo en el plan", () => {
  const cfg = coreConfig();
  const lexical = trainCore(cfg, TRAINED_SCOPE, NEED_CHANGE); // 3 eventos ≥ 3

  const learned = lexical.score(TRAINED_SCOPE, P_CHANGE);
  assert.equal(learned.weight, LEARN_WEIGHT);

  const result = predictCore(cfg, learned);
  const got = namesOf(result);

  assert.ok(result.tools.length > 0, "la barrera debe romperse");
  for (const name of NEED_CHANGE) {
    assert.ok(got.includes(name), `esperaba ${name} en el plan, obtuve [${got.join(", ")}]`);
  }
  // `item_obtener` es el máximo de la tanda: es el término que abre el flujo.
  assert.equal(result.tools[0].name, "mitumbes_item_obtener");
  assert.ok(
    result.rankedScores[0] >= MIN_SCORE,
    `relevancia ${result.rankedScores[0]} < ${MIN_SCORE}`,
  );
});

test("§1-D aislamiento (§9.4): entrenar un canal no altera la predicción de otro", () => {
  const cfg = coreConfig();
  const lexical = trainCore(cfg, TRAINED_SCOPE, NEED_CHANGE);

  const other = lexical.score(OTHER_SCOPE, P_CHANGE);
  assert.equal(other.weight, 0, "un canal sin eventos no puede puntuar");
  assert.equal(other.scores.size, 0, "el perfil ajeno no debe filtrarse");
  assert.equal(predictCore(cfg, other).tools.length, 0, "el otro canal queda idéntico al baseline");

  // El canal entrenado sí rompe la barrera: la diferencia es sólo el scope.
  assert.ok(
    namesOf(predictCore(cfg, lexical.score(TRAINED_SCOPE, P_CHANGE))).includes(
      "mitumbes_item_actualizar",
    ),
  );
});

test("§1-E sin regresión: con la capa inactiva el ranking es idéntico al de hoy", () => {
  const { graph, catalog } = coreScenario();
  const coldStart = { scores: new Map<string, number>(), weight: 0, terms: [] as string[] };
  const args = {
    zt: ZT,
    graph,
    edges: [],
    lexicalScores: new Map<string, number>(),
    learned: coldStart,
    catalog,
    modelSize: "hash",
  };
  const disabled = new RerankService(coreConfig({ LEARN_ENABLED: "false" })).graphFilter(args);
  const enabled = new RerankService(coreConfig()).graphFilter(args);
  assert.deepEqual(namesOf(enabled), namesOf(disabled));
});

// ───────────────────────────────────────────────────────────────────────────
// §2 E2E — catálogo real (45 tools) + modelo e5-small real, sobre 6776
// ───────────────────────────────────────────────────────────────────────────

const TENANT = "mitumbes-canal-1111-1111-1111-111111111111";
const OTHER_TENANT = "mitumbes-ajeno-2222-2222-2222-222222222222";

let predictUrl = "";
let closeServer: () => Promise<void> = async () => {};

before(async () => {
  // Hermético: los módulos leen `loadConfig()` del entorno al arrancar.
  process.env.ONNX_ENABLED = "1";
  process.env.ONNX_MODELS_PATH = "./models";
  process.env.QDRANT_ENABLED = "0";
  process.env.ONNX_ADAPTIVE_MIN_SCORE = String(MIN_SCORE);
  process.env.LEARN_ENABLED = "1";
  const server = await startPredictServer(0);
  delete process.env.ONNX_ENABLED;
  delete process.env.ONNX_MODELS_PATH;
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

function names(body: any): string[] {
  return Array.isArray(body?.tools) ? body.tools.map((t: { name: string }) => t.name) : [];
}

/** El warm-up (embeddings de keywords + grafo) corre en background tras /tools. */
async function waitWarm(timeoutMs = 120_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const dbg = (await get(`/debug?tenant=${TENANT}`)).data;
    const st = dbg?.stats ?? {};
    if ((st.embeddingsRecomputed ?? 0) + (st.embeddingsCached ?? 0) >= MITUMBES_TOOLS.length) return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("el warm-up del scope no terminó a tiempo");
}

async function ask(sessionId: string, text: string, tenant = TENANT): Promise<string[]> {
  const res = await post("/predict", { sessionId, tenant, text, source: "human" });
  assert.equal(res.status, 200, `POST /predict falló para "${text}"`);
  return names(res.data);
}

async function reinforce(sessionId: string, text: string, used: string[], tenant = TENANT): Promise<void> {
  const res = await post("/predict/feedback", { tenant, sessionId, text, used });
  assert.equal(res.status, 200);
  assert.equal(res.data.ok, true);
}

test("§2-A catálogo real: 45 tools de MiTumbes indexadas y warm-up listo", { timeout: 300_000 }, async () => {
  assert.equal(MITUMBES_TOOLS.length, 45, "el catálogo de MiTumbes tiene 45 tools");
  const names45 = new Set(MITUMBES_TOOLS.map((t) => t.name));
  assert.equal(names45.size, 45, "no debe haber nombres duplicados");

  const reg = await post("/tools", { tenant: TENANT, tools: MITUMBES_TOOLS });
  assert.equal(reg.status, 200);
  assert.equal(reg.data.indexed, 45);

  await waitWarm();
  const count = (await get(`/tools/count?tenant=${TENANT}`)).data;
  assert.equal(count.count, 45, "el catálogo del scope debe tener las 45 tools");

  const dbg = (await get(`/debug?tenant=${TENANT}`)).data;
  assert.equal(dbg.engine?.size, "small", "la predicción debe correr con e5-small real, no hash");
});

test("§2-B turno del log: la predicción devuelve el flujo requerido tras el feedback", { timeout: 300_000 }, async () => {
  const before = await ask("s-change-antes", P_CHANGE);
  console.log(`[§2-B] antes  → [${before.join(", ")}]`);

  await reinforce("s-change", OBS_CHANGE, NEED_CHANGE);
  const dbg = (await get(`/debug?tenant=${TENANT}`)).data;
  assert.ok((dbg.lexical?.events ?? 0) >= LEARN_MIN_EVENTS, "el canal debe acumular eventos");

  const after = await ask("s-change-despues", P_CHANGE);
  console.log(`[§2-B] después → [${after.join(", ")}]`);

  for (const name of NEED_CHANGE) {
    assert.ok(after.includes(name), `esperaba ${name} en el plan, obtuve [${after.join(", ")}]`);
  }
});

test("§2-C consulta del id: `item_obtener` en el plan", { timeout: 300_000 }, async () => {
  await reinforce("s-id", OBS_ID, NEED_ID);
  const after = await ask("s-id-despues", P_ID);
  console.log(`[§2-C] después → [${after.join(", ")}]`);
  for (const name of NEED_ID) {
    assert.ok(after.includes(name), `esperaba ${name} en el plan, obtuve [${after.join(", ")}]`);
  }
});

test("§2-D «ponlo en la categoría de servicios»: escritura + resolución del uuid", { timeout: 300_000 }, async () => {
  await reinforce("s-put", OBS_PUT, NEED_PUT);
  const after = await ask("s-put-despues", P_PUT);
  console.log(`[§2-D] después → [${after.join(", ")}]`);
  for (const name of NEED_PUT) {
    assert.ok(after.includes(name), `esperaba ${name} en el plan, obtuve [${after.join(", ")}]`);
  }
});

test("§2-E aislamiento: el canal ajeno no recibe el entrenamiento de MiTumbes", { timeout: 300_000 }, async () => {
  const reg = await post("/tools", { tenant: OTHER_TENANT, tools: MITUMBES_TOOLS });
  assert.equal(reg.status, 200);

  const other = (await get(`/debug?tenant=${OTHER_TENANT}`)).data;
  assert.equal(other.lexical?.events ?? 0, 0, "el canal ajeno no debe tener eventos");

  const trained = (await get(`/debug?tenant=${TENANT}`)).data;
  assert.ok((trained.lexical?.events ?? 0) >= LEARN_MIN_EVENTS);
});

test("§2-F /predict/feedback valida el cuerpo (400 sin tenant/sessionId/text)", { timeout: 300_000 }, async () => {
  const res = await post("/predict/feedback", { used: NEED_CHANGE });
  assert.equal(res.status, 400);
});
