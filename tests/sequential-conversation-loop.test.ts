/**
 * sequential-conversation-loop.test.ts
 *
 * Tests unitarios que verifican la resolución matemática de bucles conversacionales (loops)
 * en turnos secuenciales anafóricos ("a ver", "si", "dale") sobre herramientas de lectura
 * versus herramientas de mutación de estado.
 *
 * Fundamento matemático (AGENT.md: sin palabras clave, puramente latente):
 * 1. Proyección sobre arquetipos TOOL_MUTATION_ARCHETYPE vs TOOL_READ_ARCHETYPE.
 * 2. En régimen dependiente (!autonomous && hasCtx):
 *    - Si la herramienta previa es de lectura (!isMutation), el turno anafórico es una
 *      elaboración conversacional sobre el contenido devuelto, por lo que el sistema se abstiene
 *      (tools: []) permitiendo que el planificador/asistente razone sin re-ejecutar la búsqueda.
 *    - Si la herramienta previa es de mutación (isMutation), el turno anafórico ratifica
 *      la ejecución de la acción pendiente (tools: [mutation_tool]).
 */
import { describe, test, before, after } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config";
import { startPredictServer } from "../src/app.module";
import { VarianteCService } from "../src/predict/services/variante-c.service";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { TOOL_MUTATION_ARCHETYPE, TOOL_READ_ARCHETYPE } from "../src/shared/constants/predict/general.predict";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

process.env.QDRANT_ENABLED = "0";

const TENANT_ID = "seq-loop-tenant-" + Date.now();

const CATALOG_TOOLS: ToolDefinition[] = [
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "webSearch",
    description: "Busca en la web e internet información sobre artistas, canciones, noticias, clima o documentación.",
    tags: ["search", "web", "internet", "musica", "cancion"],
    intentSummary: "Busca información en internet",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee e inspecciona el contenido de un archivo en el workspace para consultar su código.",
    tags: ["leer", "read", "inspeccionar"],
    intentSummary: "Consulta el contenido de un archivo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace con código o texto nuevo.",
    tags: ["crear", "escribir", "write"],
    intentSummary: "Crea o sobrescribe un archivo",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "reserve_table",
    name: "reserve_table",
    group: "booking",
    category: "action",
    description: "Realiza una reserva de mesa en el restaurante para un horario y número de comensales.",
    tags: ["reservar", "reserva", "mesa"],
    intentSummary: "Ejecuta una reserva de restaurante",
    inputSchema: { type: "object", properties: { guests: { type: "number" }, time: { type: "string" } }, required: ["guests", "time"] },
  },
];

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;
let engine: EmbeddingEngineService;
let varianteC: VarianteCService;

before(async () => {
  const s = await startPredictServer(0);
  serverUrl = `http://localhost:${s.port}`;
  closeServer = s.close;

  // Registrar herramientas vía API
  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: CATALOG_TOOLS }),
  });
  assert.equal(regRes.status, 200);

  // Instanciar VarianteC directamente para pruebas geométricas unitarias
  const cfg = loadConfig();
  engine = new EmbeddingEngineService(cfg);
  varianteC = new VarianteCService(engine);
  await varianteC.indexScope(TENANT_ID, CATALOG_TOOLS);

  // Esperar warm-up del servidor
  await new Promise((r) => setTimeout(r, 1000));
});

after(async () => {
  if (closeServer) await closeServer();
});

describe("Clasificación de arquetipos latentes (Lectura vs Mutación)", () => {
  test("web_search y workspace_read_file se proyectan como lectura (no mutación)", async () => {
    const mutPassage = await engine.embedPassage(TOOL_MUTATION_ARCHETYPE, engine.defaultSize, { high: false });
    const readPassage = await engine.embedPassage(TOOL_READ_ARCHETYPE, engine.defaultSize, { high: false });

    for (const tool of [CATALOG_TOOLS[0]!, CATALOG_TOOLS[1]!]) {
      const toolEmb = await engine.embedPassage(tool.description, undefined, { high: false });
      const sMut = EmbeddingEngineService.cosine(toolEmb.embedding, mutPassage.embedding);
      const sRead = EmbeddingEngineService.cosine(toolEmb.embedding, readPassage.embedding);
      assert.ok(
        sRead >= sMut,
        `Herramienta ${tool.name} debe proyectarse más cerca de lectura que de mutación (sRead=${sRead.toFixed(4)}, sMut=${sMut.toFixed(4)})`,
      );
    }
  });

  test("workspace_write_file y reserve_table se proyectan como mutación de estado", async () => {
    const mutPassage = await engine.embedPassage(TOOL_MUTATION_ARCHETYPE, engine.defaultSize, { high: false });
    const readPassage = await engine.embedPassage(TOOL_READ_ARCHETYPE, engine.defaultSize, { high: false });

    for (const tool of [CATALOG_TOOLS[2]!, CATALOG_TOOLS[3]!]) {
      const toolEmb = await engine.embedPassage(tool.description, undefined, { high: false });
      const sMut = EmbeddingEngineService.cosine(toolEmb.embedding, mutPassage.embedding);
      const sRead = EmbeddingEngineService.cosine(toolEmb.embedding, readPassage.embedding);
      assert.ok(
        sMut > sRead,
        `Herramienta ${tool.name} debe proyectarse como mutación (sMut=${sMut.toFixed(4)}, sRead=${sRead.toFixed(4)})`,
      );
    }
  });
});

describe("Prevención de loops en turnos secuenciales informativos (Caso Joji)", () => {
  const SESSION_ID = "joji-session-" + Date.now();
  const CONTEXT_TURNO_1 = "busca en internet a JOJI Y SUS ULTIMAS CANCIONES\nJoji volvió a la música con nuevo álbum Piss In The Wind y temas como PIXELATED KISSES.";

  test("Turno 1: consulta inicial autónoma predice web_search", async () => {
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tenant: TENANT_ID,
        sessionId: SESSION_ID,
        text: "busca en internet a JOJI Y SUS ULTIMAS CANCIONES",
        source: "human",
      }),
    });
    const data = await res.json();
    assert.equal(res.status, 200);
    assert.ok(data.tools.length > 0, "Debe predecir herramientas en el turno 1");
    assert.equal(data.tools[0].name, "web_search", "Debe seleccionar web_search");
  });

  test("Turno 2 (secuencial anafórico): 'a ver' con historial NO repite web_search (abstención)", async () => {
    // 1. Verificación unitaria en VarianteC
    const rC = await varianteC.rank(TENANT_ID, "a ver", CONTEXT_TURNO_1);
    assert.ok(rC, "Debe generar ranking");
    assert.equal(rC.autonomous, false, "'a ver' debe ser clasificado como DEPENDIENTE");
    const arb = varianteC.arbitrate(rC);
    assert.equal(
      arb.decision,
      "abstain",
      "VarianteC debe abstenerse al ser dependiente sobre una herramienta de solo lectura",
    );
    assert.deepEqual(arb.tools, []);

    // 2. Verificación end-to-end contra /predict
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tenant: TENANT_ID,
        sessionId: SESSION_ID,
        text: "a ver",
        source: "human",
        history: [
          { role: "user", content: "busca en internet a JOJI Y SUS ULTIMAS CANCIONES" },
          { role: "assistant", content: "Joji regresó con PIXELATED KISSES y el álbum Piss In The Wind. ¿Quieres que profundice?" },
        ],
      }),
    });
    const data = await res.json();
    assert.equal(res.status, 200);
    assert.equal(
      data.tools.length,
      0,
      `'a ver' no debe repetir la búsqueda web en loop. Recibió: ${data.tools.map((t: any) => t.name).join(", ")}`,
    );
  });

  test("Turno 3 (secuencial anafórico): 'si' con historial acumulado NO repite web_search", async () => {
    const CONTEXT_ACUMULADO = CONTEXT_TURNO_1 + "\na ver\nAquí tienes más detalles de los sencillos.";
    const rC = await varianteC.rank(TENANT_ID, "si", CONTEXT_ACUMULADO);
    assert.ok(rC);
    assert.equal(rC.autonomous, false);
    const arb = varianteC.arbitrate(rC);
    assert.equal(arb.decision, "abstain");
    assert.deepEqual(arb.tools, []);

    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tenant: TENANT_ID,
        sessionId: SESSION_ID,
        text: "si",
        source: "human",
        history: [
          { role: "user", content: "busca en internet a JOJI Y SUS ULTIMAS CANCIONES" },
          { role: "assistant", content: "Joji regresó con PIXELATED KISSES." },
          { role: "user", content: "a ver" },
          { role: "assistant", content: "¿Quieres que profundice en alguna canción o en el álbum?" },
        ],
      }),
    });
    const data = await res.json();
    assert.equal(res.status, 200);
    assert.equal(
      data.tools.length,
      0,
      `'si' no debe repetir la búsqueda web en loop. Recibió: ${data.tools.map((t: any) => t.name).join(", ")}`,
    );
  });
});

describe("Ratificación legítima de turnos de confirmación en MUTACIONES", () => {
  const MUTATION_SESSION = "mut-session-" + Date.now();
  const CONTEXT_MUTACION = "crea el archivo styles.css con los estilos de la landing";

  test("Turno 1: orden de creación de archivo predice workspace_write_file", async () => {
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tenant: TENANT_ID,
        sessionId: MUTATION_SESSION,
        text: "crea el archivo styles.css con los estilos de la landing",
        source: "human",
      }),
    });
    const data = await res.json();
    assert.equal(res.status, 200);
    assert.ok(data.tools.length > 0);
    assert.equal(data.tools[0].name, "workspace_write_file");
  });

  test("Turno 2: 'hazlo' sobre mutación RATIFICA la herramienta", async () => {
    // 1. Verificación unitaria en VarianteC
    const rC = await varianteC.rank(TENANT_ID, "hazlo", CONTEXT_MUTACION);
    assert.ok(rC);
    assert.equal(rC.autonomous, false, "'hazlo' es dependiente");
    const arb = varianteC.arbitrate(rC);
    assert.equal(
      arb.decision,
      "ratify",
      "VarianteC debe ratificar legítimamente cuando la acción pendiente es una mutación",
    );
    assert.ok(arb.tools.includes("workspace_write_file"));

    // 2. Verificación en el orquestador
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tenant: TENANT_ID,
        sessionId: MUTATION_SESSION,
        text: "hazlo",
        source: "human",
        history: [
          { role: "user", content: "crea el archivo styles.css con los estilos de la landing" },
          { role: "assistant", content: "¿Deseas proceder a crear el archivo styles.css en el workspace?" },
        ],
      }),
    });
    const data = await res.json();
    assert.equal(res.status, 200);
    assert.ok(data.tools.length > 0, "Debe devolver la herramienta de mutación confirmada");
    assert.equal(data.tools[0].name, "workspace_write_file");
  });
});
