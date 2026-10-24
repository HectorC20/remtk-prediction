/**
 * Tests Unitarios de Diagnóstico: Secuencia Conversacional y Enrutamiento Fast-Path
 *
 * Reproduce exactamente el caso reportado por el usuario:
 * Turno 1: "Puedes buscar ppor internet MILO J su canción NIÑO ?"
 * Turno 2: "qué herramientas propias se tiene ?"
 *
 * Demuestra matemáticamente y a nivel de arquitectura por qué ambas consultas
 * son derivadas erróneamente a FAST_PATH en lugar de llegar al PLANIFICADOR.
 */
import { describe, test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "fastpath-diagnosis-tenant-" + Date.now();

const CATALOG_TOOLS: ToolDefinition[] = [
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "webSearch",
    description:
      "Busca información y contenidos en internet y la web sobre cualquier tema, consultas, preguntas de actualidad, canciones, música, noticias, personas, empresas o documentación técnica consultando varios proveedores en paralelo.",
    tags: ["search", "web", "internet", "documentation", "news", "musica", "cancion"],
    intentSummary: "Busca información general, canciones o datos en internet y la web",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "workspaceExecution",
    description: "Escribe contenido en un archivo del workspace o sandbox.",
    tags: ["archivo", "escribir", "crear"],
    intentSummary: "Crea o escribe un archivo en el espacio de trabajo",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
];

before(async () => {
  const s = await startPredictServer(0);
  serverUrl = `http://localhost:${s.port}`;
  closeServer = s.close;

  // Registrar catálogo
  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: CATALOG_TOOLS }),
  });
  assert.equal(regRes.status, 200, "El registro de herramientas debe responder 200");

  // Esperar warm-up
  await new Promise((r) => setTimeout(r, 2000));
});

after(async () => {
  if (closeServer) await closeServer();
});

describe("Diagnóstico de derivación a FAST_PATH vs PLANIFICADOR", () => {
  /**
   * Simula con total fidelidad la compuerta de decisión en aimcp-v6.service.ts (líneas 841-848):
   *
   * const topScore = initialPrediction?.rankedScores?.[0] ?? 0;
   * const topCalibrated = initialPrediction?.calibratedScores?.[0] ?? 0;
   * const noTools =
   *   !initialPrediction?.tools ||
   *   initialPrediction.tools.length === 0 ||
   *   (topScore < 0.35 && topCalibrated < 0.40);
   * const isSimple = initialPrediction?.complexity === "simple";
   * if (noTools && isSimple) {
   *   isFastPath = true; // Salta al interpretador directo, NUNCA llega al planificador
   * }
   */
  function evaluateAimcpV6FastPathGate(prediction: any): {
    topScore: number;
    topCalibrated: number;
    hasToolsInList: boolean;
    noTools: boolean;
    isSimple: boolean;
    isFastPath: boolean;
    rootCause: string;
  } {
    const topScore = prediction?.rankedScores?.[0] ?? 0;
    const topCalibrated = prediction?.calibratedScores?.[0] ?? 0;
    const hasToolsInList = Array.isArray(prediction?.tools) && prediction.tools.length > 0;
    const noTools =
      !prediction?.tools ||
      prediction.tools.length === 0 ||
      (topScore < 0.35 && topCalibrated < 0.40);
    const isSimple = prediction?.complexity === "simple";
    const isFastPath = noTools && isSimple;

    let rootCause = "LLEGA_AL_PLANIFICADOR";
    if (isFastPath) {
      if (hasToolsInList && (topScore < 0.35 && topCalibrated < 0.40)) {
        rootCause =
          `remtk-prediction sí devolvió [${prediction.tools.map((t: any) => t.name).join(", ")}], pero con topScore=${topScore.toFixed(3)} < 0.35 y topCalibrated=${topCalibrated.toFixed(3)} < 0.40. El umbral excesivo en aimcp-v6 lo descarta como 'noTools=true' y fuerza FAST_PATH.`;
      } else if (!hasToolsInList) {
        rootCause =
          "remtk-prediction devolvió 0 herramientas (piso de relevancia no superado o meta-pregunta).";
      }
    }

    return { topScore, topCalibrated, hasToolsInList, noTools, isSimple, isFastPath, rootCause };
  }

  test("Turno 1: 'Puedes buscar ppor internet MILO J su canción NIÑO ?' -> Por qué cae en FAST_PATH", async () => {
    const prompt = "Puedes buscar ppor internet MILO J su canción NIÑO ?";
    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: prompt,
        source: "human",
        tenant: TENANT_ID,
        sessionId: "session-milo-test",
      }),
    });
    assert.equal(res.status, 200);
    const data = (await res.json()) as any;

    // 1. Verificar lo que remtk-prediction devolvió
    const toolNames = (data.tools || []).map((t: any) => t.name);
    assert.ok(
      toolNames.includes("web_search"),
      `remtk-prediction DEBIÓ predecir web_search, obtenido: ${toolNames.join(", ")}`,
    );

    // 2. Evaluar la compuerta de key-remtk (aimcp-v6)
    const gate = evaluateAimcpV6FastPathGate(data);

    // Demostración del fallo del sistema:
    assert.equal(gate.hasToolsInList, true, "remtk-prediction seleccionó web_search correctamente");
    assert.ok(gate.topScore < 0.35, `topScore (${gate.topScore}) es menor a 0.35`);
    assert.ok(gate.topCalibrated < 0.40, `topCalibrated (${gate.topCalibrated}) es menor a 0.40`);
    assert.equal(gate.noTools, true, "El umbral artificial de aimcp-v6 anuló la predicción convirtiéndola en noTools");
    assert.equal(gate.isFastPath, true, "Cae en FAST_PATH y evade el planificador");

    console.log("\n[DIAGNÓSTICO TURNO 1]");
    console.log("Prompt:", prompt);
    console.log("Herramienta predicha por remtk-prediction:", toolNames);
    console.log("Puntajes:", { topScore: gate.topScore, topCalibrated: gate.topCalibrated });
    console.log("Compuerta aimcp-v6:", gate.rootCause);
  });

  test("Turno 2 (Secuencial con contexto): 'qué herramientas propias se tiene ?' -> Por qué cae en FAST_PATH", async () => {
    const promptTurn2 = "qué herramientas propias se tiene ?";
    const history = [
      { role: "user", content: "Puedes buscar ppor internet MILO J su canción NIÑO ?" },
      {
        role: "assistant",
        content:
          'No puedo buscar en internet en este momento, pero te cuento lo que sé sobre MILO J y su canción "NIÑO"...',
      },
    ];

    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: promptTurn2,
        source: "human",
        tenant: TENANT_ID,
        sessionId: "session-milo-test",
        history,
      }),
    });
    assert.equal(res.status, 200);
    const data = (await res.json()) as any;

    const gate = evaluateAimcpV6FastPathGate(data);

    // Verificación de resolución exitosa en el Turno 2 (catálogo general entregado con score alto):
    assert.equal(gate.hasToolsInList, true, "Debe incluir las herramientas del catálogo");
    assert.ok(gate.topScore >= 0.9, `topScore (${gate.topScore}) debe ser alto (>= 0.9)`);
    assert.ok(gate.topCalibrated >= 0.9, `topCalibrated (${gate.topCalibrated}) debe ser alto (>= 0.9)`);
    assert.equal(gate.noTools, false, "No debe considerarse noTools");
    assert.equal(gate.isFastPath, false, "NO debe caer en FAST_PATH y debe llegar al planificador");

    console.log("\n[DIAGNÓSTICO TURNO 2]");
    console.log("Prompt:", promptTurn2);
    console.log("Herramientas:", (data.tools || []).map((t: any) => t.name));
    console.log("Puntajes:", { topScore: gate.topScore, topCalibrated: gate.topCalibrated });
    console.log("Compuerta aimcp-v6:", gate.rootCause);
  });
});
