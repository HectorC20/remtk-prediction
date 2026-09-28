/**
 * Enriquecimiento interno de la consulta (`intentContext`).
 *
 * El planificador de key-remtk PIENSA antes de escribir el plan y ese bloque se
 * tiraba. El título de una subtarea es deliberadamente atómico: en
 * `nest-2026-09-28 (1).log` línea 119 la pasada de agente terminó ofreciendo
 * `image_generate, ch_list, ch_get, image_describe` para tareas del catálogo de
 * MiTumbes, mientras el razonamiento del mismo modelo (línea 87) nombra el
 * catálogo, las entidades y los uuid que hacen falta.
 *
 * El servidor NO anexa el bloque crudo: lo parte por ideas y se queda con los
 * `MAX_INTENT_SEGMENTS` segmentos más próximos a la consulta entrante. Medido: un
 * segmento que habla de OTRA subtarea puntúa en la misma banda de coseno que uno
 * que habla de ÉSTA (0.83-0.87 vs 0.83-0.93), así que la selección es relativa
 * (top-K) y no por umbral absoluto.
 *
 * Este archivo mide, contra el servidor vivo y el catálogo real de producción
 * (188 herramientas: 98 de MiTumbes + 90 de ruido), lo que la medición del canal
 * dejó establecido —no una mejora supuesta—:
 *   1. Un bloque que habla de otra subtarea no desplaza la herramienta que la
 *      consulta ya fijaba por sí sola.
 *   2. El bloque es RECALL, no ranking: amplía el pool de candidatas
 *      (`intentPoolAdded`) y deja la tanda de la consulta idéntica. Anexarlo al
 *      ranking —como hacía la primera versión del canal— medía lo contrario y
 *      hundía la acción de la subtarea de escritura.
 *   3. Una subconsulta de agente (`source=agent`) deja de responderse con el plan
 *      cacheado de OTRO turno de la misma sesión —sin eso el enriquecimiento no
 *      llegaría nunca al ranking: el early-exit ocurre antes de construir la
 *      consulta—.
 *   4. El bloque se acota y, sin enriquecimiento, el turno se juzga igual que antes.
 *
 * Que el consumidor (key-remtk) solo envíe el bloque en el replanteo es política
 * del cliente y no puede observarse desde aquí.
 *
 * Ejecución: pnpm test  (o pnpm tsx --test tests/predict-intent-context.test.ts)
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import { TOTAL_ACTIVE_TOOLS } from "./fixtures/mitumbes-production-tools";
import { MAX_INTENT_CONTEXT_CHARS } from "../src/shared/constants/predict";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "predict-intent-context-2026-09-28";

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: TOTAL_ACTIVE_TOOLS }),
  });
  assert.equal(regRes.status, 200, "El registro del catálogo debe responder 200");
  await fetch(`${serverUrl}/tools/ready?tenant=${TENANT_ID}`);
});

after(async () => {
  if (closeServer) await closeServer();
});

/** POST /predict y devuelve los nombres de herramienta en el orden recibido. */
async function predictNombres(body: Record<string, unknown>): Promise<string[]> {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  assert.equal(res.status, 200, `/predict debe responder 200 (tenant=${TENANT_ID})`);
  const data = (await res.json()) as { tools?: ToolDefinition[] };
  return (data.tools ?? []).map((t) => t.name);
}

/** Traza del último turno de una sesión, leída en /debug. */
async function trazaDe(sessionId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${serverUrl}/debug?tenant=${TENANT_ID}`);
  assert.equal(res.status, 200);
  const debug = (await res.json()) as { traces?: Record<string, unknown>[] };
  const mias = (debug.traces ?? []).filter((t) => t.sessionId === sessionId);
  assert.ok(mias.length > 0, `/debug debe tener la traza de la sesión ${sessionId}`);
  return mias[mias.length - 1];
}

/**
 * Pre-pensamiento LITERAL del planificador registrado en
 * `nest-2026-09-28 (3).log` línea 112: habla de la IMAGEN del ítem y de los uuid
 * de categoría/zona, así que para la subtarea de zona es un bloque CRUZADO.
 */
const PRE_IMAGEN =
  'El usuario quiere actualizar la imagen del ítem "La Pichanga Gastrobar" que fue creado anteriormente. El ítem tiene ID 3265a5e6-86a2-4156-a23f-39cfd7ff1171 y slug la-pichanga-gastrobar.\n' +
  "El usuario adjuntó una imagen con fileId ff726d32-7eeb-4f3c-acfa-051405530812 y URL pública http://backend.remtk.com/api/v1/files/public/ff726d32-7eeb-4f3c-acfa-051405530812/view.\n" +
  "Necesito planificar tareas:\n" +
  "1. Buscar/verificar si existe una herramienta para actualizar ítems de mitumbes (modo consultar).\n" +
  "2. Ejecutar la actualización de la imagen del ítem con la URL pública proporcionada (modo ejecutar).";

/**
 * Pre-pensamiento LITERAL de `nest-2026-09-28 (1).log` línea 87 (inglés, el idioma
 * en que el modelo razona): dominio mitumbes, categoría Restaurante, zona Tumbes.
 */
const PRE_PENSAMIENTO =
  'The user wants to create the item "La Pichanga Gastrobar" in the mitumbes catalog with an improved description, in multiple languages. Category is Restaurant, zone is Tumbes. ' +
  "There's a tool `mitumbes_item_crear` that exists and is active but requires categoryId and zoneId as UUIDs. " +
  'The category "Restaurante" and zone "Tumbes" need UUIDs. So I need to consult tools to get the category and zone UUIDs: ' +
  'Task 0: consultar - Obtener UUIDs de categorías del catálogo (to find "Restaurante" categoryId). ' +
  'Task 1: consultar - Obtener UUIDs de zonas del catálogo (to find "Tumbes" zoneId).';

/** Historial de OTROS dominios (imagen, redes, correo): el ruido real del log. */
const HISTORIAL_RUIDO = [
  { role: "user", content: "genera una imagen para el local y publícala en instagram" },
  { role: "assistant", content: "creé la imagen y la publiqué en la red social" },
  { role: "user", content: "ahora envía un correo con el flyer a los socios" },
  {
    role: "user",
    content:
      "crea el item La Pichanga Gastrobar, mejora su descripción 🎶✨ ¡Los fines de semana hay música en vivo!, categoría Restaurante, zona Tumbes, dirección Huascar 560, Tumbes 24001",
  },
];

/** POST /predict como lo hace el ejecutor v5: subconsulta de agente con su subtarea. */
function cuerpoAgente(sessionId: string, texto: string, extra: Record<string, unknown> = {}) {
  return {
    sessionId,
    tenant: TENANT_ID,
    text: texto,
    source: "agent",
    priorPlan: texto,
    ...extra,
  };
}

describe("intentContext: el pre-pensamiento del planificador enriquece la consulta de predicción", () => {
  test("un bloque que describe OTRA subtarea no desplaza la herramienta que la consulta ya fijaba", async () => {
    // Es el riesgo del canal: el razonamiento abarca todo el plan. Si el bloque
    // mandara, la selección por segmentos habría fallado.
    const zonaSin = await predictNombres(
      cuerpoAgente("sints-cruzada-zona", "Consultar UUID de la zona 'Tumbes' en el catálogo"),
    );
    const zonaCon = await predictNombres(
      cuerpoAgente("sintent-cruzada-zona", "Consultar UUID de la zona 'Tumbes' en el catálogo", {
        intentContext: PRE_IMAGEN,
      }),
    );
    const imagenSin = await predictNombres(
      cuerpoAgente("sints-consultada", "Actualizar imagen del ítem La Pichanga Gastrobar", {
        history: [{ role: "user", content: "adjunta esta imagen al item La Pichanga Gastrobar, es la foto nueva del local" }],
      }),
    );
    const imagenCon = await predictNombres(
      cuerpoAgente("sintent-consultada", "Actualizar imagen del ítem La Pichanga Gastrobar", {
        history: [{ role: "user", content: "adjunta esta imagen al item La Pichanga Gastrobar, es la foto nueva del local" }],
        intentContext: PRE_IMAGEN,
      }),
    );

    console.log(
      `  [cruzada] zona sin=[${zonaSin.slice(0, 3).join(", ")}] con=[${zonaCon.slice(0, 3).join(", ")}]\n` +
        `  [cruzada] imagen sin=[${imagenSin.slice(0, 3).join(", ")}] con=[${imagenCon.slice(0, 3).join(", ")}]`,
    );

    assert.equal(
      zonaCon[0],
      "mitumbes_zona_listar",
      `el bloque de la imagen no puede mover el puntero de la subtarea de zona: [${zonaCon.join(", ")}]`,
    );
    assert.equal(
      zonaSin[0],
      zonaCon[0],
      "el puntero de una consulta afinada debe permanecer idéntico",
    );
    assert.equal(
      imagenCon[0],
      "mitumbes_item_imagen_adjuntar",
      `el pre-pensamiento del plan no puede desbancar la herramienta que la consulta ya tenía en punta: [${imagenCon.join(", ")}]`,
    );
    assert.equal(imagenSin[0], imagenCon[0], "lo mismo, sin enriquecimiento");
  });

  test("el bloque ANCHA el pool de candidatas sin reescribir la tanda de la consulta", async () => {
    // La subtarea llega después de un historial de otros dominios. Medido sobre
    // `nest-2026-09-28 (3).log`, lo que el bloque aporta es recall: mete
    // candidatas en el pool (`intentPoolAdded`) que compiten con las de la
    // consulta, mientras la tanda de una consulta ya afinada queda idéntica.
    // La primera versión de este canal ANEXABA el bloque al ranking y medía lo
    // contrario: `mitumbes_item_imagen_adjuntar` desaparecía del top-5 de la
    // subtarea de escritura.
    const sin = await predictNombres(
      cuerpoAgente("sints-familia", "Actualizar imagen del ítem La Pichanga Gastrobar", {
        history: HISTORIAL_RUIDO,
      }),
    );
    const con = await predictNombres(
      cuerpoAgente("sintent-familia", "Actualizar imagen del ítem La Pichanga Gastrobar", {
        history: HISTORIAL_RUIDO,
        intentContext: PRE_IMAGEN,
      }),
    );
    const traza = await trazaDe("sintent-familia");
    console.log(
      `  [familia] sin=[${sin.join(", ")}]\n  [familia] con=[${con.join(", ")}] ` +
        `pool_agregadas=${String(traza.intentPoolAdded)}`,
    );

    assert.ok(Number(traza.intentContextChars) > 0, "el bloque debe haberse seleccionado y acotado");
    assert.ok(
      Number(traza.intentPoolAdded) > 0,
      "el bloque trae candidatas que la consulta cruda no encontró",
    );
    assert.deepEqual(
      con,
      sin,
      `el canal es aditivo: la tanda de la consulta no cambia por leer el pre-pensamiento [${sin.join(", ")}] vs [${con.join(", ")}]`,
    );
    assert.ok(
      !con.some((n) => n.startsWith("stripe_") || /^(image_generate|image_describe|ch_list|ch_get)$/.test(n)),
      `la tanda del catálogo no se llena con las herramientas ajenas que devolvió el log: [${con.join(", ")}]`,
    );
  });

  test("una subconsulta de agente no se responde con el plan cacheado de otro turno de la sesión", async () => {
    // Turno HUMANO: escribe el cache de confirmación de la sesión (estado del diálogo).
    const humano = await predictNombres({
      sessionId: "sa-cache-mixto",
      tenant: TENANT_ID,
      text: "listar las zonas del catálogo de mitumbes",
      source: "human",
    });
    assert.ok(humano.length > 0, "el turno humano debe dejar un plan cacheable");

    // Subtarea de AGENTE en la MISMA sesión, redacción breve y anafórica: antes
    // del fix el clasificador la leía como ConfirmationEmpty y devolvía lo del
    // turno humano sin mirar la consulta real.
    const agente = await predictNombres({
      sessionId: "sa-cache-mixto",
      tenant: TENANT_ID,
      text: "buscar categorías",
      source: "agent",
      priorPlan: "buscar categorías",
    });
    console.log(
      `  [cache] humano=[${humano.slice(0, 3).join(", ")}] agente=[${agente.slice(0, 3).join(", ")}]`,
    );

    assert.notDeepEqual(
      agente,
      humano,
      "la subconsulta del agente no puede recibir el plan del turno humano",
    );
    assert.ok(
      agente.some((n) => n.startsWith("mitumbes_categoria")),
      `la subconsulta del agente debe resolverse con su propia consulta (categorías): [${agente.join(", ")}]`,
    );

    // Y el agente no contamina el cache: la confirmación del usuario sigue
    // recuperando el plan del turno HUMANO previo.
    const confirmacion = await predictNombres({
      sessionId: "sa-cache-mixto",
      tenant: TENANT_ID,
      text: "hazlo",
      source: "human",
    });
    assert.deepEqual(
      confirmacion,
      humano,
      "el plan cacheado debe seguir siendo el del turno humano, no el de una subtarea",
    );
  });

  test("el bloque largo se acota: no desplaza el texto de la subtarea", async () => {
    const enorme = `${PRE_PENSAMIENTO} ${"stripe_charge_create ".repeat(800)}`;
    const sessionId = "sintent-coto";
    const nombres = await predictNombres({
      sessionId,
      tenant: TENANT_ID,
      text: "Consultar UUID de la zona 'Tumbes' en el catálogo",
      source: "agent",
      priorPlan: "Consultar UUID de la zona 'Tumbes' en el catálogo",
      intentContext: enorme,
    });
    const traza = await trazaDe(sessionId);

    assert.ok(
      typeof traza.intentContextChars === "number" &&
        traza.intentContextChars <= MAX_INTENT_CONTEXT_CHARS,
      `intentContext debe acotarse a ${MAX_INTENT_CONTEXT_CHARS} chars, llegó ${String(traza.intentContextChars)}`,
    );
    assert.ok(
      nombres.includes("mitumbes_zona_listar"),
      `con el tope aplicado la herramienta correcta sigue presente: [${nombres.join(", ")}]`,
    );
    assert.ok(
      !nombres.some((n) => n.startsWith("stripe_")),
      `el relleno del bloque no puede convertir la subtarea en otra categoría: [${nombres.join(", ")}]`,
    );
  });

  test("sin `intentContext` el comportamiento del turno es idéntico al de antes", async () => {
    // Regresión: el canal es aditivo. Un turno sin enriquecimiento no debe ver
    // alterado su ranking por los cambios de la capa de texto.
    const texto = "crear un ítem nuevo en el catálogo de mitumbes con su categoría y zona";
    const a = await predictNombres({
      sessionId: "sintregresion-a",
      tenant: TENANT_ID,
      text: texto,
      source: "human",
    });
    const b = await predictNombres({
      sessionId: "sintregresion-b",
      tenant: TENANT_ID,
      text: texto,
      source: "human",
      intentContext: "   ",
    });
    assert.deepEqual(b, a, "un intentContext vacío no debe mover ni una herramienta");
  });
});
