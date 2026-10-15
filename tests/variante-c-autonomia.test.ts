/**
 * Regresión de la autonomía del núcleo Variante C.
 *
 * Fija los dos fallos medidos el 2026-10-12 (bench/diagnostica-cmargin.ts):
 *
 *  1. C_MARGIN por encima del margen real de las acciones (0.13 vs banda
 *     calibrada: conversacional ≤0.060 < C_MARGIN ≤ 0.070 ≤ acción real)
 *     volvía DEPENDIENTE a todo turno y, sin contexto, la fusión colapsa a
 *     fused≈0.025 → abstención masiva ("nunca predice herramientas").
 *
 *  2. Un turno SIN contexto no puede ser dependiente: no hay contexto que lo
 *     resuelva. Las órdenes autónomas de margen bajo en catálogos densos
 *     ("busca el clima en lima" = 0.055 con 12 tools) deben juzgarse con la
 *     fusión del propio turno. El turno anafórico sin contexto sigue en ∅.
 */
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { createSystem, loadConfig } from "../src/index";
import { VarianteCService } from "../src/predict/services/variante-c.service";
import { C_MARGIN } from "../src/shared/constants/predict/general.predict";
import { EXAMPLE_TOOLS } from "./example-tools";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

process.env.QDRANT_ENABLED = "0";

const WORKSPACE_CATALOG: ToolDefinition[] = [
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace interno del proyecto con el contenido especificado (HTML, CSS, JavaScript, TypeScript, React, Astro, NestJS, Go, Python, SQL, Docker, etc.), páginas web, landing pages, frontend, configuraciones, scripts o seeds.",
    tags: ["workspace", "create file", "crear archivo", "html", "css", "javascript", "web", "pagina web"],
    intentSummary: "Crea o sobrescribe un archivo en el workspace del proyecto (código HTML/CSS/JS, frontend, web, scripts, datos)",
    inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee el contenido de un archivo del workspace del proyecto (HTML, CSS, JS, etc.) para inspeccionar código o verificar estructura.",
    tags: ["workspace", "leer archivo", "inspeccionar", "read file"],
    intentSummary: "Inspecciona el contenido actual de un archivo del proyecto antes de editarlo o usarlo",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "workspace_list",
    name: "workspace_list",
    group: "workspace",
    category: "filesystem",
    description: "Lista los archivos y carpetas del workspace interno en una ruta específica.",
    tags: ["workspace", "list", "archivos"],
    intentSummary: "Lista archivos y carpetas dentro del workspace del proyecto",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  },
  {
    id: "web_search",
    name: "web_search",
    group: "search",
    category: "web_search",
    description: "Busca documentación actualizada o recursos en internet.",
    tags: ["search", "web", "internet", "docs"],
    intentSummary: "Busca información actualizada en internet",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
];

let system: Awaited<ReturnType<typeof createSystem>>;
let varianteC: VarianteCService;

before(async () => {
  system = await createSystem(loadConfig());
  varianteC = new VarianteCService(system.engine);
  await varianteC.indexScope("autonomia-ws", WORKSPACE_CATALOG);
  await varianteC.indexScope("autonomia-e2e", EXAMPLE_TOOLS as ToolDefinition[]);
});

test("C_MARGIN respeta la banda calibrada (conversacional ≤0.060 < C_MARGIN ≤ 0.070 ≤ acción real)", () => {
  assert.ok(
    C_MARGIN > 0.06 && C_MARGIN <= 0.07,
    `C_MARGIN=${C_MARGIN} fuera de la banda: por encima de 0.070 ninguna acción real alcanza el régimen autónomo y todo se abstiene sin contexto`,
  );
});

test("órdenes de margen bajo SIN contexto se juzgan con la fusión del turno (no dependiente)", async () => {
  const casos = [
    { scope: "autonomia-ws", q: "créame una página web para un restaurante con HTML y CSS", espera: "workspace_write_file" },
    { scope: "autonomia-ws", q: "puedes buscar por internet MILO J su canción NIÑO?", espera: "web_search" },
    { scope: "autonomia-e2e", q: "busca información sobre el clima en lima", espera: "web_search" },
    { scope: "autonomia-e2e", q: "envía un correo a juan", espera: "send_email" },
  ];
  for (const c of casos) {
    const r = await varianteC.rank(c.scope, c.q);
    assert.ok(r, `sin ranking para "${c.q}"`);
    assert.equal(r.autonomous, true, `"${c.q}" (margen=${r.margin.toFixed(4)}) debe ser autónomo sin contexto`);
    const arb = varianteC.arbitrate(r!);
    assert.equal(arb.decision, "ratify", `"${c.q}" debe ratificar, decidió ${arb.decision}`);
    assert.ok(
      arb.tools.includes(c.espera),
      `"${c.q}" debe incluir ${c.espera}, obtuvo [${arb.tools.join(",")}]`,
    );
  }
});

test("turno anafórico SIN contexto sigue absteniéndose (no hay qué depender ni resolver)", async () => {
  for (const q of ["hazlo con eso", "sí, ese mismo"]) {
    const r = await varianteC.rank("autonomia-e2e", q);
    assert.ok(r, `sin ranking para "${q}"`);
    assert.equal(r!.autonomous, false, `"${q}" es anafórico y no debe ser autónomo`);
    const arb = varianteC.arbitrate(r!);
    assert.equal(arb.decision, "abstain", `"${q}" sin contexto debe abstenerse`);
    assert.deepEqual(arb.tools, []);
  }
});

test("mención explícita de herramienta ratifica directo (orden del planificador)", async () => {
  const r = await varianteC.rank("autonomia-e2e", "ejecuta web_search con la consulta clima lima");
  assert.ok(r, "sin ranking");
  const arb = varianteC.arbitrate(r!);
  assert.equal(arb.decision, "ratify");
  assert.ok(arb.tools.includes("web_search"), `[${arb.tools.join(",")}] debe incluir web_search`);
});
