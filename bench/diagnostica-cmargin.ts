/**
 * Diagnóstico de la abstención masiva de /predict: instrumenta la corrida REAL
 * del núcleo Variante C (rank + arbitrate) sobre el catálogo de workspace de
 * los tests, imprimiendo margen/autonomía/anáfora/estado/fused-top por turno.
 * Compara el veredicto con C_MARGIN actual (0.13) vs el calibrado (0.065).
 *
 * Uso: tsx bench/diagnostica-cmargin.ts
 */
import { createSystem, loadConfig } from "../src/index";
import { VarianteCService } from "../src/predict/services/variante-c.service";
import { C_MARGIN, C_TAU, C_ANAF_TAU } from "../src/shared/constants/predict/general.predict";
import { EXAMPLE_TOOLS } from "../tests/example-tools";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

process.env.QDRANT_ENABLED = "0";

const CATALOG: ToolDefinition[] = [
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo en el workspace interno del proyecto con el contenido especificado (HTML, CSS, JavaScript, TypeScript, React, Astro, NestJS, Go, Python, SQL, Docker, etc.), páginas web, landing pages, frontend, configuraciones, scripts o seeds.",
    tags: ["workspace", "create file", "crear archivo", "html", "css", "javascript", "web", "pagina web", "landing page"],
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
    id: "workspace_replace_in_file",
    name: "workspace_replace_in_file",
    group: "workspace",
    category: "filesystem",
    description: "Reemplaza un texto exacto dentro de un archivo existente del workspace para actualizar estilos, scripts o estructura.",
    tags: ["workspace", "editar archivo", "actualizar css", "editar html"],
    intentSummary: "Aplica reemplazos puntuales dentro de un archivo existente del proyecto interno",
    inputSchema: { type: "object", properties: { path: { type: "string" }, findText: { type: "string" }, replaceText: { type: "string" } }, required: ["path", "findText", "replaceText"] },
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

const TURNOS = [
  { n: 1, q: "créame una página web para un restaurante con HTML y CSS", esperado: "workspace_write_file", pasaBase: false },
  { n: 2, q: "genera una landing page en el sandbox con index.html, styles.css y script.js", esperado: "workspace_write_file", pasaBase: true },
  { n: 3, q: "crea un archivo styles.css con diseño responsive y animaciones", esperado: "workspace_write_file", pasaBase: false },
  { n: 4, q: "diseña un componente interactivo en javascript app.js", esperado: "workspace_write_file", pasaBase: true },
  { n: 5, q: "actualiza el diseño agregando modo oscuro al styles.css", esperado: "workspace_replace_in_file|workspace_write_file", pasaBase: true },
  { n: 6, q: "inspecciona el código de index.html generado", esperado: "workspace_read_file", pasaBase: true },
  { n: 7, q: "intenta crear de nuevo la página html", esperado: "workspace_write_file", pasaBase: false },
  { n: 8, q: "Reintentar la creación de la página HTML", esperado: "workspace_write_file", pasaBase: true },
  { n: 9, q: "guardar archivo jev-ai.html en el workspace", esperado: "workspace_write_file", pasaBase: true },
  { n: 10, q: "puedes buscar por internet MILO J su canción NIÑO?", esperado: "web_search", pasaBase: true },
];

async function main() {
  const system = await createSystem(loadConfig());
  const c = new VarianteCService(system.engine);
  const scope = "diag-tenant";
  await c.indexScope(scope, CATALOG);
  console.log(`C_MARGIN=${C_MARGIN} C_TAU=${C_TAU} C_ANAF_TAU=${C_ANAF_TAU}\n`);

  for (const t of TURNOS) {
    const r = await c.rank(scope, t.q);
    if (!r) {
      console.log(`#${t.n} SIN RANKING: ${t.q}`);
      continue;
    }
    const arb = c.arbitrate(r);
    const top = r.out[0];
    const autoCon065 = r.margin >= 0.065 && r.anaphora < C_ANAF_TAU;
    console.log(
      `#${t.n} "${t.q}"\n` +
        `   margen=${r.margin.toFixed(4)} autonomous=${r.autonomous} (con 0.065: ${autoCon065}) anaf=${r.anaphora.toFixed(3)} estado=${r.state} hasCtx=${r.hasCtx}\n` +
        `   top=${top?.name}(${top?.fused.toFixed(3)}) s1=${top?.s1.toFixed(3)} s2=${top?.s2.toFixed(3)} → decisión=${arb.decision} tools=[${arb.tools.join(",")}]\n` +
        `   esperado=${t.esperado}`,
    );
  }

  // Escenario e2e: catálogo EXAMPLE_TOOLS, consulta "busca información sobre el clima en lima".
  const scopeE2E = "diag-e2e";
  await c.indexScope(scopeE2E, EXAMPLE_TOOLS as ToolDefinition[]);
  console.log("\n── e2e (EXAMPLE_TOOLS) ──");
  for (const q of [
    "busca información sobre el clima en lima",
    "envía un correo a juan",
    "traduce este texto al inglés",
    "hola, ¿cómo estás?",
    "gracias por todo",
    "hazlo con eso",
    "sí, ese mismo",
  ]) {
    const r = await c.rank(scopeE2E, q);
    if (!r) {
      console.log(`SIN RANKING: ${q}`);
      continue;
    }
    const arb = c.arbitrate(r);
    const top = r.out[0];
    console.log(
      `"${q}"\n   margen=${r.margin.toFixed(4)} autonomous=${r.autonomous} anaf=${r.anaphora.toFixed(3)} estado=${r.state}\n   top=${top?.name}(${top?.fused.toFixed(3)}) s1=${top?.s1.toFixed(3)} s2=${top?.s2.toFixed(3)} → ${arb.decision} [${arb.tools.join(",")}]`,
    );
  }
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
