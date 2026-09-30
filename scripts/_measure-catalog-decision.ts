/* TEMP: mide la proyección sobre el catálogo (el criterio del juicio cuando el
 * rerank devuelve ∅) con el modelo activo. Borrar tras usar.
 * Uso: npx tsx scripts/_measure-catalog-decision.ts */
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-production-tools";
import { matchTokenSet, nameAffinity, toolIdentityTokens } from "../src/predict/keywords";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

function shortText(text?: string, maxLen = 320): string {
  if (!text) return "";
  const cut = text.search(/\n|[{\[]/);
  const head = cut >= 0 ? text.slice(0, cut) : text;
  return head.replace(/\s+/g, " ").trim().slice(0, maxLen);
}
function toolSignature(t: ToolDefinition): string {
  return [t.name, t.group, t.category, shortText(t.description), (t.tags ?? []).join(" "), shortText(t.intentSummary)]
    .filter(Boolean)
    .join(" ");
}
function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += a[i] * b[i];
  return Math.min(1, Math.max(0, d));
}

async function main(): Promise<void> {
  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  console.log(`### model=${cfg.onnxModelSize} rho=${cfg.juicioSpecificityRho} ###`);

  const tools = TOTAL_ACTIVE_TOOLS;
  const nodes = new Map<string, Float32Array>();
  for (const t of tools) {
    nodes.set(t.name, (await engine.embedPassage(toolSignature(t), engine.defaultSize, { high: false })).embedding);
  }
  console.log(`nodes=${nodes.size}`);

  // Difusión (penalización por generalidad), igual que EspecificidadService.
  const names = [...nodes.keys()];
  const byTool = new Map<string, number>();
  let total = 0;
  for (const a of names) {
    const ea = nodes.get(a)!;
    let acc = 0;
    for (const b of names) {
      if (a === b) continue;
      acc += cosine(ea, nodes.get(b)!);
    }
    const d = acc / (names.length - 1);
    byTool.set(a, d);
    total += d;
  }
  const mean = total / names.length;
  const penalty = (n: string) => 1 / (1 + cfg.juicioSpecificityRho * Math.max(0, (byTool.get(n) ?? mean) - mean));

  // Positivos: el turno SÍ debe resolver a una herramienta del catálogo.
  const positivos: Array<[string, string]> = [
    ["categoria", 'cambia de categoría "Lugares turísticos JRCM Abogados" a Servicios'],
    ["imagen", "asignar la imagen del ítem La Pichanga Gastrobar con la url http://x/y.png"],
    ["crear", "crea el item La Pichanga Gastrobar como restaurante en la zona Tumbes con su descripción"],
    ["estadisticas", "muéstrame las estadísticas de clics de la campaña de Fiestas Patrias"],
    ["cache", "purga la caché de cloudflare porque ya publiqué los cambios"],
    ["hero", "configura el hero de la portada con el eslogan Tumbes te espera"],
    ["ruta", "crea una ruta turística que recorra Punta Sal y Zorritos en dos días"],
    ["usuario", "desactiva la cuenta del editor Julieta Montenegro"],
    ["aliado", "registra un nuevo aliado comercial: Hotel Costa Verde"],
    ["ayuda", "qué campos necesita la herramienta item_crear y qué efectos tiene"],
    ["eliminar", "elimina el item Cevichería Javi Valle que quedó duplicado"],
    ["relacionar", "relaciona el item La Pichanga Gastrobar con el aliado Hotel Costa Verde"],
  ];
  // Distractores REALES del test §3 (FUERA_DE_TEMA).
  const distractores: Array<[string, string]> = [
    ["tiempo", "hola que hora es hoy en tumbes"],
    ["derivada", "explícame qué es una derivada parcial"],
    ["telefonia", "quiero cancelar mi suscripción de telefonía"],
    ["haiku", "escribe un haiku sobre el mar"],
    ["ceviche", "necesito una receta de ceviche de conchas negras"],
    ["logerror", "descargar el log de error del servidor"],
    ["clima", "el clima en lima mañana"],
    ["alquiler", "busco apartamento en alquiler en miraflores"],
  ];

  // Anclaje simbólico: max afinidad entre los tokens de la consulta y el NOMBRE.
  const anchor = (q: string): { best: string; aff: number } => {
    const qt = matchTokenSet(q);
    let best = "";
    let aff = 0;
    for (const t of tools) {
      const a = nameAffinity(qt, t.name);
      if (a > aff) {
        aff = a;
        best = t.name;
      }
    }
    return { best, aff };
  };

  const medir = async (label: string, q: string, ok: boolean) => {
    const qe = (await engine.embedQuery(q, engine.defaultSize, { high: true })).embedding;
    const scored = names
      .map((n) => ({ n, s: cosine(qe, nodes.get(n)!) * penalty(n) }))
      .sort((a, b) => b.s - a.s);
    const sum = scored.reduce((a, x) => a + x.s, 0);
    const meanScore = sum / scored.length;
    const top = scored[0];
    const prom = top.s - meanScore;
    const gap = top.s - scored[1].s;
    const an = anchor(q);
    console.log(
      `  ${ok ? "POS" : "DIS"} ${label.padEnd(12)} top=${top.s.toFixed(3)}(${top.n}) prom=${prom.toFixed(3)} gap=${gap.toFixed(3)} ` +
        `anch=${an.aff.toFixed(2)}(${an.best.slice(-24)})`,
    );
  };

  console.log("\n-- PROYECCIÓN SOBRE EL CATÁLOGO (capacidad × penalización) --");
  for (const [l, q] of positivos) await medir(l, q, true);
  for (const [l, q] of distractores) await medir(l, q, false);

  // ¿La afinidad de identidad separa por sí sola?
  const idHits = (q: string) => {
    const qt = matchTokenSet(q);
    return tools.filter((t) => toolIdentityTokens(t.name).some((x) => qt.has(x))).map((t) => t.name);
  };
  console.log("\n-- ANCLAJE (tools con token de identidad en la consulta) --");
  for (const [l, q, ok] of [...positivos.map((p) => [p[0], p[1], true] as const), ...distractores.map((p) => [p[0], p[1], false] as const)]) {
    const hits = idHits(q);
    console.log(`  ${ok ? "POS" : "DIS"} ${l.padEnd(12)} ancladas=${hits.length} [${hits.slice(0, 6).join(", ")}]`);
  }
}

void main();
