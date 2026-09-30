/* TEMP: mide estadística del manifold (top/second/mean/std/z/gap) sobre el
 * catálogo real para calibrar el criterio de decisión del juicio cuando el
 * rerank devuelve ∅. Borrar tras usar.
 * Uso: npx tsx scripts/_measure-manifold-stats.ts */
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-production-tools";
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

const CASOS_TEXTO: Array<[string, string]> = [
  ["cat-categoria", 'cambia de categoría "Lugares turísticos JRCM Abogados" a Servicios'],
  ["cat-imagen", "asignar la imagen del ítem La Pichanga Gastrobar con la url http://x/view"],
  ["cat-crear", "crea el item La Pichanga Gastrobar como restaurante en la zona Tumbes con su descripción"],
  ["cat-estadisticas", "muéstrame las estadísticas de clics de la campaña de Fiestas Patrias"],
  ["cat-cache", "purga la caché de cloudflare porque ya publiqué los cambios"],
  ["cat-hero", "configura el hero de la portada con el eslogan Tumbes te espera"],
  ["cat-ruta", "crea una ruta turística que recorra Punta Sal y Zorritos en dos días"],
  ["cat-usuario", "desactiva la cuenta del editor Julieta Montenegro"],
  ["cat-aliado", "registra un nuevo aliado comercial: Hotel Costa Verde"],
  ["cat-ayuda", "qué campos necesita la herramienta item_crear y qué efectos tiene"],
  ["cat-eliminar", "elimina el item Cevichería Javi Valle que quedó duplicado"],
  ["cat-relacionar", "relaciona el item La Pichanga Gastrobar con el aliado Hotel Costa Verde"],
];

const FUERA: Array<[string, string]> = [
  ["dis-hora", "hola que hora es hoy en tumbes"],
  ["dis-derivada", "explícame qué es una derivada parcial"],
  ["dis-telefonia", "quiero cancelar mi suscripción de telefonía"],
  ["dis-haiku", "escribe un haiku sobre el mar"],
  ["dis-ceviche", "necesito una receta de ceviche de conchas negras"],
  ["dis-log", "descargar el log de error del servidor"],
  ["dis-clima", "el clima en lima mañana"],
  ["dis-depa", "busco apartamento en alquiler en miraflores"],
  ["dis-ventas", "dame un resumen de las ventas del mes"],
  ["dis-musica", "pon música relajante para trabajar"],
  ["dis-traduce", "traduce este párrafo al francés"],
];

async function main(): Promise<void> {
  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  console.log(`### model=${cfg.onnxModelSize} rho=${cfg.juicioSpecificityRho} ###`);

  const tools = TOTAL_ACTIVE_TOOLS;
  const nodes = new Map<string, Float32Array>();
  for (const t of tools) {
    nodes.set(t.name, (await engine.embedPassage(toolSignature(t), engine.defaultSize, { high: false })).embedding);
  }
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
  const meanD = total / names.length;
  const penalty = (n: string) => 1 / (1 + cfg.juicioSpecificityRho * Math.max(0, (byTool.get(n) ?? meanD) - meanD));

  const medir = async (label: string, q: string, ok: boolean) => {
    const qe = (await engine.embedQuery(q, engine.defaultSize, { high: true })).embedding;
    const scored = names
      .map((n) => ({ n, s: cosine(qe, nodes.get(n)!) * penalty(n) }))
      .sort((a, b) => b.s - a.s);
    const sum = scored.reduce((a, x) => a + x.s, 0);
    const mean = sum / scored.length;
    const variance = scored.reduce((a, x) => a + (x.s - mean) * (x.s - mean), 0) / scored.length;
    const std = Math.sqrt(variance);
    const top = scored[0];
    const second = scored[1];
    const z = (top.s - mean) / (std || 1e-6);
    const gap = top.s - second.s;
    const gapZ = gap / (std || 1e-6);
    console.log(
      `  ${ok ? "POS" : "DIS"} ${label.padEnd(18)} top=${top.s.toFixed(3)}(${top.n}) 2nd=${second.s.toFixed(3)}(${second.n}) ` +
        `mean=${mean.toFixed(3)} std=${std.toFixed(3)} z=${z.toFixed(2)} gap=${gap.toFixed(3)} gapZ=${gapZ.toFixed(2)}`,
    );
  };

  console.log("\n-- POSITIVOS (catálogo debe resolverlos) --");
  for (const [l, q] of CASOS_TEXTO) await medir(l, q, true);
  console.log("\n-- FUERA DE TEMA / RUIDO --");
  for (const [l, q] of FUERA) await medir(l, q, false);
}

void main();
