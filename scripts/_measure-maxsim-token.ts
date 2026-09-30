/* TEMP: mide MaxSim token-level (doble centrado) sobre el catálogo real para
 * calibrar la corroboración del juicio cuando el rerank devuelve ∅.
 * Borrar tras usar. Uso: npx tsx scripts/_measure-maxsim-token.ts */
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-production-tools";
import { onnxTextPrefix } from "../src/shared/constants/predict/embedding.constants";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

function shortText(text?: string, maxLen = 320): string {
  if (!text) return "";
  const cut = text.search(/\n|[{\[]/);
  const head = cut >= 0 ? text.slice(0, cut) : text;
  return head.replace(/\s+/g, " ").trim().slice(0, maxLen);
}
function signature(t: ToolDefinition): string {
  return [t.name, t.intentSummary, t.description, (t.tags ?? []).join(" ")].filter(Boolean).join(" ");
}
function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += a[i] * b[i];
  return Math.min(1, Math.max(0, d));
}

const POS: Array<[string, string]> = [
  ["cat-categoria", 'cambia de categoría "Lugares turísticos JRCM Abogados" a Servicios'],
  ["cat-imagen", "asignar la imagen del ítem La Pichanga Gastrobar con la url"],
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
const DIS: Array<[string, string]> = [
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
];

async function main(): Promise<void> {
  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  console.log(`### model=${cfg.onnxModelSize} ###`);

  const tools = TOTAL_ACTIVE_TOOLS;
  const toolTokens = new Map<string, Float32Array[]>();
  for (const t of tools) {
    toolTokens.set(
      t.name,
      (await engine.embedTokens(`${onnxTextPrefix[engine.defaultSize].passage}${signature(t)}`, engine.defaultSize, { high: false })).slice(0, cfg.juicioMaxsimMaxTokens),
    );
  }

  const medir = async (label: string, q: string, ok: boolean) => {
    const qt = await engine.embedTokens(`${onnxTextPrefix[engine.defaultSize].query}${q}`, engine.defaultSize, { high: true });
    const m = qt.length;
    const n = tools.length;
    const mat: number[][] = [];
    const rowMeans = new Float64Array(m);
    const colSums = new Float64Array(n);
    for (let i = 0; i < m; i++) {
      const row: number[] = new Array(n);
      let rSum = 0;
      const qv = qt[i];
      for (let j = 0; j < n; j++) {
        let best = 0;
        for (const tv of toolTokens.get(tools[j].name)!) {
          const s = cosine(qv, tv);
          if (s > best) best = s;
        }
        row[j] = best;
        rSum += best;
        colSums[j] += best;
      }
      mat.push(row);
      rowMeans[i] = rSum / n;
    }
    const colMeans = new Float64Array(n);
    let grandSum = 0;
    for (let j = 0; j < n; j++) {
      colMeans[j] = colSums[j] / m;
      grandSum += colMeans[j];
    }
    const grandMean = grandSum / n;

    let bestI = -Infinity, secondI = -Infinity, bestRaw = 0, bestTool = "";
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const interaction = mat[i][j] - rowMeans[i] - colMeans[j] + grandMean;
        if (interaction > bestI) { secondI = bestI; bestI = interaction; bestRaw = mat[i][j]; bestTool = tools[j].name; }
        else if (interaction > secondI) secondI = interaction;
      }
    }
    // MaxSim clásico top-1
    const msAll = tools.map((t) => {
      const vecs = toolTokens.get(t.name)!;
      let acc = 0;
      for (const tv of vecs) { let b = 0; for (const qv of qt) { const s = cosine(qv, tv); if (s > b) b = s; } acc += b; }
      return { n: t.name, s: acc / vecs.length };
    }).sort((a, b) => b.s - a.s);
    console.log(
      `  ${ok ? "POS" : "DIS"} ${label.padEnd(17)} tokens=${m} bestI=${bestI.toFixed(4)} margin=${(bestI - secondI).toFixed(4)} raw=${bestRaw.toFixed(3)} tool=${bestTool} | maxsimTop=${msAll[0].s.toFixed(3)}(${msAll[0].n}) 2nd=${msAll[1].s.toFixed(3)}`,
    );
  };

  console.log("\n-- MAXSIM TOKEN-LEVEL --");
  for (const [l, q] of POS) await medir(l, q, true);
  for (const [l, q] of DIS) await medir(l, q, false);
}

void main();
