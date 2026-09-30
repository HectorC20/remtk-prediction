/* TEMP: mide los cosenos del subsistema de skills con el modelo activo, para
 * calibrar umbrales por modelo. Borrar tras usar.
 * Uso: npx tsx scripts/_measure-skill-thresholds.ts           (modelo por defecto = gte)
 *      $env:ONNX_MODEL_SIZE="large"; npx tsx scripts/_measure-skill-thresholds.ts  (e5-large) */
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += a[i] * b[i];
  return Math.min(1, Math.max(0, d));
}

async function main(): Promise<void> {
  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  console.log(`### model=${cfg.onnxModelSize} ###`);

  const embP = async (t: string) => (await engine.embedPassage(t, engine.defaultSize, { high: false })).embedding;
  const embQ = async (t: string) => (await engine.embedQuery(t, engine.defaultSize, { high: true })).embedding;

  const skillCrear = await embP(
    "Crear ítem en catálogo: crear nuevo item lugar hotel restaurante actividad en el catálogo Crea un nuevo ítem turístico, restaurante, hotel o actividad",
  );
  const skillImg = await embP(
    "Actualizar imagen o foto de ítem: actualizar cambiar subir asignar imagen foto del item generado o existente Actualiza la imagen de portada o foto principal de un ítem existente",
  );
  const skillGastro = await embP(
    "Gestión de Restaurantes y Bares: crear restaurante gastronomico Crear y actualizar restaurantes, gastrobares y locales de comida",
  );
  const skills: Array<[string, Float32Array]> = [
    ["crear", skillCrear],
    ["imagen", skillImg],
    ["gastronomia", skillGastro],
  ];

  const anclas: Array<[string, string]> = [
    ["categoryId", "Restaurante y Gastronomía"],
    ["categoryId", "Hoteles y Hospedajes"],
    ["categoryId", "Playas y Atractivos Naturales"],
    ["zoneId", "Tumbes Centro"],
    ["zoneId", "Punta Sal"],
    ["zoneId", "Zorritos"],
  ];

  // Positivos: el turno SÍ describe una skill/ancla.
  const positivos: Array<[string, string]> = [
    ["crear-hotel", "crea un hotel en punta sal llamado Karibian"],
    ["actualiza-img", "actualiza la imagen de ese item que se ha generado"],
    ["img-indirecta", "te pido que actualices la imagen"],
  ];
  // Distractores: el turno NO describe ninguna skill/ancla del scope.
  const distractores: Array<[string, string]> = [
    ["saludo", "hola, buenos dias, como estas"],
    ["clima", "cual es el clima hoy en la ciudad de lima"],
    ["borrar", "elimina todos los registros del sistema"],
    ["ventas", "dame un resumen de las ventas del mes"],
    ["musica", "pon musica relajante para trabajar"],
    ["mate", "resuelve esta ecuacion diferencial"],
    ["chiste", "cuentame un chiste corto"],
    ["traduce", "traduce este parrafo al frances"],
  ];

  const skillMax = async (q: string) => {
    const qe = await embQ(q);
    let best = 0;
    let who = "";
    for (const [n, e] of skills) {
      const s = cosine(qe, e);
      if (s > best) {
        best = s;
        who = n;
      }
    }
    return { best, who, qe };
  };

  console.log("\n-- SKILL MATCH --");
  const all: Array<[string, string, boolean]> = [
    ...positivos.map(([l, q]) => [l, q, true] as [string, string, boolean]),
    ...distractores.map(([l, q]) => [l, q, false] as [string, string, boolean]),
  ];
  const actionEmb = await embP("image upload");
  for (const [l, q, ok] of all) {
    const { best, who } = await skillMax(q);
    const actionSim = cosine(await embQ(q), actionEmb);
    console.log(
      `  ${ok ? "POS " : "DIS "} ${l.padEnd(14)} skillMax=${best.toFixed(4)} (${who}) actionSim=${actionSim.toFixed(4)}`,
    );
  }

  console.log("\n-- ANCLAS (piso y margen intra-clave) --");
  for (const [l, q, ok] of all) {
    const qe = await embQ(q);
    const out: string[] = [];
    for (const key of ["categoryId", "zoneId"]) {
      const ranked: Array<{ name: string; sim: number }> = [];
      for (const [k, name] of anclas.filter((a) => a[0] === key)) {
        ranked.push({ name, sim: cosine(qe, await embP(`${k}: ${name}`)) });
      }
      ranked.sort((a, b) => b.sim - a.sim);
      out.push(`${key}: top=${ranked[0].sim.toFixed(4)}(${ranked[0].name}) margen=${(ranked[0].sim - ranked[1].sim).toFixed(4)}`);
    }
    console.log(`  ${ok ? "POS " : "DIS "} ${l.padEnd(14)} ${out.join(" | ")}`);
  }

  // Escenario EXACTO del test multitenant: solo 2 skills (gastronomia + inmuebles),
  // query "te pido que actualices la imagen" -> ¿qué score tiene la mejor skill?
  const skillInmuebles = await embP(
    "Gestión Inmobiliaria: crear propiedad inmobiliaria Crear y actualizar propiedades, departamentos y casas",
  );
  const qm = await embQ("te pido que actualices la imagen");
  console.log("\n-- Escenario multitenant (gastronomia + inmuebles) --");
  console.log(`  gastronomia=${cosine(qm, skillGastro).toFixed(4)} inmuebles=${cosine(qm, skillInmuebles).toFixed(4)}`);
}

void main();
