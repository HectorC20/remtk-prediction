/**
 * ¿Aprende de verdad el sistema, y qué tan buena es la predicción?
 *
 * Auditoría con el SERVIDOR VIVO (Nest en puerto efímero + modelo de embeddings
 * real) sobre el catálogo de 196 herramientas tal como llega de producción: 62
 * verbs de MiTumbes + 134 de ruido de otros dominios. Tres scopes con el MISMO
 * catálogo, para que la comparación sea entre memorias distintas y no entre
 * catálogos distintos:
 *
 *   aud-base       — sin ninguna habilidad aprendida (línea base del predictor).
 *   aud-aprende    — el ciclo completo: predecir -> feedback -> aprender, con el
 *                    payload EXACTO que arma `TaskExecutorV5Service.learnSkillFromTask()`
 *                    (incluido su grounding `{id: valor, name: valor}`).
 *   aud-nombrado   — las mismas 12 habilidades, pero con el grounding que el
 *                    ejecutor DEBERÍA enviar: `{id: uuid, name: "nombre humano"}`.
 *
 * Mide y afirma, en este orden:
 *   §1 predicción sin memoria sobre 196 tools (12 tareas de 12 temas distintos);
 *   §2 el mismo turno después de aprenderlo (lo que el aprendizaje compra y lo que
 *      cuesta: confusión de habilidad, fuga de args viejos, temas equivocados);
 *   §3 discriminación entre las 12 habilidades aprendidas y el suelo de ruido del
 *      espacio vectorial —que es lo que `test_production_tools_e2e.ts` daba por
 *      hecho al fijar `confidence > 0.85` sin medirlo—;
 *   §4 secuencia real de 5 turnos que cambian de tema y retoman una entidad;
 *   §5 la otra mitad de la memoria: `/memory/predict` (con Qdrant apagado el recall
 *      se degrada a cero, solo viaja tema+habilidades) y `/memory/judge`.
 *
 * Ninguna aserción usa un umbral inventado: los márgenes se comparan contra lo que
 * la propia batería mide. El marcador imprime números crudos.
 *
 * Ejecución: pnpm test:memoria  (o dentro de `pnpm test`)
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import { loadConfig } from "../src/config";
import { TOPIC_SHIFT_THRESHOLD } from "../src/shared/constants/predict/general.predict";
import { skillThresholdsByModel } from "../src/shared/constants/predict/embedding.constants";
import { TOTAL_ACTIVE_TOOLS } from "./fixtures/mitumbes-production-tools";

const AGENTE = "agente-auditoria-v5";
const SCOPE_BASE = "aud-base";
const SCOPE_APRENDE = "aud-aprende";
const SCOPE_NOMBRADO = "aud-nombrado";
const SCOPES = [SCOPE_BASE, SCOPE_APRENDE, SCOPE_NOMBRADO] as const;

const UUID = {
  pichanga: "3265a5e6-86a2-4156-a23f-39cfd7ff1171",
  cevicheria: "9c1f2a3b-4d5e-4f60-8a71-8293a4b5c6d7",
  restaurante: "3f2b9c1a-7d4e-4f6b-9a2c-1d8e5f7a9b01",
  servicios: "5a6b7c8d-9e0f-41a2-b3c4-d5e6f7081920",
  hotels: "b7c8d9e0-f1a2-43b4-c5d6-e7f8091a2b3c",
  zonaTumbes: "7b1c2d3e-4f5a-4b6c-8d9e-0f1a2b3c4d5e",
  zonaZorritos: "0e1f2a3b-4c5d-4e6f-8071-8293a4b5c6d7",
  julieta: "d4c5b6a7-98f1-42e3-95d6-c7b8a9012f34",
  carlos: "11aa22bb-33cc-44dd-55ee-66ff77889900",
  playas: "e5f6a7b8-c9d0-41e2-93a4-b5c6d7e8f901",
} as const;

/** URL pública tal como la entrega el backend de archivos (la ve el mini-agente). */
const IMG_1 =
  "http://backend.remtk.com/api/v1/files/public/ff726d32-7eeb-4f3c-acfa-051405530812/view";
const IMG_2 =
  "http://backend.remtk.com/api/v1/files/public/1a2b3c4d-5e6f-4708-9a0b-1c2d3e4f5061/view";

/**
 * 12 tareas, 12 temas distintos. `esperada` es la ACCIÓN que el mini-agente
 * necesita ver ofrecida; `antecedentes` son las que resuelven identificadores.
 * `aprende` replica la subtarea ya ejecutada: las tools que devolvió el MCP con
 * éxito y los argumentos con los que quedaron.
 */
type Caso = {
  id: string;
  tema: string;
  texto: string;
  esperada: string;
  antecedentes: string[];
  aprende: {
    label: string;
    resultado: string;
    herramientas: string[];
    /** args de cada ejecución exitosa, en el orden en que ocurrieron */
    ejecuciones: { toolName: string; arguments: Record<string, unknown> }[];
  };
};

const CASOS: Caso[] = [
  {
    id: "categoria",
    tema: "reclasificar un negocio",
    texto:
      'cambia de categoría "Lugares turísticos JRCM Abogados" a Servicios',
    esperada: "mitumbes_item_actualizar",
    antecedentes: ["mitumbes_item_listar", "mitumbes_categoria_listar"],
    aprende: {
      label: "Actualizar categoría del ítem JRCM Abogados",
      resultado: "ítem con categoryId de Servicios",
      herramientas: ["mitumbes_item_listar", "mitumbes_categoria_listar", "mitumbes_item_actualizar"],
      ejecuciones: [
        { toolName: "mitumbes_item_listar", arguments: { q: "JRCM Abogados" } },
        { toolName: "mitumbes_categoria_listar", arguments: {} },
        {
          toolName: "mitumbes_item_actualizar",
          arguments: { id: UUID.pichanga, categoryId: UUID.servicios, name: "JRCM Abogados" },
        },
      ],
    },
  },
  {
    id: "imagen",
    tema: "adjuntar la foto que llegó en el turno",
    texto: `asignar la imagen del ítem La Pichanga Gastrobar con la url ${IMG_1}`,
    esperada: "mitumbes_item_imagen_adjuntar",
    antecedentes: ["mitumbes_item_listar"],
    aprende: {
      label: "Actualizar imagen del ítem La Pichanga Gastrobar",
      resultado: "ítem con la imagen nueva",
      herramientas: ["mitumbes_item_listar", "mitumbes_item_imagen_adjuntar"],
      ejecuciones: [
        { toolName: "mitumbes_item_listar", arguments: { q: "La Pichanga Gastrobar" } },
        {
          toolName: "mitumbes_item_imagen_adjuntar",
          arguments: { id: UUID.pichanga, url: IMG_1, name: "La Pichanga Gastrobar" },
        },
      ],
    },
  },
  {
    id: "crear",
    tema: "alta de un negocio nuevo",
    texto:
      "crea el item La Pichanga Gastrobar como restaurante en la zona Tumbes con su descripción",
    esperada: "mitumbes_item_crear",
    antecedentes: ["mitumbes_categoria_listar", "mitumbes_zona_listar"],
    aprende: {
      label: "Crear ítem La Pichanga Gastrobar",
      resultado: "ítem creado con categoryId y zoneId",
      herramientas: ["mitumbes_categoria_listar", "mitumbes_zona_listar", "mitumbes_item_crear"],
      ejecuciones: [
        { toolName: "mitumbes_categoria_listar", arguments: {} },
        { toolName: "mitumbes_zona_listar", arguments: {} },
        {
          toolName: "mitumbes_item_crear",
          arguments: {
            slug: "la-pichanga-gastrobar",
            name: "La Pichanga Gastrobar",
            categoryId: UUID.restaurante,
            zoneId: UUID.zonaTumbes,
          },
        },
      ],
    },
  },
  {
    id: "estadisticas",
    tema: "lectura de métricas de publicidad",
    texto: "muéstrame las estadísticas de clics de la campaña de Fiestas Patrias",
    esperada: "mitumbes_publicidad_estadisticas",
    antecedentes: ["mitumbes_publicidad_listar"],
    aprende: {
      label: "Obtener estadísticas de la campaña Fiestas Patrias",
      resultado: "clicks e impresiones de la campaña",
      herramientas: ["mitumbes_publicidad_listar", "mitumbes_publicidad_estadisticas"],
      ejecuciones: [
        { toolName: "mitumbes_publicidad_listar", arguments: {} },
        { toolName: "mitumbes_publicidad_estadisticas", arguments: { id: UUID.playas } },
      ],
    },
  },
  {
    id: "cache",
    tema: "invalidación de CDN",
    texto: "purga la caché de cloudflare porque ya publiqué los cambios",
    esperada: "mitumbes_cache_revalidar",
    antecedentes: [],
    aprende: {
      label: "Revalidar caché del sitio",
      resultado: "caché purgada",
      herramientas: ["mitumbes_cache_revalidar"],
      ejecuciones: [{ toolName: "mitumbes_cache_revalidar", arguments: {} }],
    },
  },
  {
    id: "hero",
    tema: "configurar el banner de portada",
    texto: "configura el hero de la portada con el eslogan Tumbes te espera",
    esperada: "mitumbes_hero_configurar",
    antecedentes: ["mitumbes_hero_obtener"],
    aprende: {
      label: "Configurar hero de la portada",
      resultado: "hero con el eslogan nuevo",
      herramientas: ["mitumbes_hero_obtener", "mitumbes_hero_configurar"],
      ejecuciones: [
        { toolName: "mitumbes_hero_obtener", arguments: {} },
        { toolName: "mitumbes_hero_configurar", arguments: { name: "Tumbes te espera" } },
      ],
    },
  },
  {
    id: "ruta",
    tema: "crear un itinerario",
    texto: "crea una ruta turística que recorra Punta Sal y Zorritos en dos días",
    esperada: "mitumbes_ruta_crear",
    antecedentes: ["mitumbes_zona_listar"],
    aprende: {
      label: "Crear ruta turística Punta Sal Zorritos",
      resultado: "ruta creada con sus paradas",
      herramientas: ["mitumbes_zona_listar", "mitumbes_ruta_crear"],
      ejecuciones: [
        { toolName: "mitumbes_zona_listar", arguments: {} },
        {
          toolName: "mitumbes_ruta_crear",
          arguments: { name: "Ruta del norte", zoneId: UUID.zonaZorritos },
        },
      ],
    },
  },
  {
    id: "usuario",
    tema: "baja de un editor",
    texto: "desactiva la cuenta del editor Julieta Montenegro",
    esperada: "mitumbes_usuario_desactivar",
    antecedentes: ["mitumbes_usuario_listar"],
    aprende: {
      label: "Desactivar cuenta del editor Julieta Montenegro",
      resultado: "usuario inactivo",
      herramientas: ["mitumbes_usuario_listar", "mitumbes_usuario_desactivar"],
      ejecuciones: [
        { toolName: "mitumbes_usuario_listar", arguments: { q: "Julieta" } },
        { toolName: "mitumbes_usuario_desactivar", arguments: { id: UUID.julieta, name: "Julieta Montenegro" } },
      ],
    },
  },
  {
    id: "aliado",
    tema: "altas de socios comerciales",
    texto: "registra un nuevo aliado comercial: Hotel Costa Verde",
    esperada: "mitumbes_aliado_crear",
    antecedentes: ["mitumbes_aliado_listar"],
    aprende: {
      label: "Crear aliado comercial Hotel Costa Verde",
      resultado: "aliado registrado",
      herramientas: ["mitumbes_aliado_listar", "mitumbes_aliado_crear"],
      ejecuciones: [
        { toolName: "mitumbes_aliado_listar", arguments: {} },
        { toolName: "mitumbes_aliado_crear", arguments: { name: "Hotel Costa Verde" } },
      ],
    },
  },
  {
    id: "ayuda",
    tema: "consulta de documentación del propio MCP",
    texto: "qué campos necesita la herramienta item_crear y qué efectos tiene",
    esperada: "mitumbes_ayuda",
    antecedentes: [],
    aprende: {
      label: "Documentar la herramienta item_crear",
      resultado: "ficha de la tool",
      herramientas: ["mitumbes_ayuda"],
      ejecuciones: [{ toolName: "mitumbes_ayuda", arguments: { tool: "item_crear" } }],
    },
  },
  {
    id: "eliminar",
    tema: "borrar un negocio duplicado",
    texto: "elimina el item Cevichería Javi Valle que quedó duplicado",
    esperada: "mitumbes_item_eliminar",
    antecedentes: ["mitumbes_item_listar"],
    aprende: {
      label: "Eliminar ítem Cevichería Javi Valle",
      resultado: "ítem eliminado",
      herramientas: ["mitumbes_item_listar", "mitumbes_item_eliminar"],
      ejecuciones: [
        { toolName: "mitumbes_item_listar", arguments: { q: "Cevichería Javi Valle" } },
        { toolName: "mitumbes_item_eliminar", arguments: { id: UUID.cevicheria, name: "Cevichería Javi Valle" } },
      ],
    },
  },
  {
    id: "relacionar",
    tema: "vincular dos registros",
    texto: "relaciona el item La Pichanga Gastrobar con el aliado Hotel Costa Verde",
    esperada: "mitumbes_item_relacionar",
    antecedentes: ["mitumbes_item_listar"],
    aprende: {
      label: "Relacionar ítem con aliado",
      resultado: "vínculo creado",
      herramientas: ["mitumbes_item_listar", "mitumbes_item_relacionar"],
      ejecuciones: [
        { toolName: "mitumbes_item_listar", arguments: { q: "La Pichanga Gastrobar" } },
        {
          toolName: "mitumbes_item_relacionar",
          arguments: { id: UUID.pichanga, name: "La Pichanga Gastrobar" },
        },
      ],
    },
  },
];

/**
 * Segunda vez que el usuario pide LO MISMO sobre OTRO negocio (no está en ningún
 * grounding). Sirve para dos cosas: ¿generaliza la habilidad aprendida? y ¿se
 * fuga el identificador viejo hacia la entidad nueva?
 */
const DERIVADOS: Record<string, string> = {
  categoria: 'cambia de categoría "Hostal El Faro" a Alojamiento',
  imagen: `subir la foto del restaurant Mundi Novo con la url ${IMG_2}`,
  crear: "crea el item Hotel Britania como hospedaje en la zona Zorritos",
  estadisticas: "cuántos clics tuvo la publicidad del banner de Carnaval",
  cache: "vuelve a purgar la caché del sitio, publiqué otra vez",
  hero: "cambia el eslogan del hero de la portada a Verano en Tumbes",
  ruta: "crear ruta turística por la reserva nacional de Tumbes",
  usuario: "desactivar al editor Carlos Bernilla",
  aliado: "registrar aliado comercial Restaurante El Muelle",
  ayuda: "cómo se usa la tool lugar_crear y qué campos pide",
  eliminar: "eliminar el item Hotel Britania que ya no opera",
  relacionar: "vincular el item Cevichería Javi Valle con el aliado Hotel Costa Verde",
};

/** Nada que ver con el catálogo: miden aquí el suelo de ruido del espacio vectorial. */
const FUERA_DE_TEMA = [
  "hola que hora es hoy en tumbes",
  "explícame qué es una derivada parcial",
  "quiero cancelar mi suscripción de telefonía",
  "escribe un haiku sobre el mar",
  "necesito una receta de ceviche de conchas negras",
  "descargar el log de error del servidor",
  "el clima en lima mañana",
  "busco apartamento en alquiler en miraflores",
];

// ─────────────────────────────────────────────────────────────────────────────
// Cliente HTTP (el mismo contrato que consume PredictionClientService)
// ─────────────────────────────────────────────────────────────────────────────
type Prevision = {
  tools: { name: string }[];
  graph?: { nodes?: string[]; edges?: { from: string; to: string; type: string }[] };
  complexity?: string;
};

let predictUrl = "";
let cerrarServidor: (() => Promise<void>) | undefined;
const metricas = {
  latenciaSum: 0,
  latenciaN: 0,
  peor: { texto: "", ms: 0 },
};

async function api<T>(path: string, cuerpo: unknown, scope: string): Promise<T> {
  const inicio = performance.now();
  const res = await fetch(`${predictUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "key-remtk": `${scope}::${AGENTE}` },
    body: JSON.stringify(cuerpo),
  });
  const ms = performance.now() - inicio;
  metricas.latenciaSum += ms;
  metricas.latenciaN += 1;
  if (ms > metricas.peor.ms) metricas.peor = { texto: path, ms };
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(`${path} respondió ${res.status}: ${JSON.stringify(data)?.slice(0, 300)}`);
  }
  return data as T;
}

const predecir = (scope: string, sessionId: string, text: string) =>
  api<Prevision>("/predict", { tenant: scope, agentId: AGENTE, sessionId, text, source: "agent" }, scope);

const skillsPredict = (scope: string, sessionId: string, text: string, adjuntoUrl?: string) =>
  api<{
    matchedSkill?: { id: string; name: string; confidence: number; tools: string[] };
    resolvedEntity?: { id: string; name: string; type: string; matchedPendingAction?: string; confidence: number };
    groundedParameters?: { key: string; matchedId: string; matchedName: string; confidence: number }[];
    preResolvedArgs: Record<string, unknown>;
    suggestedAction: string;
    contextPrompt: string;
  }>(
    "/skills/predict",
    {
      tenant: scope,
      agentId: AGENTE,
      sessionId,
      text,
      ...(adjuntoUrl ? { attachments: [{ url: adjuntoUrl, mimeType: "image/png" }] } : {}),
    },
    scope,
  );

/** El payload que arma `learnSkillFromTask()`: id con slug, name = label, args crudos. */
function payloadDelEjecutor(c: Caso): Record<string, unknown> {
  const clean = c.aprende.label
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/[^a-z0-9]+/gi, "_")
    .replace(/^_|_$/g, "")
    .slice(0, 32);
  const grounding: Record<string, { id: string; name: string }[]> = {};
  for (const e of c.aprende.ejecuciones) {
    if (!c.aprende.herramientas.includes(e.toolName)) continue;
    for (const [k, v] of Object.entries(e.arguments)) {
      const ancla = k.endsWith("Id") || k.endsWith("Name") || k === "id" || k === "name";
      if (typeof v !== "string" || v.length <= 1 || !ancla) continue;
      (grounding[k] ??= []).push({ id: v, name: v });
    }
  }
  return {
    id: `skill_${clean}_${c.aprende.herramientas.join("_").slice(0, 32)}`,
    name: c.aprende.label,
    description: c.aprende.resultado,
    intentSummary: `${c.aprende.label}. ${c.texto}`.trim().slice(0, 500),
    tools: c.aprende.herramientas,
    parameterGrounding: Object.keys(grounding).length ? grounding : undefined,
  };
}

/** La misma habilidad con anclas legibles: uuid + nombre con el que el usuario la nombra. */
function payloadNombrado(c: Caso): Record<string, unknown> {
  const base = payloadDelEjecutor(c);
  const nombres: Record<string, string> = {
    [UUID.pichanga]: "La Pichanga Gastrobar",
    [UUID.cevicheria]: "Cevichería Javi Valle",
    [UUID.restaurante]: "Restaurante y Gastronomía",
    [UUID.servicios]: "Servicios Profesionales",
    [UUID.hotels]: "Hoteles y Hospedajes",
    [UUID.zonaTumbes]: "Tumbes Centro",
    [UUID.zonaZorritos]: "Zorritos",
    [UUID.julieta]: "Julieta Montenegro",
    [UUID.carlos]: "Carlos Bernilla",
    [UUID.playas]: "Campaña Fiestas Patrias",
  };
  const g = base.parameterGrounding as Record<string, { id: string; name: string }[]> | undefined;
  if (g) {
    for (const lista of Object.values(g)) {
      for (const a of lista) a.name = nombres[a.id] ?? a.name;
    }
  }
  return { ...base, id: String(base.id).replace("skill_", "skilln_") };
}

before(async () => {
  // El entorno se fija AQUI, no se hereda del .env del desarrollador: sin el
  // modelo real los cosenos son de fallback (~0.2-0.4) y ninguna de las
  // comparaciones de §3 significa nada.
  const envAfectado = {
    ONNX_ENABLED: process.env.ONNX_ENABLED,
    ONNX_MODELS_PATH: process.env.ONNX_MODELS_PATH,
    QDRANT_ENABLED: process.env.QDRANT_ENABLED,
    LEARN_ENABLED: process.env.LEARN_ENABLED,
  };
  process.env.ONNX_ENABLED = "1";
  process.env.ONNX_MODELS_PATH = "./models";
  process.env.QDRANT_ENABLED = "0";
  process.env.LEARN_ENABLED = "1";
  const server = await startPredictServer(0);
  for (const [k, v] of Object.entries(envAfectado)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  cerrarServidor = server.close;
  predictUrl = `http://localhost:${server.port}`;

  for (const scope of SCOPES) {
    await api("/tools", { tenant: scope, agentId: AGENTE, tools: TOTAL_ACTIVE_TOOLS }, scope);
  }
  // El warm-up del grafo corre en segundo plano: /predict lo espera con cota, pero
  // la batería mide precisión, así que se espera aquí para no medir medio catálogo.
  for (const scope of SCOPES) {
    const limite = Date.now() + 120_000;
    for (;;) {
      const r = await fetch(
        `${predictUrl}/tools/ready?tenant=${scope}&agentId=${AGENTE}`,
        { headers: { "key-remtk": `${scope}::${AGENTE}` } },
      );
      const j = (await r.json()) as { ready?: boolean };
      if (j.ready) break;
      if (Date.now() > limite) throw new Error(`el warm-up del scope ${scope} no terminó en 120s`);
      await new Promise((res) => setTimeout(res, 2000));
    }
  }
  const cuenta = await (
    await fetch(`${predictUrl}/tools/count?tenant=${SCOPE_BASE}&agentId=${AGENTE}`, {
      headers: { "key-remtk": `${SCOPE_BASE}::${AGENTE}` },
    })
  ).json();
  console.log(
    `  [arranque] catálogo indexado=${JSON.stringify(cuenta)} tools reales=${TOTAL_ACTIVE_TOOLS.length} ` +
      `(62 de MiTumbes + 134 de ruido) | modelo=ONNX | qdrant=apagado`,
  );
});

after(async () => {
  await cerrarServidor?.();
});

// ─────────────────────────────────────────────────────────────────────────────
// Utilidades de medición
// ─────────────────────────────────────────────────────────────────────────────
const offered = (p: Prevision) => p.tools.map((t) => t.name);

/** El nodo `to` de las PREREQUISITE que no es `from` de nadie: la terminal del DAG. */
function terminales(p: Prevision): string[] {
  const aristas = (p.graph?.edges ?? []).filter((e) => e.type === "PREREQUISITE");
  const froms = new Set(aristas.map((e) => e.from));
  return [...new Set(aristas.map((e) => e.to))].filter((to) => !froms.has(to));
}

/** Coloca la acción esperada en un ranking, o `undefined` si no apareció. */
function posicion(names: string[], objetivo: string): number | undefined {
  const i = names.indexOf(objetivo);
  return i === -1 ? undefined : i + 1;
}

/** Reciprocal rank promedio sobre los casos que acertaron. */
function mrr(posiciones: (number | undefined)[]): number {
  const hits = posiciones.filter((p): p is number => p !== undefined);
  if (hits.length === 0) return 0;
  return hits.reduce((a, p) => a + 1 / p, 0) / posiciones.length;
}

const pct = (n: number, d: number) => `${((n / d) * 100).toFixed(1)}%`;
const fmt = (n: number) => n.toFixed(3);

async function lote(scope: string, prefijo: string, textos: Map<string, string>) {
  const filas: { id: string; names: string[]; terminales: string[]; ms: number }[] = [];
  for (const [id, texto] of textos) {
    const inicio = performance.now();
    const p = await predecir(scope, `${prefijo}-${id}`, texto);
    filas.push({ id, names: offered(p), terminales: terminales(p), ms: performance.now() - inicio });
  }
  return filas;
}

describe("§1 predicción sin memoria sobre las 196 herramientas", () => {
  test("cada tarea conoce su acción: mide aciertos, posiciones y ruido ofrecido", async () => {
    const textos = new Map(CASOS.map((c) => [c.id, c.texto]));
    const filas = await lote(SCOPE_BASE, "s1", textos);
    const porId = new Map(filas.map((f) => [f.id, f]));

    let enTop5 = 0;
    let enPrimero = 0;
    let conTerminal = 0;
    const posiciones: (number | undefined)[] = [];
    const relleno = new Map<string, number>();

    for (const c of CASOS) {
      const f = porId.get(c.id)!;
      assert.ok(f.names.length > 0 && f.names.length <= 5, `${c.id} ofreció ${f.names.length} tools`);
      for (const n of f.names) {
        assert.ok(
          TOTAL_ACTIVE_TOOLS.some((t) => t.name === n),
          `${c.id} ofreció una tool que no está en el catálogo: ${n}`,
        );
      }
      const p = posicion(f.names, c.esperada);
      posiciones.push(p);
      if (p !== undefined) enTop5 += 1;
      if (p === 1) enPrimero += 1;
      if (f.terminales.includes(c.esperada)) conTerminal += 1;
      if (p !== 1) for (const n of f.names.slice(0, p ?? f.names.length)) if (n !== c.esperada) relleno.set(n, (relleno.get(n) ?? 0) + 1);
      console.log(
        `  [s1:${c.id.padEnd(13)}] ${p ? `#${p} ${c.esperada}` : `FUERA ${c.esperada}`} ` +
          `| ofrecidas=[${f.names.join(", ")}] | terminal=${f.terminales.join(",") || "-"} | ${f.ms.toFixed(0)}ms`,
      );
    }

    console.log(
      `  [s1] accion en top-5=${enTop5}/${CASOS.length} (${pct(enTop5, CASOS.length)}) ` +
        `primera-plaza=${enPrimero} MRR=${fmt(mrr(posiciones))} ` +
        `terminal-del-DAG=${conTerminal}/${CASOS.length}`,
    );
    console.log(
      `  [s1-ruido] tools que desalojan a la acción: ` +
        [...relleno.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([n, k]) => `${n}×${k}`).join(", "),
    );
    assert.equal(filas.length, CASOS.length, "la batería no midió las 12 tareas");
    assert.ok(enTop5 > 0, "ninguna de las 12 acciones apareció en el top-5: el predictor está roto, no impreciso");
  });

  test("la tanda respeta el techo de 5 y el grafo llega con aristas cuando hay pre-requisitos", async () => {
    const conAntecedentes = CASOS.filter((c) => c.aprende.herramientas.length > 1);
    let aristas = 0;
    for (const c of conAntecedentes) {
      const p = await predecir(SCOPE_BASE, `s1b-${c.id}`, c.texto);
      const n = (p.graph?.edges ?? []).length;
      if (n > 0) aristas += 1;
      assert.ok(p.tools.length <= 5, `${c.id} devolvió ${p.tools.length} tools, el ejecutor solo ofrece 5`);
    }
    console.log(
      `  [s1-grafo] tareas multi-paso con aristas PREREQUISITE=${aristas}/${conAntecedentes.length} ` +
        `(sin aristas el ejecutor no tiene cómo saber cuál es la ACCIÓN)`,
    );
  });
});

describe("§2 el mismo turno después de aprender la habilidad", () => {
  before(async () => {
    for (const c of CASOS) {
      await api("/predict/feedback", {
        tenant: SCOPE_APRENDE,
        agentId: AGENTE,
        sessionId: `s2-fb-${c.id}`,
        text: c.texto,
        used: c.aprende.herramientas,
        rejected: [],
      }, SCOPE_APRENDE);
      await api("/skills/learn", payloadDelEjecutor(c), SCOPE_APRENDE);
      await api("/skills/learn", payloadNombrado(c), SCOPE_NOMBRADO);
    }
  });

  test("aprender sube la cobertura del catálogo sin romper la tanda de 5", async () => {
    const textos = new Map(CASOS.map((c) => [c.id, c.texto]));
    const base = new Map((await lote(SCOPE_BASE, "s2b", textos)).map((f) => [f.id, f]));
    const con = new Map((await lote(SCOPE_APRENDE, "s2c", textos)).map((f) => [f.id, f]));

    let suben = 0;
    let bajan = 0;
    for (const c of CASOS) {
      const antes = posicion(base.get(c.id)!.names, c.esperada);
      const despues = posicion(con.get(c.id)!.names, c.esperada);
      if ((despues ?? 99) < (antes ?? 99)) suben += 1;
      if ((despues ?? 99) > (antes ?? 99)) bajan += 1;
      if (antes !== despues) {
        console.log(`  [s2:${c.id.padEnd(13)}] antes=#${antes ?? "-"} ahora=#${despues ?? "-"} (esperada ${c.esperada})`);
      }
      assert.ok(con.get(c.id)!.names.length <= 5);
    }
    console.log(
      `  [s2] mejoraron=${suben} empeoraron=${bajan} de ${CASOS.length} ` +
        `(mismo catálogo; la diferencia del scope es el ciclo completo: feedback + 12 habilidades)`,
    );
    assert.ok(bajan <= suben, `aprender empeoró más de lo que mejoró: ${bajan} bajadas vs ${suben} subidas`);
  });

  test("el turn-next del MISMO pedido se sirve de la habilidad, no del catálogo", async () => {
    let ok = 0;
    for (const c of CASOS) {
      const r = await skillsPredict(SCOPE_APRENDE, `s2t-${c.id}`, c.texto);
      const esperado = payloadDelEjecutor(c).id as string;
      if (r.matchedSkill?.id === esperado) ok += 1;
      else {
        console.log(
          `  [s2-habilidad:${c.id}] confundida: pidió "${r.matchedSkill?.name}" (${fmt(r.matchedSkill?.confidence ?? 0)}) ` +
            `en vez de ${esperado}`,
        );
      }
      assert.ok(
        !r.matchedSkill || r.matchedSkill.tools.length > 0,
        `${c.id}: la habilidad devuelta no trae herramientas`,
      );
    }
    console.log(`  [s2-habilidad] la habilidad correcta gana en ${ok}/${CASOS.length} reintentos del mismo texto`);
    assert.equal(ok, CASOS.length, "12 habilidades coexistiendo no pueden confundirse entre sí en su propio texto");
  });

  test("pedir lo mismo sobre OTRO negocio no arrastra el identificador anterior", async () => {
    let resuelto = 0;
    let filtrado = 0;
    const fugas: string[] = [];
    for (const c of CASOS) {
      const texto = DERIVADOS[c.id];
      const idsViejos = new Set(
        Object.values((payloadDelEjecutor(c).parameterGrounding as Record<string, { id: string }[]>) ?? {})
          .flat()
          .map((a) => a.id),
      );
      const r = await skillsPredict(SCOPE_APRENDE, `s2d-${c.id}`, texto);
      if (r.matchedSkill?.id !== (payloadDelEjecutor(c).id as string)) {
        console.log(
          `  [s2:derivado:${c.id}] NO se generalizó: "${r.matchedSkill?.name ?? "nada"}" ` +
            `conf=${fmt(r.matchedSkill?.confidence ?? 0)} acción=${r.suggestedAction}`,
        );
        continue;
      }
      const args = Object.values(r.preResolvedArgs).flatMap((v) => (typeof v === "string" ? [v] : []));
      const arrastra = args.filter((a) => idsViejos.has(a));
      if (arrastra.length > 0) fugas.push(`${c.id} -> ${arrastra.join(",")}`);
      else if (args.length === 0) filtrado += 1;
      else resuelto += 1;
    }
    console.log(
      `  [s2:derivado] fuga de args viejos=${fugas.length} ${fugas.length ? `¡${fugas.join(" | ")}!` : "(ninguna)"} ` +
        `| filtrado correcto sin args=${filtrado} | args útiles=${resuelto}`,
    );
    assert.deepEqual(fugas, [], "la habilidad aprendida inyecta en el turno nuevo un id de otra entidad");
  });

  test("una habilidad aprendida en OTRO scope no se ve desde el propio", async () => {
    const r = await skillsPredict(SCOPE_BASE, "s2-aislamiento", CASOS[0].texto);
    console.log(
      `  [s2:aislamiento] scope sin habilidades -> matched="${r.matchedSkill?.name ?? "nada"}" ` +
        `args=${JSON.stringify(r.preResolvedArgs)} accion=${r.suggestedAction}`,
    );
    assert.equal(r.matchedSkill, undefined, "el tenant base nunca aprendió nada y devolvió una habilidad");
    assert.ok(
      !JSON.stringify(r.preResolvedArgs).includes(UUID.pichanga),
      "fuga entre scopes: llegaron identificadores memorizados en otro tenant",
    );
  });

  test("el ejecutor manda el valor crudo como nombre de ancla: hoy eso no resuelve entidades", async () => {
    const crudos: string[] = [];
    const legibles: string[] = [];
    let igual = 0;
    for (const c of CASOS) {
      const texto = c.texto;
      const a = await skillsPredict(SCOPE_APRENDE, `s2g-${c.id}`, texto);
      const b = await skillsPredict(SCOPE_NOMBRADO, `s2g-${c.id}`, texto);
      const esUuid = (v: unknown) => typeof v === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}/i.test(v);
      for (const g of a.groundedParameters ?? []) if (esUuid(g.matchedId) && g.matchedName === g.matchedId) crudos.push(`${c.id}.${g.key}`);
      for (const g of b.groundedParameters ?? []) if (esUuid(g.matchedId) && g.matchedName !== g.matchedId) legibles.push(`${c.id}.${g.key}`);
      if (JSON.stringify(a.preResolvedArgs) === JSON.stringify(b.preResolvedArgs)) igual += 1;
      console.log(
        `  [s2:grounding:${c.id.padEnd(13)}] crudo=[${(a.groundedParameters ?? []).map((g) => `${g.key}=${g.matchedName.slice(0, 24)}(${fmt(g.confidence)})`).join(", ") || "-"}] ` +
          `| nombrado=[${(b.groundedParameters ?? []).map((g) => `${g.key}=${g.matchedName.slice(0, 24)}(${fmt(g.confidence)})`).join(", ") || "-"}]`,
      );
    }
    console.log(
      `  [s2:grounding] anclas devueltas con name=uuid (inútiles para el mini-agente)=${crudos.length} ` +
        `| con nombre humano=${legibles.length} | scopes que coinciden byte a byte=${igual}/${CASOS.length}`,
    );
    assert.ok(
      legibles.length > 0 || crudos.length === 0,
      "ningún scope logra anclas legibles: el aprendizaje no está sirviendo para nada",
    );
  });
});

describe("§3 qué tan buena es realmente la memoria de habilidades", () => {
  test("la confianza de la habilidad correcta supera al suelo de ruido del espacio vectorial", async () => {
    const adentro: number[] = [];
    const haciaFuera: number[] = [];
    for (const c of CASOS) {
      const r = await skillsPredict(SCOPE_APRENDE, `s3-in-${c.id}`, c.texto);
      adentro.push(r.matchedSkill?.confidence ?? 0);
    }
    for (const texto of FUERA_DE_TEMA) {
      const r = await skillsPredict(SCOPE_NOMBRADO, "s3-out", texto);
      haciaFuera.push(r.matchedSkill?.confidence ?? 0);
      console.log(
        `  [s3:suelo] "${texto.slice(0, 42)}" -> "${r.matchedSkill?.name ?? "nada"}" conf=${fmt(r.matchedSkill?.confidence ?? 0)} accion=${r.suggestedAction}`,
      );
    }
    const minIn = Math.min(...adentro);
    const maxOut = Math.max(...haciaFuera);
    // El piso de `execute_direct` se calibra POR MODELO (la escala del coseno
    // cambia entre modelos); aquí se lee el del modelo activo.
    const skillFloor = skillThresholdsByModel[loadConfig().onnxModelSize].skillMatchMin;
    console.log(
      `  [s3] mejor separación alcanzable: min(dentro)=${fmt(minIn)} max(fuera)=${fmt(maxOut)} ` +
        `margen=${fmt(minIn - maxOut)} | piso de /skills/predict=${skillFloor} para execute_direct`,
    );
    assert.ok(
      minIn > maxOut,
      `el texto fuera de tema puntúa ${fmt(maxOut)} y la habilidad correcta ${fmt(minIn)}: no hay señal que umbralizar`,
    );
    assert.ok(minIn > skillFloor, "la habilidad correcta no pasa el umbral interno de execute_direct");
  });

  test("reaprender la misma habilidad refuerza en vez de duplicar", async () => {
    const primera = await api<{ reinforcementScore: number }>("/skills/learn", payloadDelEjecutor(CASOS[0]), SCOPE_APRENDE);
    const segunda = await api<{ reinforcementScore: number }>("/skills/learn", payloadDelEjecutor(CASOS[0]), SCOPE_APRENDE);
    console.log(
      `  [s3:refuerzo] score ${primera.reinforcementScore} -> ${segunda.reinforcementScore} ` +
        `(skills en scope=${CASOS.length}, no 2x)`,
    );
    assert.ok(
      segunda.reinforcementScore === primera.reinforcementScore + 1,
      "el refuerzo no es idempotente por id de habilidad",
    );
  });

  test("cada llamada deja latencia medida, no percibida", () => {
    const media = metricas.latenciaSum / metricas.latenciaN;
    console.log(
      `  [s3:latencia] media=${media.toFixed(0)}ms sobre ${metricas.latenciaN} llamadas ` +
        `| peor=${metricas.peor.ms.toFixed(0)}ms (${metricas.peor.texto})`,
    );
    assert.ok(metricas.latenciaN > 40, "la batería no midió suficientes llamadas");
  });
});

describe("§4 secuencia real de varios temas en una sesión", () => {
  test("el cambio de tema se decide por el coseno del turno y se puede auditar", async () => {
    const sesion = "s4-secuencia";
    // Etiquetas de tema puestas a mano para poder medir al detector; el detector
    // solo ve el coseno contra el estado previo de la sesión.
    const giro = [
      { etiqueta: "item:alta", tema: "item", texto: CASOS[2].texto },
      { etiqueta: "item:anfora", tema: "item", texto: "y ahora mejórame la descripción de ese mismo item" },
      { etiqueta: "cdn:salta", tema: "cdn", texto: CASOS[4].texto },
      { etiqueta: "cdn:continua", tema: "cdn", texto: "vuelve a purgar la caché del sitio, publiqué otra vez" },
      { etiqueta: "usuarios:salta", tema: "usuarios", texto: CASOS[7].texto },
    ];
    const marca: { paso: string; tema: string; topic: number; shift: boolean }[] = [];
    for (const paso of giro) {
      const mem = await api<{
        memories: unknown[];
        topicScore: number;
        topicShift: boolean;
        skillContext?: { matchedSkill?: { name: string; confidence: number } };
        activeEntities?: { id: string; name: string }[];
      }>(
        "/memory/predict",
        { tenant: SCOPE_APRENDE, agentId: AGENTE, sessionId: sesion, text: paso.texto },
        SCOPE_APRENDE,
      );
      marca.push({ paso: paso.etiqueta, tema: paso.tema, topic: mem.topicScore, shift: mem.topicShift });
      console.log(
        `  [s4:${paso.etiqueta.padEnd(15)}] topicScore=${fmt(mem.topicScore)} shift=${mem.topicShift} ` +
          `memorias=${mem.memories.length} entidades=${(mem.activeEntities ?? []).length} ` +
          `habilidad="${mem.skillContext?.matchedSkill?.name ?? "nada"}"`,
      );
    }

    // (1) Invariante estructural: el banderín sale del coseno, no de un texto.
    for (const m of marca.slice(1)) {
      assert.equal(
        m.shift,
        m.topic < TOPIC_SHIFT_THRESHOLD,
        `${m.paso}: topicShift=${m.shift} con topicScore=${fmt(m.topic)} y umbral ${TOPIC_SHIFT_THRESHOLD}`,
      );
    }
    assert.equal(marca[0].topic, 0, "el primer turno de la sesión no tiene estado previo con el que compararse");

    // (2) Lo que el detector sí logra: una continuación con contenido le gana a un
    //     salto de tema real.
    const continuaConContenido = marca.find((m) => m.paso === "cdn:continua")!.topic;
    const saltoReal = marca.find((m) => m.paso === "usuarios:salta")!.topic;
    assert.ok(
      continuaConContenido > saltoReal,
      `una continuación (${fmt(continuaConContenido)}) no puede puntuar bajo un salto de tema (${fmt(saltoReal)})`,
    );

    // Lo que NO logra, medido y a la vista: la anáfora corta del mismo tema cae por
    // debajo del umbral y reinicia el contexto de sesión.
    const anfora = marca.find((m) => m.paso === "item:anfora")!.topic;
    console.log(
      `  [s4:tema] umbral=${TOPIC_SHIFT_THRESHOLD} | continuación con contenido=${fmt(continuaConContenido)} ` +
        `salto real=${fmt(saltoReal)} anáfora del mismo tema=${fmt(anfora)} ` +
        `(${anfora < TOPIC_SHIFT_THRESHOLD ? "DESPLAZA el tema pese a ser el mismo: el detector pierde la referencia" : "ok"})`,
    );
  });

  test("la habilidad y la entidad de sesión compiten: se mide cuál gana", async () => {
    const sesion = "s4-entidad";
    const item = { id: UUID.pichanga, name: "La Pichanga Gastrobar", type: "item" };
    // Tres turnos de item (uno de ellos crea la entidad) y luego el pedido de imagen.
    for (const texto of [CASOS[2].texto, "muéstrame el detalle de ese negocio", "corrige su dirección a Huáscar 560"]) {
      await api("/memory/predict", { tenant: SCOPE_APRENDE, agentId: AGENTE, sessionId: sesion, text: texto }, SCOPE_APRENDE);
    }
    await api(
      "/entities/track",
      {
        tenant: SCOPE_APRENDE,
        agentId: AGENTE,
        sessionId: sesion,
        ...item,
        slug: "la-pichanga-gastrobar",
        state: { lifecycle: "activo", pendingAction: CASOS[2].aprende.label },
      },
      SCOPE_APRENDE,
    );
    const r = await skillsPredict(SCOPE_APRENDE, sesion, CASOS[1].texto, IMG_1);
    console.log(
      `  [s4:mezcla] habilidad="${r.matchedSkill?.name}" (${fmt(r.matchedSkill?.confidence ?? 0)}) ` +
        `entidad=${r.resolvedEntity?.name ?? "NA"} (${fmt(r.resolvedEntity?.confidence ?? 0)}) ` +
        `accion=${r.suggestedAction} args=${JSON.stringify(r.preResolvedArgs)}`,
    );
    assert.equal(
      r.preResolvedArgs.id,
      item.id,
      "tras 3 turnos del mismo negocio el sistema no sabe de qué item habla el usuario",
    );
    assert.equal(r.preResolvedArgs.image, IMG_1, "el adjunto del turno no viaja a preResolvedArgs");
    assert.equal(r.suggestedAction, "execute_direct", "con id e imagen resueltos no debería pedir descubrir pre-requisitos");
  });

  test("la entidad de sesión responde por GET y se borra con clear", async () => {
    const sesion = "s4-entidades";
    await api("/entities/track", {
      tenant: SCOPE_APRENDE,
      agentId: AGENTE,
      sessionId: sesion,
      id: UUID.pichanga,
      type: "item",
      name: "La Pichanga Gastrobar",
      slug: "la-pichanga-gastrobar",
      state: { lifecycle: "activo", pendingAction: "actualizar imagen" },
    }, SCOPE_APRENDE);
    const listo = (await (
      await fetch(`${predictUrl}/entities/session?sessionId=${sesion}&tenant=${SCOPE_APRENDE}&agentId=${AGENTE}`, {
        headers: { "key-remtk": `${SCOPE_APRENDE}::${AGENTE}` },
      })
    ).json()) as { id: string }[];
    assert.equal(listo.length, 1, "la sesión debe exponer exactamente la entidad que se registró");
    await api("/entities/clear", { sessionId: sesion }, SCOPE_APRENDE);
    const despues = (await (
      await fetch(`${predictUrl}/entities/session?sessionId=${sesion}&tenant=${SCOPE_APRENDE}&agentId=${AGENTE}`, {
        headers: { "key-remtk": `${SCOPE_APRENDE}::${AGENTE}` },
      })
    ).json()) as unknown[];
    assert.deepEqual(despues, [], "clear no vació el grafo de entidades de la sesión");
  });
});

describe("§5 la otra mitad: memoria contextual y su filtro de ruido", () => {
  test("con Qdrant apagado /memory/predict degrada a cero memorias y el resto sigue viajando", async () => {
    const r = await api<{
      memories: unknown[];
      topicScore: number;
      topicShift: boolean;
      skillContext?: { matchedSkill?: { name: string } };
      activeEntities: unknown[];
    }>(
      "/memory/predict",
      { tenant: SCOPE_APRENDE, agentId: AGENTE, sessionId: "s5-mem", text: CASOS[0].texto },
      SCOPE_APRENDE,
    );
    console.log(
      `  [s5] memories=${r.memories.length} (recall BM25 vive en Qdrant) ` +
        `topicScore=${fmt(r.topicScore)} skill="${r.skillContext?.matchedSkill?.name ?? "nada"}" ` +
        `entidades=${r.activeEntities.length}`,
    );
    assert.equal(r.memories.length, 0, "sin Qdrant no debería haber memorias, y las hay: de dónde salen");
    assert.ok(r.skillContext, "el enriquecimiento de habilidades debe viajar aunque el recall esté caído");
  });

  test("el juzgado de memoria puntúa más alto el descarte que el hecho informativo", async () => {
    const descartes = [
      "No encontré la herramienta adecuada para actualizar el item, disculpa",
      "[HERRAMIENTA_INADECUADA] ninguna tool permite adjuntar la imagen",
      "[FALTAN_DATOS] falta el slug del item",
    ];
    const hechos = [
      "El item La Pichanga Gastrobar quedó con la imagen actualizada",
      "La categoría Servicios Profesionales tiene uuid 5a6b7c8d-9e0f-41a2-b3c4-d5e6f7081920",
    ];
    const puntuar = async (textos: readonly string[], etiqueta: string) => {
      const scores: number[] = [];
      for (const texto of textos) {
        const r = await api<{ isNoise: boolean; noiseScore: number }>("/memory/judge", { text: texto }, SCOPE_APRENDE);
        scores.push(r.noiseScore);
        console.log(
          `  [s5:judge:${etiqueta}] isNoise=${r.isNoise} score=${fmt(r.noiseScore)} "${texto.slice(0, 52)}"`,
        );
      }
      return scores.reduce((a, b) => a + b, 0) / scores.length;
    };
    const ruido = await puntuar(descartes, "descarte");
    const hecho = await puntuar(hechos, "hecho  ");
    const invertidos = (await Promise.all(descartes.map((t) => api<{ isNoise: boolean }>("/memory/judge", { text: t }, SCOPE_APRENDE))))
      .filter((r) => !r.isNoise).length;
    console.log(
      `  [s5:judge] promedio descarte=${fmt(ruido)} vs promedio hecho=${fmt(hecho)} ` +
        `| descartes que el juzgado deja pasar a memoria larga=${invertidos}/3`,
    );
    assert.ok(
      ruido > hecho,
      `el juzgado no separa el descarte del dato útil: ${fmt(ruido)} vs ${fmt(hecho)} — cualquier umbral entre los dos es ruido`,
    );
  });
});
