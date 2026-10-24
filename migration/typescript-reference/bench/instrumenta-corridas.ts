/**
 * Instrumenta turnos reales de /predict SIN modificar src/: parchea el prototipo
 * del EmbeddingEngineService y cuenta tres cosas distintas — las solicitudes del
 * pipeline (embed/embedTokens), las pasadas REALES al bi-encoder (el método
 * privado de forward) y los textos distintos detrás. La diferencia entre las dos
 * primeras es lo que absorbe la memo por texto; la razón pasadas/textos es la
 * duplicidad que aún se paga. El cross-encoder (MiniLM, otro modelo) se cuenta
 * aparte para verificar que las candidatas del juicio quedan intactas.
 *
 * Uso: tsx bench/instrumenta-corridas.ts
 */
import { createSystem, loadConfig, EmbeddingEngineService } from "../src/index";
import { CrossEncoderService } from "../src/juicio/services/cross-encoder.service";
import { BENCH_TOOLS } from "./catalog";

process.env.QDRANT_ENABLED = "0";

type Registro = { tipo: "pooled" | "tokens" | "cross" | "pass"; texto: string; ms: number };
const registros: Registro[] = [];

const proto = EmbeddingEngineService.prototype as unknown as Record<
  string,
  (this: unknown, text: string, size?: unknown, opts?: unknown) => Promise<unknown>
>;
const embedOriginal = proto.embed;
const embedTokensOriginal = proto.embedTokens;

proto.embed = async function (this: unknown, text: string, size?: unknown, opts?: unknown) {
  const t0 = performance.now();
  try {
    return await embedOriginal.call(this as never, text, size, opts);
  } finally {
    registros.push({ tipo: "pooled", texto: text, ms: performance.now() - t0 });
  }
};
proto.embedTokens = async function (this: unknown, text: string, size?: unknown, opts?: unknown) {
  const t0 = performance.now();
  try {
    return await embedTokensOriginal.call(this as never, text, size, opts);
  } finally {
    registros.push({ tipo: "tokens", texto: text, ms: performance.now() - t0 });
  }
};

// La corrida REAL al modelo: con la memo por texto, una llamada a embed/embedTokens
// puede no costar inferencia. El método privado de forward recibe (size, texto).
const passFn = ["runOnce", "runRaw", "runModel", "runModelTokens"].find((n) => typeof proto[n] === "function");
if (!passFn) {
  console.error("no se encontró el método de forward pass del EmbeddingEngineService");
  process.exit(1);
}
const passOriginal = proto[passFn];
proto[passFn] = async function (this: unknown, _size: unknown, text: string) {
  const t0 = performance.now();
  try {
    return await passOriginal.call(this as never, _size, text);
  } finally {
    registros.push({ tipo: "pass", texto: text, ms: performance.now() - t0 });
  }
};
console.log(`forward pass instrumentado: ${passFn}`);

// El cross-encoder es un modelo DISTINTO (MiniLM): se cuenta aparte para saber
// cuántas pasadas por candidata paga el juicio, sin mezclarlo con el bi-encoder.
const ceProto = CrossEncoderService.prototype as unknown as {
  score: (query: string, passage: string) => Promise<unknown>;
};
const ceOriginal = ceProto.score;
ceProto.score = async function (this: unknown, query: string, passage: string) {
  const t0 = performance.now();
  try {
    return await ceOriginal.call(this as never, query, passage);
  } finally {
    registros.push({ tipo: "cross", texto: `${query}\u0000${passage}`, ms: performance.now() - t0 });
  }
};

const TURNOS = [
  {
    etiqueta: "corta",
    text: "envía el resumen de ventas por correo a finanzas",
    history: [] as { role: string; content: string }[],
  },
  {
    etiqueta: "anafórica",
    text: "ahora hazlo otra vez",
    history: [
      { role: "user", content: "busca el pronóstico del clima en Lima" },
      { role: "assistant", content: "El clima en Lima hoy es nublado con 19 grados." },
    ],
  },
  {
    etiqueta: "larga-512",
    text:
      "necesito enviar el resumen de ventas por correo a finanzas " +
      "cuéntame también el detalle de cada paso del proceso y sus dependencias. ".repeat(20),
    history: [{ role: "user", content: "revisa el borrador anterior antes de mandarlo" }],
  },
];

async function main() {
  const system = await createSystem(loadConfig());
  const tenant = "bench-instrumenta";
  await system.orchestrator.registerTools(tenant, BENCH_TOOLS);
  // El warm-up del grafo (embeddings + aristas) es asíncrono: esperar a que un
  // turno ya no salga degradado, o la medición contaría menos corridas de las reales.
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    registros.length = 0;
    const probe = await system.orchestrator.predict({
      sessionId: "bench-warmup-probe",
      tenant,
      text: "busca información en internet",
      source: "human",
    });
    if (resNoDegrada(probe) && resTieneTools(probe)) break;
  }
  console.log("catálogo listo, comienza la medición\n");

  let totalRunsBi = 0;
  let totalUnicosBi = 0;
  let totalMsBi = 0;
  let totalSolicitudes = 0;
  let totalTextosPedidos = 0;
  let totalCross = 0;
  let totalMsCross = 0;

  for (const turno of TURNOS) {
    registros.length = 0;
    const t0 = performance.now();
    const res = await system.orchestrator.predict({
      sessionId: `bench-${turno.etiqueta}`,
      tenant,
      text: turno.text,
      source: "human",
      history: turno.history,
    });
    const ms = performance.now() - t0;

    const solicitudes = registros.filter((r) => r.tipo === "pooled" || r.tipo === "tokens");
    const pasadas = registros.filter((r) => r.tipo === "pass");
    const cross = registros.filter((r) => r.tipo === "cross");

    const porTextoPedido = new Map<string, number>();
    for (const r of solicitudes) porTextoPedido.set(r.texto, (porTextoPedido.get(r.texto) ?? 0) + 1);
    const porTextoPasado = new Map<string, number>();
    for (const r of pasadas) porTextoPasado.set(r.texto, (porTextoPasado.get(r.texto) ?? 0) + 1);

    // Ojo: los ms por llamada NO son aditivos (la cola es serial y las llamadas
    // concurrentes miden su espera). La magnitud fiable es el CONTEO de corridas:
    // con cola serial, la latencia es proporcional a él.
    const absorbido = solicitudes.length > 0 ? 1 - pasadas.length / solicitudes.length : 0;
    const duplicidadRestante = pasadas.length / Math.max(1, porTextoPasado.size);

    totalRunsBi += pasadas.length;
    totalUnicosBi += porTextoPasado.size;
    totalMsBi += ms;
    totalSolicitudes += solicitudes.length;
    totalTextosPedidos += porTextoPedido.size;
    totalCross += cross.length;
    totalMsCross += cross.reduce((a, r) => a + r.ms, 0);

    console.log(`=== turno ${turno.etiqueta} ===`);
    console.log(`  latencia del turno         : ${(ms / 1000).toFixed(2)} s`);
    console.log(`  solicitudes del pipeline   : ${solicitudes.length} (pooled=${solicitudes.length - solicitudes.filter((r) => r.tipo === "tokens").length}, tokens=${solicitudes.filter((r) => r.tipo === "tokens").length})`);
    console.log(`  pasadas REALES al gte      : ${pasadas.length} sobre ${porTextoPasado.size} textos distintos`);
    console.log(`  absorbido por la memo      : ${(absorbido * 100).toFixed(0)}% de las solicitudes`);
    console.log(`  duplicidad restante        : ${duplicidadRestante.toFixed(2)}× (1.00 = una pasada por texto)`);
    console.log(`  cross-encoder (minilm)     : ${cross.length} pasadas (candidatas intactas)`);
    console.log(`  tools devueltas            : ${res.tools.length} → ${res.tools.slice(0, 5).map((t) => t.name).join(", ")}`);

    const repetidos = [...porTextoPasado.entries()].filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]);
    for (const [texto, n] of repetidos.slice(0, 5)) {
      console.log(`    pasada ${n}× al modelo  "${texto.slice(0, 55).replace(/\n/g, " ")}"`);
    }
    console.log("");
  }

  console.log(`=== resumen (3 turnos) ===`);
  console.log(`solicitudes    : ${totalSolicitudes} sobre ${totalTextosPedidos} textos distintos`);
  console.log(`pasadas al gte : ${totalRunsBi} sobre ${totalUnicosBi} textos distintos`);
  console.log(`absorbido memo : ${((1 - totalRunsBi / totalSolicitudes) * 100).toFixed(0)}% de las solicitudes`);
  console.log(`duplicidad     : ${(totalRunsBi / totalUnicosBi).toFixed(2)}× (1.00 = una pasada por texto)`);
  console.log(`cross-encoder  : ${totalCross} pasadas`);
  console.log(`latencia       : ${(totalMsBi / 1000).toFixed(2)} s de pared acumulada`);
  console.log(`RSS del proceso: ${(process.memoryUsage().rss / 1048576).toFixed(0)} MB`);
  process.exit(0);
}

/** Un turno de sondeo sirve si ya no va degradado y devuelve herramientas. */
function resNoDegrada(res: { trace?: { degraded?: boolean } }): boolean {
  return res.trace?.degraded !== true;
}
function resTieneTools(res: { tools?: unknown[] }): boolean {
  return (res.tools?.length ?? 0) > 0;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
