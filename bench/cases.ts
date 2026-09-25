/**
 * Casos del benchmark de contexto: cada caso es una mini-conversación cuyo
 * último turno solo se resuelve bien si se entiende la intención (negación,
 * correferencia, confirmación, cambio de tema), no solo la similitud léxica.
 *
 * - `turns` se juegan completos contra remtk (mismo sessionId; el último turno
 *   es el evaluado) y se pasan a Jev como `state.history` + `state.text`.
 * - `expected`: herramientas aceptadas. Vacío = no se espera ninguna.
 * - `multi`: el turno requiere más de una herramienta (Jev `choice` solo puede
 *   acertar parcialmente: se anota como limitación del formato).
 */
export interface BenchTurn {
  role: "user" | "assistant";
  content: string;
}

export interface BenchCase {
  id: string;
  why: string;
  turns: BenchTurn[];
  expected: string[];
  multi?: boolean;
}

export const BENCH_CASES: BenchCase[] = [
  {
    id: "negacion-borrador",
    why: "La negación invierte la acción: 'correo' + 'envíes' atraen léxicamente a send_email, pero la intención es NO enviar.",
    turns: [
      { role: "user", content: "No envíes el correo todavía, mejor guárdalo como borrador" },
    ],
    expected: ["save_draft"],
  },
  {
    id: "confirmacion-plan",
    why: "'sí, hazlo' no tiene señal léxica: la herramienta correcta vive en el turno anterior.",
    turns: [
      { role: "user", content: "quiero mandar el resumen semanal al equipo por correo" },
      { role: "assistant", content: "Perfecto, ¿envío el correo con el resumen semanal al equipo?" },
      { role: "user", content: "sí, hazlo" },
    ],
    expected: ["send_email"],
  },
  {
    id: "correferencia-tiempo",
    why: "'¿y mañana?' es anafórico: solo con el historial se sabe que sigue hablando del clima.",
    turns: [
      { role: "user", content: "¿qué clima hace hoy en Lima?" },
      { role: "assistant", content: "Hoy en Lima está nublado, 18°C." },
      { role: "user", content: "¿y mañana?" },
    ],
    expected: ["get_weather"],
  },
  {
    id: "cambio-tema",
    why: "Cambio de tema explícito: el historial de correo NO debe arrastrar send_email.",
    turns: [
      { role: "user", content: "envía la factura al cliente por correo" },
      { role: "assistant", content: "Correo enviado al cliente." },
      { role: "user", content: "ahora sí, ¿a cuánto está el dólar hoy?" },
    ],
    expected: ["get_exchange_rate"],
  },
  {
    id: "cross-idioma",
    why: "La petición viene en inglés y el catálogo está en español.",
    turns: [
      { role: "user", content: "send the monthly report to my team by email" },
    ],
    expected: ["send_email"],
  },
  {
    id: "multi-intencion",
    why: "Dos acciones encadenadas en un solo turno: descargar Y enviar.",
    turns: [
      { role: "user", content: "descarga el reporte de ventas de ayer y envíamelo por correo" },
    ],
    expected: ["download_report", "send_email"],
    multi: true,
  },
  {
    id: "meta-pregunta",
    why: "Pregunta sobre capacidades: no debe disparar ninguna herramienta.",
    turns: [{ role: "user", content: "¿qué herramientas tienes disponibles?" }],
    expected: [],
  },
  {
    id: "small-talk",
    why: "Saludo puro: no debe disparar ninguna herramienta.",
    turns: [{ role: "user", content: "hola, ¿cómo estás?" }],
    expected: [],
  },
  {
    id: "pronombre-contexto",
    why: "'muéstrame la última otra vez' solo se resuelve con el historial (facturas).",
    turns: [
      { role: "user", content: "lista mis facturas de junio" },
      { role: "assistant", content: "Aquí están tus 8 facturas de junio." },
      { role: "user", content: "muéstrame la última otra vez" },
    ],
    expected: ["list_invoices"],
  },
  {
    id: "intencion-implicita",
    why: "No hay verbo del catálogo: 'mi jefe necesita el balance' implica descargar el reporte.",
    turns: [
      { role: "user", content: "mi jefe me pide el balance de ventas de ayer, es urgente" },
    ],
    expected: ["download_report"],
  },
  {
    id: "rechazo-propuesta",
    why: "Rechazo de una propuesta: no debe disparar la herramienta propuesta ni ninguna.",
    turns: [
      { role: "user", content: "¿me agendarías la reunión con ventas mañana?" },
      { role: "assistant", content: "¿Agendo el evento en tu calendario para mañana 10am?" },
      { role: "user", content: "no, todavía no" },
    ],
    expected: [],
  },
  {
    id: "herramienta-explicita",
    why: "El usuario nombra la herramienta literalmente: debe respetarse la orden.",
    turns: [
      { role: "user", content: "usa web_search para buscar el precio del dólar en Perú" },
    ],
    expected: ["web_search"],
  },
];
