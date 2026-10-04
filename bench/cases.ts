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
  {
    id: "auto-correccion-tiempo-real",
    why: "El usuario cambia de opinión a mitad de frase: empieza pidiendo generar imagen, pero la orden final es solo escribir un archivo.",
    turns: [
      {
        role: "user",
        content: "Genera una imagen para el flyer del evento... espera, no, no generes nada todavía, solo escribe un archivo con las notas de la reunión.",
      },
    ],
    expected: ["write_file"],
  },
  {
    id: "ironia-pragmatica",
    why: "El tono irónico formula una acción que en realidad se prohíbe explícitamente.",
    turns: [
      {
        role: "user",
        content: "Sí, claro, mándale el borrador sin revisar al cliente para que vea todos nuestros errores... obviamente guárdalo para revisarlo mañana.",
      },
    ],
    expected: ["save_draft"],
  },
  {
    id: "secuestro-lexico-codigo",
    why: "Bloque de código con términos críticos que compiten con una consulta simple al final.",
    turns: [
      {
        role: "user",
        content: "Estoy revisando este script en Python: `import requests; def delete_database_records(): database_query('DROP TABLE users')`. Ignora ese código, lo que realmente necesito es saber a cuánto está el tipo de cambio del dólar hoy.",
      },
    ],
    expected: ["get_exchange_rate"],
  },
  {
    id: "doble-negacion-cruzada",
    why: "'No dejes de...' es una orden afirmativa enfática, mientras que la segunda parte es una exclusión estricta.",
    turns: [
      {
        role: "user",
        content: "No dejes de enviarle el balance de ventas de ayer a mi jefe por correo, pero por favor no toques la lista de facturas ni consultes la base de datos.",
      },
    ],
    expected: ["download_report", "send_email"],
    multi: true,
  },
  {
    id: "revocacion-post-propuesta",
    why: "El usuario rechaza una propuesta previa con lenguaje educado y vacilante.",
    turns: [
      { role: "user", content: "¿me agendarías la reunión con el equipo mañana?" },
      { role: "assistant", content: "¿Agendo el evento en tu calendario para mañana 10am?" },
      { role: "user", content: "Pensándolo bien, mejor déjalo ahí, el cliente me acaba de decir que no podrá asistir." },
    ],
    expected: [],
  },
  {
    id: "condicional-contrafactico",
    why: "La acción futura depende de un resultado que el asistente primero debe inspeccionar.",
    turns: [
      {
        role: "user",
        content: "Si el dólar está a más de 3.80 soles envíale una alerta por correo a finanzas, pero si está más bajo no hagas nada.",
      },
    ],
    expected: ["get_exchange_rate"],
  },
  {
    id: "polisemia-dominio-cruzado",
    why: "La palabra 'fondo' se usa con dos significados totalmente distintos en la misma orden.",
    turns: [
      {
        role: "user",
        content: "Genera una imagen con el fondo nuevo para la web y luego revisa el tipo de cambio para el fondo de inversiones.",
      },
    ],
    expected: ["image_generate", "get_exchange_rate"],
    multi: true,
  },
  {
    id: "multi-leer-resumir",
    why: "Dos acciones acopladas: lectura de archivo y resumen de documento.",
    turns: [
      {
        role: "user",
        content: "Lee el archivo propuesta.pdf en el workspace y genera un resumen del documento para la gerencia.",
      },
    ],
    expected: ["read_file", "summarize_document"],
    multi: true,
  },
  {
    id: "multi-clima-whatsapp",
    why: "Consulta de pronóstico del tiempo y notificación inmediata por mensajería.",
    turns: [
      {
        role: "user",
        content: "Revisa qué clima hace hoy en Trujillo y envíale un mensaje de WhatsApp a Carlos con el pronóstico.",
      },
    ],
    expected: ["get_weather", "send_whatsapp"],
    multi: true,
  },
  {
    id: "multi-scrape-archivo",
    why: "Extracción web y persistencia de contenido en archivo.",
    turns: [
      {
        role: "user",
        content: "Extrae el contenido de la URL del portal de noticias y escribe un archivo con los datos recopilados.",
      },
    ],
    expected: ["scrape_url", "write_file"],
    multi: true,
  },
  {
    id: "multi-tipo-cambio-calendario",
    why: "Consulta de finanzas/moneda y creación de evento de calendario coordinado.",
    turns: [
      {
        role: "user",
        content: "Consulta el tipo de cambio actual del dólar y crea un evento en el calendario para la reunión de finanzas a las 3pm.",
      },
    ],
    expected: ["get_exchange_rate", "create_calendar_event"],
    multi: true,
  },
  {
    id: "multi-traducir-correo",
    why: "Traducción de idioma y envío por correo electrónico.",
    turns: [
      {
        role: "user",
        content: "Traduce este texto técnico al inglés y envía un correo electrónico al cliente con la traducción.",
      },
    ],
    expected: ["translate_text", "send_email"],
    multi: true,
  },
  {
    id: "multi-sql-reporte",
    why: "Consulta de base de datos relacional y descarga de balance/reporte.",
    turns: [
      {
        role: "user",
        content: "Ejecuta una consulta SQL en la base de datos para ver los usuarios y descarga el reporte de ventas de este mes.",
      },
    ],
    expected: ["database_query", "download_report"],
    multi: true,
  },
];
