/**
 * Catálogo del benchmark: las 12 herramientas de ejemplo + 5 específicas para
 * los casos de contexto (borrador, clima, tipo de cambio, reportes, facturas).
 * El mismo catálogo alimenta a remtk-prediction (POST /tools) y a Jev
 * (criterios de la pregunta `choice`), para que la comparación sea pareja.
 */
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";
import { EXAMPLE_TOOLS } from "../tests/example-tools";

export const BENCH_TOOLS: ToolDefinition[] = [
  ...EXAMPLE_TOOLS,
  {
    id: "13",
    name: "save_draft",
    group: "communication",
    category: "mail",
    description: "Guarda un correo electrónico como borrador sin enviarlo.",
    tags: ["borrador", "correo", "guardar"],
    intentSummary: "Guardar un borrador de correo",
    problemSpace: "no quiero enviarlo todavía, déjalo pendiente para revisarlo después",
    inputSchema: { to: "string", subject: "string", body: "string" },
  },
  {
    id: "14",
    name: "get_weather",
    group: "data",
    category: "weather",
    description: "Obtiene el pronóstico del clima actual o de próximos días para una ciudad.",
    tags: ["clima", "pronostico", "tiempo"],
    intentSummary: "Consultar el clima",
    problemSpace: "no sé si llevar paraguas o abrigo, qué tiempo hace afuera",
    inputSchema: { city: "string", date: "string" },
  },
  {
    id: "15",
    name: "get_exchange_rate",
    group: "data",
    category: "finance",
    description: "Consulta el tipo de cambio actual entre dos monedas, por ejemplo dólar a soles.",
    tags: ["dolar", "tipo de cambio", "moneda"],
    intentSummary: "Consultar el tipo de cambio",
    problemSpace: "necesito cambiar dólares a soles y quiero saber a cuánto está hoy",
    inputSchema: { from: "string", to: "string" },
  },
  {
    id: "16",
    name: "download_report",
    group: "data",
    category: "reports",
    description: "Genera y descarga un reporte o balance de ventas de un período.",
    tags: ["reporte", "balance", "ventas", "descargar"],
    intentSummary: "Descargar un reporte",
    problemSpace: "mi jefe me pide el balance de ventas, necesito presentar las cifras del período",
    inputSchema: { period: "string" },
  },
  {
    id: "17",
    name: "list_invoices",
    group: "data",
    category: "billing",
    description: "Lista las facturas emitidas en un mes o período determinado.",
    tags: ["facturas", "listar", "billing"],
    intentSummary: "Listar facturas",
    problemSpace: "el contador me pide cuánto facturé este mes, tengo que revisar las boletas",
    inputSchema: { period: "string" },
  },
];
