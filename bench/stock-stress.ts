/**
 * Benchmark de Catálogo Extenso (74+ Herramientas en Stock):
 * Somete a prueba la capacidad de discriminación vectorial del predictor
 * cuando el tenant tiene un inventario masivo de herramientas con dominios
 * cercanos y potencialmente ambiguos (CRM, ERP, CMS, DevOps, FS, Comms, Data).
 *
 * pnpm bench:stock
 */
import { BENCH_TOOLS } from "./catalog";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";
import { RemtkBenchClient } from "./remtk.client";

// ── 1. HERRAMIENTAS ADICIONALES ENTERPRISE (CRM, ERP, DevOps, HR) ─────────────
const ENTERPRISE_TOOLS: ToolDefinition[] = [
  {
    id: "crm_contact_update",
    name: "crm_contact_update",
    group: "crm",
    category: "contacts",
    description: "Actualiza los datos de un contacto o cliente en el CRM empresarial.",
    tags: ["crm", "contacto", "cliente", "actualizar"],
    intentSummary: "Actualizar datos de contacto en CRM",
    inputSchema: { contactId: "string", email: "string", phone: "string" },
  },
  {
    id: "crm_lead_convert",
    name: "crm_lead_convert",
    group: "crm",
    category: "leads",
    description: "Convierte un prospecto (lead) calificado en una oportunidad de venta.",
    tags: ["crm", "lead", "prospecto", "convertir", "venta"],
    intentSummary: "Convertir lead a oportunidad de venta",
    inputSchema: { leadId: "string", dealValue: "number" },
  },
  {
    id: "finance_invoice_generate",
    name: "finance_invoice_generate",
    group: "finance",
    category: "billing",
    description: "Emite y timbra una nueva factura fiscal para un cliente.",
    tags: ["factura", "emitir", "crear", "timbrar", "finanzas"],
    intentSummary: "Emitir una nueva factura fiscal",
    inputSchema: { customerId: "string", items: "array", total: "number" },
  },
  {
    id: "finance_payment_refund",
    name: "finance_payment_refund",
    group: "finance",
    category: "payments",
    description: "Procesa la devolución o reembolso parcial/total de un pago.",
    tags: ["reembolso", "devolucion", "pago", "finanzas"],
    intentSummary: "Procesar reembolso de pago",
    inputSchema: { transactionId: "string", amount: "number", reason: "string" },
  },
  {
    id: "devops_deployment_trigger",
    name: "devops_deployment_trigger",
    group: "devops",
    category: "ci_cd",
    description: "Dispara el despliegue automático de un microservicio a staging o producción.",
    tags: ["despliegue", "deploy", "pipeline", "devops"],
    intentSummary: "Desplegar microservicio en producción o staging",
    inputSchema: { service: "string", environment: "string", commit: "string" },
  },
  {
    id: "devops_logs_view",
    name: "devops_logs_view",
    group: "devops",
    category: "monitoring",
    description: "Consulta los logs de auditoría y trazas de un contenedor o pod.",
    tags: ["logs", "trazas", "auditoria", "monitoring"],
    intentSummary: "Ver logs de auditoría o monitoreo",
    inputSchema: { podName: "string", lines: "number" },
  },
  {
    id: "hr_employee_onboard",
    name: "hr_employee_onboard",
    group: "hr",
    category: "employees",
    description: "Registra el alta de un nuevo colaborador en la plataforma de recursos humanos.",
    tags: ["rrhh", "empleado", "alta", "onboarding"],
    intentSummary: "Dar de alta un nuevo empleado",
    inputSchema: { name: "string", role: "string", startDate: "string" },
  },
  {
    id: "hr_leave_request",
    name: "hr_leave_request",
    group: "hr",
    category: "vacations",
    description: "Solicita días de vacaciones o permiso laboral en recursos humanos.",
    tags: ["rrhh", "vacaciones", "permiso", "solicitud"],
    intentSummary: "Solicitar vacaciones o permiso laboral",
    inputSchema: { startDate: "string", endDate: "string", reason: "string" },
  },
  {
    id: "inventory_stock_check",
    name: "inventory_stock_check",
    group: "logistics",
    category: "inventory",
    description: "Consulta la disponibilidad física de existencias o stock de un producto en almacén.",
    tags: ["inventario", "stock", "existencias", "almacen"],
    intentSummary: "Consultar el stock disponible de un producto",
    inputSchema: { sku: "string", warehouseId: "string" },
  },
  {
    id: "inventory_order_create",
    name: "inventory_order_create",
    group: "logistics",
    category: "orders",
    description: "Crea una orden de reposición o pedido de compra a proveedores de almacén.",
    tags: ["inventario", "pedido", "compra", "proveedor"],
    intentSummary: "Crear una orden de compra a proveedores",
    inputSchema: { supplierId: "string", sku: "string", quantity: "number" },
  },
  {
    id: "support_ticket_assign",
    name: "support_ticket_assign",
    group: "support",
    category: "helpdesk",
    description: "Reasigna un ticket de soporte técnico a un agente o departamento especializado.",
    tags: ["soporte", "ticket", "reasignar", "helpdesk"],
    intentSummary: "Reasignar un ticket de soporte técnico",
    inputSchema: { ticketId: "string", agentId: "string" },
  },
];

// ── 2. HERRAMIENTAS MITUMBES (CMS/TURISMO REAL) ──────────────────────────────
const MITUMBES_ENTITIES = [
  "item", "categoria", "lugar", "evento", "ruta", "actividad",
  "servicio", "zona", "post", "media", "usuario", "resena", "reserva"
];

function buildMitumbesStock(): ToolDefinition[] {
  const tools: ToolDefinition[] = [];
  for (const ent of MITUMBES_ENTITIES) {
    for (const op of ["listar", "obtener", "crear", "actualizar", "eliminar"]) {
      const fullName = `mitumbes_${ent}_${op}`;
      tools.push({
        id: fullName,
        name: fullName,
        group: "plugin_mitumbes",
        category: "plugin_tool",
        description: `Operación ${op} sobre la entidad ${ent} en la plataforma MiTumbes.`,
        tags: ["MiTumbes", ent, op],
        intentSummary: `${op} ${ent} en MiTumbes`,
        inputSchema: { id: "string", query: "string" },
      });
    }
  }
  return tools;
}

/** Catálogo Stock Masivo: 17 base + 11 enterprise + 65 mitumbes = 93 HERRAMIENTAS */
export const MASSIVE_STOCK_CATALOG: ToolDefinition[] = [
  ...BENCH_TOOLS,
  ...ENTERPRISE_TOOLS,
  ...buildMitumbesStock(),
];

// ── 3. CASOS DE PRUEBA SOBRE CATÁLOGO MASIVO ─────────────────────────────────
interface StockCase {
  id: string;
  description: string;
  context: string;
  expectedTop: string;
  expectedIncludes?: string[];
}

const STOCK_CASES: StockCase[] = [
  {
    id: "stock-1-crm-vs-cms",
    description: "Diferencia actualizar contacto de CRM vs actualizar item de CMS en texto con preámbulo",
    context: "Estaba conversando con el área comercial sobre los nuevos clientes. Necesito actualizar los datos del contacto del cliente en el CRM.",
    expectedTop: "crm_contact_update",
  },
  {
    id: "stock-2-devops-vs-fs",
    description: "Consulta de logs en DevOps entre herramientas de archivos y base de datos",
    context: "El servicio en producción está dando errores de timeout. Necesito ver los logs de auditoría del contenedor para diagnosticar la falla.",
    expectedTop: "devops_logs_view",
  },
  {
    id: "stock-3-reembolso-vs-factura",
    description: "Procesar reembolso en finanzas en lugar de emitir o listar facturas",
    context: "El cliente solicitó la devolución de su dinero por una duplicidad de cobro. Hay que procesar el reembolso de su pago.",
    expectedTop: "finance_payment_refund",
  },
  {
    id: "stock-4-stock-vs-reporte",
    description: "Consulta de existencias de almacén sin confundir con descarga de reporte",
    context: "El vendedor necesita saber cuántas unidades físicas nos quedan del producto SKU-99 en el almacén central.",
    expectedTop: "inventory_stock_check",
  },
  {
    id: "stock-5-vacaciones-rrhh",
    description: "Solicitud de vacaciones en RRHH rodeada de conversación sobre el proyecto",
    context: "El proyecto viene muy avanzado y el equipo ha trabajado bastante. Quería pedir el registro de mis días de vacaciones para la próxima semana.",
    expectedTop: "hr_leave_request",
  },
  {
    id: "stock-6-mitumbes-categoria-vs-item",
    description: "Precisión quirúrgica en catálogo masivo para listar categorías de CMS",
    context: "Dame la lista de categorías del sistema para revisar las opciones disponibles.",
    expectedTop: "mitumbes_categoria_listar",
    expectedIncludes: ["mitumbes_categoria_listar"],
  },
  {
    id: "stock-7-multi-intencion-masiva",
    description: "Multi-intención en catálogo de 93 tools: consultar clima + tipo de cambio + enviar correo",
    context: "Descarga el reporte de ventas, revisa el tipo de cambio del dólar y envíame todo por correo.",
    expectedTop: "download_report",
    expectedIncludes: ["download_report", "get_exchange_rate", "send_email"],
  },
];

async function main(): Promise<void> {
  const client = new RemtkBenchClient();
  console.log("=================================================================");
  console.log(`   REMTK-PREDICTION: BENCHMARK DE CATÁLOGO MASIVO (STOCK: ${MASSIVE_STOCK_CATALOG.length} TOOLS)`);
  console.log("=================================================================\n");
  console.log(`[stock-bench] Registrando catálogo masivo de ${MASSIVE_STOCK_CATALOG.length} herramientas...`);
  await client.start(MASSIVE_STOCK_CATALOG, { qdrant: false });

  let passed = 0;

  for (const c of STOCK_CASES) {
    const res = await client.predict(`stock-${c.id}`, c.context, []);
    const tools = res.tools;
    const topTool = tools[0] ?? "∅";

    const topOk = topTool === c.expectedTop;
    const includesOk = c.expectedIncludes
      ? c.expectedIncludes.every((inc) => tools.includes(inc))
      : true;

    const ok = topOk && includesOk;
    if (ok) passed++;

    const mark = ok ? "✓ [PASS]" : "✗ [FAIL]";
    console.log(`${mark} ${c.id.padEnd(32)} | Top Esperado: ${c.expectedTop.padEnd(25)} | Top Obtenido: ${topTool}`);
    console.log(`   └─ Petición: "${c.context.slice(0, 80)}..."`);
    console.log(`   └─ Salida completa (${tools.length} tools): [${tools.slice(0, 5).join(", ")}]`);
  }

  await client.close();

  console.log("\n=================================================================");
  console.log(` RESULTADO FINAL DE STOCK MASIVO (${MASSIVE_STOCK_CATALOG.length} TOOLS): ${passed}/${STOCK_CASES.length} (${((passed / STOCK_CASES.length) * 100).toFixed(1)}%)`);
  console.log("=================================================================\n");

  if (passed < STOCK_CASES.length) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fallo en el benchmark de stock masivo:", err);
  process.exit(1);
});
