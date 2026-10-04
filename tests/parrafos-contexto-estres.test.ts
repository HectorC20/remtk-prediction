/**
 * Test de estrés con párrafos extensos:
 * 1. Herramientas con descripciones de múltiples párrafos (detalles, reglas, schemas).
 * 2. Memoria contextual extensa (múltiples turnos con párrafos de historial y contexto acumulado).
 * 3. Consultas de usuario con párrafos explicativos, contexto previo y solicitudes compuestas.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { VarianteCService } from "../src/predict/services/variante-c.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

const MULTI_PARAGRAPH_TOOLS: ToolDefinition[] = [
  {
    id: "p1",
    name: "database_analytics_query",
    group: "data",
    category: "database",
    description: `Ejecuta consultas analíticas complejas y de solo lectura sobre el Data Warehouse corporativo.

Esta herramienta se conecta a las réplicas de análisis de PostgreSQL y BigQuery para extraer métricas agregadas de clientes, estados de cuenta, facturación y tendencias de ventas. Admite filtros temporales, agrupaciones y agregaciones.

Requisitos de seguridad: No permite mutaciones (INSERT/UPDATE/DELETE). Si el usuario solicita cálculos financieros o reportes agregados que dependan de tablas de datos, esta es la herramienta primaria adecuada.`,
    tags: ["sql", "analitica", "base de datos", "reportes", "metricas"],
    intentSummary: "Ejecutar consultas analíticas en la base de datos corporativa",
  },
  {
    id: "p2",
    name: "financial_report_generator",
    group: "finance",
    category: "reports",
    description: `Genera y exporta balances financieros, estados de pérdidas y ganancias, y reportes ejecutivos en PDF/Excel.

Diseñado para consolidar información contable al cierre de cada mes o trimestre fiscal. Incluye gráficos de rendimiento, comparativas interanuales y notas de auditoría para la dirección ejecutiva.

Casos de uso: Preparar el dossier contable para la junta directiva, emitir el balance mensual de ventas, o preparar resúmenes ejecutivos listos para enviar por correo a los auditores.`,
    tags: ["balance", "financiero", "reporte", "pdf", "ventas", "contabilidad"],
    intentSummary: "Generar y descargar reportes y balances financieros ejecutivos",
  },
  {
    id: "p3",
    name: "enterprise_mail_dispatcher",
    group: "communication",
    category: "mail",
    description: `Envía comunicados, notificaciones formales y correos electrónicos corporativos a clientes y personal interno.

Permite adjuntar reportes generados, configurar múltiples destinatarios en copia (CC/CCO) y dar formato HTML al cuerpo del mensaje. Se integra con el servidor SMTP corporativo.

Importante: Úsese solo cuando exista una confirmación explícita o una orden directa de envío final al destinatario. Si el usuario aún está redactando o revisando, debe preferirse guardar en borrador.`,
    tags: ["correo", "email", "enviar", "notificacion", "comunicacion"],
    intentSummary: "Enviar correos electrónicos y comunicados corporativos",
  },
  {
    id: "p4",
    name: "draft_document_vault",
    group: "communication",
    category: "drafts",
    description: `Almacena borradores temporales de correos, propuestas comerciales o documentos en revisión sin publicarlos ni enviarlos.

Mantiene el estado editable de cualquier comunicación o documento para que el equipo pueda auditarlo, modificarlo o aprobarlo antes de su despacho final.

Regla de aplicación: Debe seleccionarse siempre que el usuario indique que la información no debe enviarse todavía, que desea revisarla después, o que prefiere dejarla pendiente de aprobación.`,
    tags: ["borrador", "guardar", "pendiente", "revision", "draft"],
    intentSummary: "Guardar borradores y documentos en revisión sin enviar",
  },
  {
    id: "p5",
    name: "exchange_rate_oracle",
    group: "finance",
    category: "forex",
    description: `Consulta cotizaciones de divisas oficiales y tipos de cambio en tiempo real (USD, EUR, PEN, etc.).

Proporciona información actualizada de los tipos de cambio de compra y venta según los bancos centrales y entidades bancarias de referencia. Soporta conversiones históricas y pronósticos a corto plazo.

Utilidad: Necesario al momento de calcular presupuestos en moneda extranjera o definir el valor en moneda local para facturas y transacciones comerciales.`,
    tags: ["tipo de cambio", "dolar", "divisa", "moneda", "cotizacion"],
    intentSummary: "Consultar cotizaciones de divisas y tipos de cambio actuales",
  },
];

interface ParagraphTestCase {
  id: string;
  name: string;
  historyText?: string;
  userQuery: string;
  expected: string[];
  why: string;
}

const PARAGRAPH_TEST_CASES: ParagraphTestCase[] = [
  {
    id: "caso-1-correo-largo-con-ruido",
    name: "Párrafo largo con contexto de negocio e intención final de envío",
    historyText: undefined,
    userQuery: `Estimado asistente, durante la sesión de trabajo de hoy con el comité de operaciones estuvimos evaluando el rendimiento del último trimestre. Hemos notado que los números en el sector retail han superado las expectativas en un 15%, mientras que los costos operativos se mantuvieron estables según las directivas del CFO.

Por favor, redacta un correo formal con estas conclusiones y envíaselo directamente a todos los gerentes de área para que estén informados antes de la reunión de mañana a las 9am.`,
    expected: ["enterprise_mail_dispatcher"],
    why: "A pesar de contener un párrafo largo de cifras y contexto de negocio, la acción ejecutiva es el despacho de correo a gerentes.",
  },
  {
    id: "caso-2-memoria-contextual-larga-con-decision-condicional",
    name: "Historial largo acumulado + resolución por pronombre y memoria",
    historyText: `Usuario: Necesitamos preparar el cierre financiero del mes de septiembre porque el directorio requiere auditar los gastos de infraestructura y las proyecciones de flujo de caja para el próximo año.
Asistente: He preparado la plantilla con los datos preliminares del balance contable de septiembre, incluyendo pérdidas, ganancias e impuestos deducibles. ¿Deseas que proceda a generarlo y exportarlo en el formato ejecutivo?`,
    userQuery: `Sí, exactamente, genera ese balance de una vez con todos los anexos para poder presentarlo hoy.`,
    expected: ["financial_report_generator"],
    why: "La memoria contextual contiene 2 párrafos largos de contexto contable; el turno del usuario 'Sí, genera ese balance' debe resolver 'financial_report_generator'.",
  },
  {
    id: "caso-3-multi-intencion-parrafos-compuestos",
    name: "Multi-intención en párrafos extensos (SQL analítico + Tipo de cambio)",
    historyText: undefined,
    userQuery: `Para la auditoría externa necesitamos cruzar dos fuentes de información críticas de la empresa.

En primer lugar, ejecuta una consulta a la base de datos analítica para obtener la lista agregada de todos los clientes corporativos con saldos vencidos mayores a 10,000 dólares en el último semestre.

Además de eso, consulta el tipo de cambio del dólar a soles del día de hoy para poder registrar los montos convertidos en el informe preliminar.`,
    expected: ["database_analytics_query", "exchange_rate_oracle"],
    why: "El texto incluye párrafos estructurados con 2 intenciones claras (consulta analítica SQL y cotización de tipo de cambio).",
  },
  {
    id: "caso-4-negacion-y-borrador-en-parrafo-largo",
    name: "Párrafo complejo con negación pragmática (Guardar borrador, NO enviar)",
    historyText: `Usuario: Tenemos lista la respuesta legal para el reclamo del proveedor sobre el retraso en la entrega de suministros.
Asistente: Tengo listo el borrador con la respuesta formal para enviarla por correo al representante legal del proveedor.`,
    userQuery: `He estado revisando la cláusula tercera y todavía no me convence la redacción de las penalidades contractuales. Por ningún motivo envíes el correo todavía a los abogados; simplemente guarda todo el texto como un borrador en el sistema hasta que finanzas dé su visto bueno.`,
    expected: ["draft_document_vault"],
    why: "El usuario incluye un párrafo de argumentación legal y una negación enfática ('Por ningún motivo envíes el correo... guarda como borrador'). Debe seleccionar 'draft_document_vault' y descartar 'enterprise_mail_dispatcher'.",
  },
];

import { loadConfig } from "../src/config";

test("Estrés de arquitectura con herramientas y memoria de múltiples párrafos", async () => {
  const cfg = loadConfig();
  const engine = new EmbeddingEngineService(cfg);
  const service = new VarianteCService(engine);

  const scope = "stress-paragraphs-test";
  const startIdx = Date.now();
  await service.indexScope(scope, MULTI_PARAGRAPH_TOOLS);
  const indexLatencyMs = Date.now() - startIdx;
  console.log(`\n[Test Párrafos] Índice construido para ${MULTI_PARAGRAPH_TOOLS.length} herramientas multi-párrafo en ${indexLatencyMs}ms`);

  let passCount = 0;

  for (const c of PARAGRAPH_TEST_CASES) {
    const t0 = Date.now();
    const rankRes = await service.rank(scope, c.userQuery, c.historyText);
    const latency = Date.now() - t0;

    assert.ok(rankRes, `El ranking no debe ser undefined para ${c.id}`);
    const arb = service.arbitrate(rankRes);

    const isPass = c.expected.every((e) => arb.tools.includes(e));
    if (isPass) passCount++;

    console.log(`\n─────────────────────────────────────────────────────────────`);
    console.log(`Caso: ${c.name} (${c.id})`);
    console.log(`Esperado:   [${c.expected.join(", ")}]`);
    console.log(`Obtenido:   [${arb.tools.join(", ") || "∅"}] (Decisión: ${arb.decision}, Régimen: ${rankRes.autonomous ? "AUTO" : "DEP"}, Margen: ${rankRes.margin.toFixed(3)})`);
    console.log(`Latencia:   ${latency}ms | ¿Pasa?: ${isPass ? "✓ PASS" : "✗ FAIL"}`);
    console.log(`Detalle C:  Top 1=${rankRes.out[0]?.name} (fused=${rankRes.out[0]?.fused.toFixed(3)})`);
  }

  console.log(`\n=============================================================`);
  console.log(`RESULTADO TEST PÁRRAFOS Y MEMORIA EXTENSA: ${passCount}/${PARAGRAPH_TEST_CASES.length} superados.`);
  console.log(`=============================================================\n`);

  assert.equal(passCount, PARAGRAPH_TEST_CASES.length, "Todos los casos de estrés de párrafos deben ser superados exitosamente");
});
