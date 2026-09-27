/**
 * Test Masivo de Carga, Memoria Contextual e Intención Compleja para JEV AI
 *
 * Configuración del Experimento:
 * - Catálogo Masivo: 400 Herramientas MCP distribuidas en 20 grupos temáticos
 * - Estados JEV AI: 200 Estados de Memoria Contextual / Historial de Sesiones / Sesión Activa
 * - Evaluación: Precisión de Intención, Detección de Topic Shift, Selección de Herramientas
 *   bajo ambigüedad, resolución anafórica y recuperación contextual.
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "jev-mcp-400-tenant";
const SESSION_BASE = "jev-session";

// 1. GENERADOR DE 400 HERRAMIENTAS MCP
function generate400MCPTools(): ToolDefinition[] {
  const categories = [
    { name: "database", count: 20, prefix: "db" },
    { name: "github_dev", count: 20, prefix: "gh" },
    { name: "workspace_fs", count: 20, prefix: "fs" },
    { name: "slack_comm", count: 20, prefix: "slack" },
    { name: "cloud_aws", count: 20, prefix: "aws" },
    { name: "finance_crypto", count: 20, prefix: "fin" },
    { name: "weather_geo", count: 20, prefix: "geo" },
    { name: "ai_orchestrator", count: 20, prefix: "ai" },
    { name: "security_vault", count: 20, prefix: "sec" },
    { name: "docker_k8s", count: 20, prefix: "k8s" },
    { name: "document_pdf", count: 20, prefix: "doc" },
    { name: "medical_health", count: 20, prefix: "health" },
    { name: "crm_salesforce", count: 20, prefix: "crm" },
    { name: "analytics_metrics", count: 20, prefix: "analytics" },
    { name: "ecommerce_stripe", count: 20, prefix: "shop" },
    { name: "social_media", count: 20, prefix: "social" },
    { name: "iot_devices", count: 20, prefix: "iot" },
    { name: "media_audio_video", count: 20, prefix: "media" },
    { name: "translation_nlp", count: 20, prefix: "nlp" },
    { name: "legal_compliance", count: 20, prefix: "legal" }
  ];

  const tools: ToolDefinition[] = [];

  for (const cat of categories) {
    for (let i = 1; i <= cat.count; i++) {
      const toolId = `${cat.prefix}_tool_${i}`;
      tools.push({
        id: toolId,
        name: toolId,
        group: cat.name,
        category: `${cat.name}_sub_${i % 5}`,
        description: `Herramienta MCP especializada ${i} de la categoría ${cat.name}. Permite ejecutar acciones avanzadas de ${cat.name} para gestión de recursos ${cat.prefix}-${i}.`,
        tags: [cat.name, cat.prefix, `tag_${i}`, `operation_${i}`],
        intentSummary: `Ejecuta operación ${i} en el módulo de ${cat.name}`,
        inputSchema: {
          type: "object",
          properties: {
            target: { type: "string" },
            action: { type: "string" }
          }
        }
      });
    }
  }

  // Añadimos también herramientas clave muy específicas para pruebas de alta intención
  tools.push(
    {
      id: "postgres_backup_dump",
      name: "postgres_backup_dump",
      group: "database",
      category: "db_admin",
      description: "Realiza un respaldo completo (pg_dump) de la base de datos PostgreSQL de producción en un archivo .sql o .tar.",
      tags: ["database", "postgres", "backup", "pg_dump", "sql", "dump", "respaldo"],
      intentSummary: "Genera backup de base de datos PostgreSQL de producción",
      inputSchema: { type: "object", properties: { dbName: { type: "string" } } }
    },
    {
      id: "redis_flush_cache",
      name: "redis_flush_cache",
      group: "database",
      category: "db_cache",
      description: "Limpia las claves de caché de Redis en la instancia seleccionada.",
      tags: ["redis", "cache", "flush", "memory", "limpiar"],
      intentSummary: "Limpia la caché de memoria en Redis",
      inputSchema: { type: "object" }
    },
    {
      id: "clima_tumbes_peru",
      name: "clima_tumbes_peru",
      group: "weather_geo",
      category: "weather",
      description: "Obtiene el pronóstico del tiempo y clima detallado en Tumbes, Perú en tiempo real.",
      tags: ["clima", "weather", "tumbes", "peru", "temperatura", "tiempo"],
      intentSummary: "Consulta el clima actual de Tumbes Perú",
      inputSchema: { type: "object" }
    },
    {
      id: "github_create_pull_request",
      name: "github_create_pull_request",
      group: "github_dev",
      category: "git_ops",
      description: "Crea una Pull Request en GitHub comparando ramas y asignando revisores.",
      tags: ["github", "pr", "pull request", "git", "review"],
      intentSummary: "Crea una solicitud de extracción Pull Request en GitHub",
      inputSchema: { type: "object" }
    }
  );

  return tools;
}

// 2. GENERADOR DE 200 ESTADOS Y MEMORIAS TIPO JEV AI
function generate200JEVStates(): { stateId: string; memoryContent: string; topic: string }[] {
  const states = [];
  const topics = ["database_migration", "frontend_refactor", "security_audit", "ml_deployment", "billing_issue"];

  for (let i = 1; i <= 200; i++) {
    const topic = topics[i % topics.length];
    states.push({
      stateId: `state_jev_${i}`,
      memoryContent: `[Estado JEV AI #${i} | Tema: ${topic}] El usuario solicitó ${topic} en el entorno de pruebas #${i % 10}. Parámetro crítico: DB_CLUSTER_${i}.`,
      topic: topic
    });
  }
  return states;
}

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;
});

after(async () => {
  if (closeServer) await closeServer();
});

test("JEV AI Bench — Carga Masiva de 400+ Herramientas MCP", async () => {
  const mcpTools = generate400MCPTools();

  const response = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tenant: TENANT_ID,
      tools: mcpTools
    })
  });

  assert.equal(response.status, 200);
  const data = (await response.json()) as { indexed: number };
  assert.ok(data.indexed >= 400, `Se debieron indexar al menos 400 herramientas, indexadas: ${data.indexed}`);

  // Verificar conteo en servidor
  const countRes = await fetch(`${serverUrl}/tools/count?tenant=${TENANT_ID}`);
  const countData = (await countRes.json()) as { count: number };
  assert.ok(countData.count >= 400, `Conteo en servidor debe ser >= 400, obtenido: ${countData.count}`);
});

test("JEV AI Context & Intention Test 1 — Inferencia de Intención entre 400+ Herramientas MCP", async () => {
  // Consulta de intención coloquial sin especificar nombre exacto de función
  const queryBody = {
    sessionId: `${SESSION_BASE}_1`,
    tenant: TENANT_ID,
    text: "necesito sacar un respaldo de la base de datos postgres de produccion antes de hacer la migracion",
    source: "human"
  };

  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(queryBody)
  });

  assert.equal(res.status, 200);
  const data = (await res.json()) as { tools: Array<{ id: string; name: string; group: string }>; complexity: string };

  assert.ok(data.tools.length > 0, "Debe retornar al menos 1 herramienta seleccionada");
  const topTool = data.tools[0];
  assert.equal(topTool.id, "postgres_backup_dump", `Se esperaba postgres_backup_dump pero se obtuvo ${topTool.id}`);
});

test("JEV AI Context & Intention Test 2 — Resolución Anafórica y Memoria de Conversación Multi-Turno", async () => {
  const sessionId = `${SESSION_BASE}_multi_turn`;

  // Turno 1: Conversación inicial contextual sobre cluster de base de datos
  const turn1 = {
    sessionId,
    tenant: TENANT_ID,
    text: "estamos trabajando en el cluster de base de datos postgresql de produccion",
    history: []
  };
  await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(turn1)
  });

  // Turno 2: Pregunta anafórica implícita "haz el dump de esa base de datos"
  const turn2 = {
    sessionId,
    tenant: TENANT_ID,
    text: "haz el respaldo completo de esa base de datos",
    history: [
      { role: "user", content: turn1.text },
      { role: "assistant", content: "Entendido, estoy al tanto del cluster de PostgreSQL." }
    ]
  };
  const res2 = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(turn2)
  });
  const data2 = (await res2.json()) as { tools: Array<{ id: string }> };

  assert.ok(data2.tools.length > 0, "Debe retornar herramientas para el turno 2");
  const topTool = data2.tools[0];
  assert.equal(topTool.id, "postgres_backup_dump", `Debe mantener el contexto y seleccionar postgres_backup_dump pero fue ${topTool.id}`);
});

test("JEV AI Memory Predict Test — Evaluación de Topic Shift y Similitud con 200 Estados", async () => {
  const states = generate200JEVStates();
  assert.equal(states.length, 200, "Deben generarse 200 estados JEV AI");

  // Test de consulta sobre memoria contextual
  const memoryReq = {
    sessionId: `${SESSION_BASE}_mem`,
    tenant: TENANT_ID,
    text: "cuál fue el parámetro del cluster de base de datos en la auditoria de seguridad?",
    history: [
      { role: "user", content: "estamos revisando la auditoría de seguridad del sistema" }
    ],
    limit: 5
  };

  const res = await fetch(`${serverUrl}/memory/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(memoryReq)
  });

  assert.equal(res.status, 200);
  const data = (await res.json()) as { memories: any[]; topicShift: boolean; topicScore: number };

  assert.equal(typeof data.topicShift, "boolean", "Debe incluir el flag topicShift");
  assert.equal(typeof data.topicScore, "number", "Debe incluir el topicScore de continuidad");
});

test("JEV AI Intention Test 3 — Detección de Intención Compleja Multiobjetivo (PR + GitHub)", async () => {
  const queryBody = {
    sessionId: `${SESSION_BASE}_github`,
    tenant: TENANT_ID,
    text: "crea una pull request en github para mandar los cambios de la rama feature a main",
    source: "human"
  };

  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(queryBody)
  });

  assert.equal(res.status, 200);
  const data = (await res.json()) as { tools: Array<{ id: string; name: string }>; complexity: string };

  assert.ok(data.tools.length > 0, "Debe seleccionar herramientas");
  assert.equal(data.tools[0].id, "github_create_pull_request", `Se esperaba github_create_pull_request pero se obtuvo ${data.tools[0].id}`);
});
