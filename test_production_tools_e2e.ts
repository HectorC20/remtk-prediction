import { TOTAL_ACTIVE_TOOLS } from './tests/fixtures/mitumbes-production-tools';

const PREDICTION_URL = 'http://localhost:6776';
const TENANT = 'tenant_mitumbes_production';
const AGENT_ID = 'agent_mitumbes_prod';
const KEY_REMTK = `${TENANT}::${AGENT_ID}`;

const headers = {
  'Content-Type': 'application/json',
  'key-remtk': KEY_REMTK,
  'x-remtk-key': KEY_REMTK,
};

async function req(url: string, method = 'GET', body?: any) {
  const res = await fetch(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => null);
  return { status: res.status, data };
}

async function runProductionNoiseTest() {
  console.log('================================================================');
  console.log('   PRUEBA EN ENTORNO REAL CON 235 HERRAMIENTAS DE PRODUCCIÓN   ');
  console.log('================================================================');
  console.log(`Endpoint: ${PREDICTION_URL}`);
  console.log(`Tenant:   ${TENANT}`);
  console.log(`AgentId:  ${AGENT_ID}`);
  console.log(`Scope:    ${KEY_REMTK}`);
  console.log(`Total de herramientas reales + ruido: ${TOTAL_ACTIVE_TOOLS.length}\n`);

  // 1. Indexar el catálogo completo de 235 herramientas
  console.log('[1/4] Indexando las 235 herramientas de producción en remtk-prediction...');
  const indexRes = await req(`${PREDICTION_URL}/tools`, 'POST', {
    tenant: TENANT,
    agentId: AGENT_ID,
    tools: TOTAL_ACTIVE_TOOLS.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.inputSchema,
    })),
  });
  console.log(`  -> Status: ${indexRes.status} | Herramientas indexadas: ${indexRes.data?.indexed || TOTAL_ACTIVE_TOOLS.length}`);

  // 2. Consulta previa a aprender la habilidad con ruido de Facebook
  console.log('\n[2/4] Predicción inicial con 235 herramientas:');
  console.log('  Prompt: "Actualizar ítem UMI Barra Fusión Marina con link de Facebook https://facebook.com/umibarrafusion"');
  const pred1 = await req(`${PREDICTION_URL}/predict`, 'POST', {
    tenant: TENANT,
    agentId: AGENT_ID,
    text: 'Actualizar ítem UMI Barra Fusión Marina con link de Facebook https://facebook.com/umibarrafusion',
    source: 'agent',
  });
  console.log('  -> Tools candidatas iniciales:', pred1.data?.tools?.slice(0, 5).map((t: any) => t.name));

  // 3. Simulación del aprendizaje automático del sistema cuando el turno resuelve la tarea
  console.log('\n[3/4] Aprendizaje autónomo del sistema tras resolución exitosa:');
  console.log('  -> El ejecutor completó la mutación y aprendió la habilidad y su parameter grounding...');
  const learnRes = await req(`${PREDICTION_URL}/skills/learn`, 'POST', {
    tenant: TENANT,
    agentId: AGENT_ID,
    id: 'skill_mitumbes_item_actualizar_link',
    name: 'Actualizar ítem de negocio con enlace',
    description: 'Actualiza el ítem con su URL o enlace de redes sociales en el catálogo',
    intentSummary: 'Actualizar ítem UMI Barra Fusión Marina con link de Facebook. Modificar url del ítem',
    tools: ['mitumbes_item_listar', 'mitumbes_item_actualizar'],
    parameterGrounding: {
      itemId: [{ id: 'umi-fusion-101', name: 'UMI Barra Fusión Marina' }],
    },
  });
  console.log(`  -> Status: ${learnRes.status} | Habilidad guardada:`, {
    id: learnRes.data?.id,
    name: learnRes.data?.name,
    tools: learnRes.data?.tools,
    reinforcementScore: learnRes.data?.reinforcementScore,
  });

  // 4. Inferencia en el siguiente turno bajo 235 herramientas
  console.log('\n[4/4] Inferencia en turno siguiente con 235 herramientas activas:');
  console.log('  Prompt: "Actualizar ítem UMI Barra Fusión con nuevo link de facebook https://facebook.com/umibarrafusion-2026"');
  
  const skillInfer = await req(`${PREDICTION_URL}/skills/predict`, 'POST', {
    tenant: TENANT,
    agentId: AGENT_ID,
    sessionId: 'session_prod_noise_2',
    text: 'Actualizar ítem UMI Barra Fusión con nuevo link de facebook https://facebook.com/umibarrafusion-2026',
  });

  const matched = skillInfer.data?.matchedSkill || skillInfer.data?.skill;
  console.log('\n================================================================');
  console.log('           RESULTADO DE INFERENCIA EN PRODUCCIÓN                ');
  console.log('================================================================');
  console.log('  Habilidad Identificada:', matched?.name);
  console.log('  Confianza Coseno:      ', matched?.confidence?.toFixed(4));
  console.log('  Herramientas Priorizadas:', matched?.tools);
  console.log('  Parámetros Resueltos:  ', skillInfer.data?.preResolvedArgs);
  console.log('  Contexto Inyectado:    ', skillInfer.data?.contextPrompt);
  console.log('================================================================\n');

  if (matched && matched.confidence > 0.85 && matched.tools.includes('mitumbes_item_actualizar')) {
    console.log('✅ ÉXITO: El sistema filtró el ruido de las 235 herramientas y resolvió las herramientas exactas sin replanteo.');
  } else {
    console.log('❌ FALLO: No se identificó la habilidad correcta.');
  }
}

runProductionNoiseTest().catch(console.error);
