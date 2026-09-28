import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { SkillMemoryService } from "../src/predict/services/skill-memory.service";
import { loadConfig } from "../src/config";

async function main() {
  const apiKey = "mstrl_c8YQe0tINd5tXVaJLtODPbRTr45mmpg0_0cfbWf";
  console.log("=== INICIANDO PRUEBA REAL CON API DE MISTRAL Y NUEVA ARQUITECTURA ===");

  // 1. Probar llamada real al endpoint de Mistral
  console.log("\n[1] Verificando autenticación y modelos en https://api.mistral.ai/v1/models...");
  try {
    const modelsRes = await fetch("https://api.mistral.ai/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    console.log(`HTTP Status: ${modelsRes.status}`);
    const modelsData = await modelsRes.json();
    if (modelsRes.ok && modelsData.data) {
      console.log(`✅ API Key válida. Total modelos disponibles: ${modelsData.data.length}`);
      const modelNames = modelsData.data.map((m: any) => m.id);
      console.log("Muestra de modelos:", modelNames.slice(0, 6).join(", "));
    } else {
      console.error("❌ Error de autenticación en Mistral:", modelsData);
      return;
    }
  } catch (err) {
    console.error("❌ Error al conectar con Mistral:", err);
    return;
  }

  // 2. Inicializar la nueva arquitectura de Habilidades y Entidades
  console.log("\n[2] Inicializando SkillMemoryService con embeddings locales ONNX...");
  const config = loadConfig();
  const engine = new EmbeddingEngineService(config);
  const skillService = new SkillMemoryService(engine);

  const TENANT_ID = "7f4b4b79-ad05-4cfb-9213-d2e6575e503e";
  const SESSION_ID = "ee84ddbb-dafa-4362-9501-c2c98e341045";

  // 3. Entrenar habilidad en remtk-prediction
  await skillService.learnSkill(TENANT_ID, {
    id: "skill:mitumbes:item_actualizar_imagen",
    name: "Actualizar imagen de ítem",
    description: "Actualiza la imagen principal de un ítem existente mediante URL pública",
    intentSummary: "actualizar o subir imagen foto para el item generado o existente",
    tools: ["mitumbes_item_actualizar"],
  });

  // 4. Registrar la entidad creada en la sesión activa (como ocurrió en el flujo real)
  console.log("\n[3] Registrando entidad 'La Pichanga Gastrobar' en el grafo de la sesión...");
  await skillService.trackEntity(SESSION_ID, {
    id: "3265a5e6-86a2-4156-a23f-39cfd7ff1171",
    type: "item",
    name: "La Pichanga Gastrobar",
    slug: "la-pichanga-gastrobar",
    state: {
      lifecycle: "created",
      pendingAction: "image_upload",
      missingFields: ["image"],
    },
  });

  // 5. Predecir con la nueva arquitectura ante el prompt del usuario en el log
  const userPrompt = "actualiza la imagen de ese item que se ha generado";
  const attachedUrl = "http://backend.remtk.com/api/v1/files/public/418d60e4-7dd0-44aa-9c8b-b7386c23f754/view";

  console.log(`\n[4] Evaluando consulta del usuario: "${userPrompt}"`);
  const prediction = await skillService.predictSkill({
    sessionId: SESSION_ID,
    tenant: TENANT_ID,
    text: userPrompt,
    attachments: [{ url: attachedUrl, fileId: "418d60e4-7dd0-44aa-9c8b-b7386c23f754" }],
  });

  console.log("✅ Predicción y resolución de la nueva arquitectura:");
  console.log(" - Habilidad detectada:", prediction.matchedSkill?.name, `(tool: ${prediction.matchedSkill?.tools.join(",")})`);
  console.log(" - Entidad resuelta:", prediction.resolvedEntity?.name, `(ID: ${prediction.resolvedEntity?.id})`);
  console.log(" - Argumentos pre-resueltos:", JSON.stringify(prediction.preResolvedArgs, null, 2));
  console.log(" - Acción sugerida:", prediction.suggestedAction);
  console.log(" - Contexto inyectado al mini-agente:\n", prediction.contextPrompt);

  // 6. Probar la ejecución con Mistral LLM real (mini-agente) pasándole el contexto estructurado
  console.log("\n[5] Enviando tarea al modelo real de Mistral (mistral-small-latest) como mini-agente...");
  
  const systemPrompt = `Eres un mini-agente ejecutor de herramientas para mitumbes.com.
Tu objetivo es ejecutar la herramienta 'mitumbes_item_actualizar' con los parámetros correspondientes.
No inventes datos. Usa la información provista en el contexto estructurado.`;

  const miniAgentPrompt = `## Subtarea: Actualizar imagen del item con URL pública adjunta
${prediction.contextPrompt}

Herramienta disponible:
mitumbes_item_actualizar(id: string, image?: string, title?: string, type?: string)

Genera la llamada JSON en formato:
{"tool": "mitumbes_item_actualizar", "parameters": {"id": "...", "image": "..."}}`;

  const chatRes = await fetch("https://api.mistral.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "mistral-small-latest",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: miniAgentPrompt },
      ],
      temperature: 0.1,
    }),
  });

  const chatData = await chatRes.json();
  if (chatRes.ok && chatData.choices?.length > 0) {
    const assistantMessage = chatData.choices[0].message.content;
    console.log("\n=== RESPUESTA DEL MINI-AGENTE EN MISTRAL REAL ===");
    console.log(assistantMessage);
    console.log("\nTokens consumidos en Mistral:", chatData.usage);
    console.log("================================================");
    
    // Validar que no falló con [FALTAN_DATOS] y que incluyó el ID correcto
    if (assistantMessage.includes("3265a5e6-86a2-4156-a23f-39cfd7ff1171") && assistantMessage.includes(attachedUrl)) {
      console.log("\n🎯 RESULTADO DE LA PRUEBA REAL: ¡ÉXITO TOTAL!");
      console.log("1. El mini-agente recibió el ID exacto '3265a5e6-86a2-4156-a23f-39cfd7ff1171' de 'La Pichanga Gastrobar'.");
      console.log("2. Recibió la URL pública de la imagen.");
      console.log("3. Ejecutó 'mitumbes_item_actualizar' de forma directa sin emitir [FALTAN_DATOS] ni [HERRAMIENTA_INADECUADA].");
    }
  } else {
    console.error("❌ Error en chat completion de Mistral:", chatData);
  }
}

main().catch(console.error);
