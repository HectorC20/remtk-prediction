# Diagnóstico Forense de Predicción y Enrutamiento: Fallos en Solicitudes de Cron Jobs (`schedule_task` / `schedule_list`)

**Sesión analizada:** `df08bb63-cf1e-41c1-9a2b-8c68100ed00c`  
**Tenant (`userId`):** `4f4e8014-0a50-42ad-b1db-d6e4ffd4f9c8`  
**Catálogo activo del Tenant:** 103 herramientas registradas en memoria (`remtk-prediction`)  
**Fecha del incidente:** 2026-10-25 (20:32 UTC – 20:37 UTC)

---

## 1. Resumen Ejecutivo

Durante la sesión `df08bb63-cf1e-41c1-9a2b-8c68100ed00c`, el usuario interactuó con el Agente General (`actorType: "owner"`, negocio *Tiki's Rooftop*) para configurar y consultar alertas programadas cada 10 minutos sobre descripciones temáticas de Halloween. Se produjeron dos fallos críticos donde el agente **no ejecutó ninguna herramienta real (`toolExecutions: []`) y alucinó tanto la creación como el listado de los cron jobs**:

1. **Caso A (`"solo quiero que crees esa alerta de cada 10 min, solo eso quiero probar"` — 20:36:30 UTC):**
   - **Resultado observado:** `situation: "SINGLE_ACTION"`, `toolsCount: 1`, el planificador seleccionó `publicar_en_pagina` en lugar de `schedule_task`. En el bucle de ejecución (`HarnessLoopV8Service`), el LLM rechazó invocar `publicar_en_pagina` porque el usuario había pedido una alerta interna y no publicar en Facebook, terminando con `toolExecutions: []` y alucinando una tabla de confirmación.
   - **Causa raíz técnica:** En la Fase 1 de Recall Léxico (`ToolRegistry::retrieve_candidates`) y Fase 7 de Fusión Semántica + Afinidad de Nombre (`PredictionService::predict`), `schedule_task` obtuvo un score de **`0.83359313` (puesto #16 de 20)**, quedando **por debajo del umbral mínimo adaptativo (`adaptive_min_score = 0.84`)**. Por su parte, las 8 herramientas ganadoras (`publicar_en_pagina` con `0.8585`, `ask_questions`, `responder_comentario_publico`, `search_memory`, `search_knowledge_base`, `interest_status`, `web_fetch`, `schedule_update`) superaron `0.8502`. Como `schedule_task` fue descartada por el filtro adaptativo y nunca llegó a las 8 herramientas ofrecidas al `PlannerV8Service`, el planificador eligió `publicar_en_pagina` por contaminación semántica del contexto histórico (evento de Halloween de *Tiki's Rooftop*).

2. **Caso B (`"que cron jobs hay actualmente"` — 20:37:19 UTC):**
   - **Resultado observado:** `situation: "DIRECT_CHAT"`, `toolsCount: 0`, `modo: "razonar"`, `requiereHerramienta: false`. El LLM respondió listando 2 cron jobs (incluyendo la alerta de Halloween inexistente del turno anterior) leyendo el historial conversacional sin consultar la base de datos.
   - **Causa raíz técnica:** Aunque `schedule_list` sí entró en el Recall Léxico de Fase 1 por el término `"cron"`, su similitud coseno en `multilingual-e5-small` más `name_boost` fue de **`0.83726543` (puesto #4 de 20)**, quedando a tan solo **`0.0027` puntos por debajo del piso duro `adaptive_min_score = 0.84`**. Solo 2 herramientas superaron `0.84`: `schedule_update` (`0.84357864`) y `agenda_get_today_summary` (`0.8420608`). Al recibir únicamente `[schedule_update, agenda_get_today_summary]` y ver que ninguna servía para listar cron jobs (y además los dos cron jobs ya aparecían detallados en tablas Markdown en los mensajes previos del historial), `PlannerV8Service` clasificó la tarea como `modo: "razonar"` (`hasToolTasks = false`), vaciando `candidateTools = []` y forzando `situation = "DIRECT_CHAT"`.

---

## 2. Reproducción Exacta en el Motor `remtk-prediction` (Puerto 6876)

Al evaluar ambos prompts directamente contra el contenedor activo `remtk_prediction_rust` (`POST http://localhost:6876/predict`) para el tenant `4f4e8014-0a50-42ad-b1db-d6e4ffd4f9c8`, se obtienen las distribuciones matemáticas exactas que explican el comportamiento extremo a extremo.

### 2.1. Comparativa de Scores: ¿Por qué `"tienes cron jobs?"` (20:12 UTC) funcionó y los Casos A y B fallaron?

| Turno (UTC) | Prompt del Usuario | Score de `schedule_task` | Score de `schedule_list` | Umbral (`adaptive_min_score`) | Herramientas Ganadoras (`winning_tools`) |
|---|---|---|---|---|---|
| **20:12:05** (Éxito) | `"tienes cron jobs?"` | `0.84879` (#10) | **`0.86531` (#2)** ✅ | `0.8400` | `["schedule_update", "schedule_list", "workspace_replace_in_file", ...]` (8 tools) |
| **20:36:30** (Caso A) | `"solo quiero que crees esa alerta de cada 10 min, solo eso quiero probar"` | **`0.83359` (#16)** ❌ | *(Fuera de Recall Top-20)* | `0.8400` | `["publicar_en_pagina", "ask_questions", "responder_comentario_publico", "search_memory", "search_knowledge_base", "interest_status", "web_fetch", "schedule_update"]` (8 tools) |
| **20:37:19** (Caso B) | `"que cron jobs hay actualmente"` | `0.83281` (#5) ❌ | **`0.83727` (#4)** ❌ | `0.8400` | `["schedule_update", "agenda_get_today_summary"]` (2 tools) |

---

## 3. Análisis Técnico Profundo — Caso A

**Prompt:** `"solo quiero que crees esa alerta de cada 10 min, solo eso quiero probar"`

### 3.1. Fase 1: Recall Léxico en `ToolRegistry::retrieve_candidates` (`src/support/registry/tool_cache.rs:70-115`)

El método `retrieve_candidates` tokeniza la consulta filtrando términos de longitud `> 2`:
- Términos extraídos: `["solo", "quiero", "que", "crees", "esa", "alerta", "cada", "min", "solo", "eso", "quiero", "probar"]`.
- Puntuación léxica por herramienta:
  - `+3.0` si `metadata.name` contiene el término.
  - `+2.0` si `metadata.intent_summary` contiene el término.
  - `+1.0` si `metadata.description` contiene el término.
- **Brecha en `schedule_task` (`scheduler.mcp.ts:13-34`):**
  - `name`: `"schedule_task"` (0 coincidencias).
  - `intentSummary`: `"Programa una tarea del agente (única o recurrente con cron): recordatorios, seguimientos, reportes y delegaciones autónomas"` (0 coincidencias: no contiene `"alerta"`, `"cada"`, ni `"min"`).
  - `description`: Contiene `"cada"` (`"cada día/lunes..."`), pero **no contiene la palabra `"alerta"`** (la palabra `"Alerta"` está dentro de `inputSchema.properties.type.description`, pero **`ToolDefinitionDto` no indexa `inputSchema`** al registrar herramientas en `upsert_tools`).
- Como resultado, `schedule_task` apenas suma puntos por términos genéricos (`"que"`, `"cada"`, `"solo"`), entrando en la cola del Top-20 de recall pero sin ventaja frente a herramientas de negocio y comunicación con descripciones largas en español (`publicar_en_pagina`, `responder_comentario_publico`, `interest_status`).

### 3.2. Fase 6 y 7: Desconexión del Cabezal MoE y Puntuación E5 (`src/core/prediction/service.rs:115-158`)

Al inspeccionar `src/core/prediction/service.rs:115-158`:
1. En la línea 116 se ejecuta `let moe_output = self.engine.evaluate_moe(...)`, que calcula `moe_output.ranked_scores` mediante el modelo neuronal `remtk-juicio-moe-int8`.
2. **Sin embargo, en las líneas 124-155, `moe_output.ranked_scores` NO se utiliza para calcular el `final_score` de cada herramienta candidata.** En su lugar, el código calcula únicamente:
   ```rust
   let sem_sim = cosine_similarity_normalized(&query_emb, &tool.embedding);
   // + name_boost si el nombre de la herramienta contiene palabras del prompt (> 3 chars)
   let final_score = (sem_sim + name_boost).clamp(0.0, 1.0);
   ```
3. Como el prompt `"solo quiero que crees esa alerta de cada 10 min, solo eso quiero probar"` no contiene ninguna subcadena de `"schedule_task"` (`name_boost = 0.0`) y está formulado con lenguaje coloquial/anafórico (`"esa alerta"`, `"solo eso quiero probar"`), su similitud coseno `e5-small` contra el pasaje canónico de `schedule_task` cae a **`0.83359313`**.
4. En cambio, herramientas cortas del catálogo de negocio como `publicar_en_pagina` (`0.85853356`) y `ask_questions` (`0.8577933`) presentan una mayor densidad de solapamiento en la variedad estrecha de `multilingual-e5-small` (donde todos los cosenos se comprimen en la banda `[0.80, 0.87]`).

### 3.3. Fase 8: Corte en `adaptive_threshold` (`src/core/prediction/service.rs:202-242`)

El método `adaptive_threshold` aplica:
- `min_score = 0.84` (`ONNX_ADAPTIVE_MIN_SCORE` en `src/shared/config/mod.rs:71`).
- `max_tools = 8` (`MAX_OUTPUT_TOOLS` en `src/shared/config/mod.rs:80`).

El ranking completo de las 20 candidatas en el Caso A fue:

| Posición | Herramienta | Score Final | Estado tras `adaptive_threshold` |
|---|---|---|---|
| #1 | `publicar_en_pagina` | `0.85853356` | ✅ Seleccionada (Top-8) |
| #2 | `ask_questions` | `0.85779330` | ✅ Seleccionada (Top-8) |
| #3 | `responder_comentario_publico` | `0.85709230` | ✅ Seleccionada (Top-8) |
| #4 | `search_memory` | `0.85612100` | ✅ Seleccionada (Top-8) |
| #5 | `search_knowledge_base` | `0.85488045` | ✅ Seleccionada (Top-8) |
| #6 | `interest_status` | `0.85464364` | ✅ Seleccionada (Top-8) |
| #7 | `web_fetch` | `0.85131540` | ✅ Seleccionada (Top-8) |
| #8 | `schedule_update` | `0.85025424` | ✅ Seleccionada (Top-8, corte `max_output_tools = 8`) |
| #9 | `process_order_request_review` | `0.85022730` | ❌ Descartada por tope `max_output_tools = 8` |
| ... | ... | ... | ... |
| **#16** | **`schedule_task`** | **`0.83359313`** | ❌ **Descartada (`score < 0.84` y fuera de Top-8)** |

### 3.4. Contaminación del Planificador y Alucinación Final en `key-remtk`

1. `AimcpServiceV8.processPrompt` (`aimcp-v8.service.ts:586`) pasa las 8 herramientas ganadoras a `PlannerV8Service.generatePlan`.
2. El planificador ve la conversación sobre las descripciones del evento de Halloween de *Tiki's Rooftop* y observa que **no existe `schedule_task`** en `HERRAMIENTAS RELEVANTES CANDIDATAS`, pero sí están `publicar_en_pagina` y `schedule_update`. Como se trata de una creación (`"crees esa alerta"`), descarta `schedule_update` (cuya descripción prohíbe explícitamente usarla para crear tareas nuevas: *"NO la uses para programar una tarea nueva"*) y asigna erróneamente:
   - `modo: "ejecutar"`, `herramienta: "publicar_en_pagina"`, `objetivo: "Crear una alerta/publicación recurrente cada 10 minutos en la Página de Facebook..."`.
3. Debido a la regla de reconciliación en `aimcp-v8.service.ts:650-656`:
   ```ts
   const hasUnresolvedOperationalTask = (plan?.tasks || []).some(
     (t) => t.requiresTool && t.modo !== "programar" && !t.tool,
   );
   ```
   Como `t.tool` **sí tenía un valor asignado** (`"publicar_en_pagina"`), **`hasUnresolvedOperationalTask` fue `false`** y `candidateTools.length` era `8` (`!== 0`), por lo que **no se disparó la segunda predicción anafórica (`groundedSelection`)** con el `plan.objetivo`.
4. En `HarnessLoopV8Service.buildStepTools` (`harness-loop-v8.service.ts:1999-2024`), al ser `situation === "SINGLE_ACTION"` y tener `planRequired = ["publicar_en_pagina"]`, el presupuesto de herramientas secundarias se redujo a `0` (`toolsCount: 1`).
5. En el Paso 1 del bucle (`THINKING`), el LLM ejecutor recibió únicamente la herramienta `publicar_en_pagina` (cuya descripción exige: *"Usar solo si fbAutoPublishEnabled esta activo y el usuario lo solicita explicitamente"*). Al notar que el usuario solo quería un recordatorio interno y no publicar en Facebook, el LLM decidió **no invocar `publicar_en_pagina`** (`toolCalls = []`) y redactó en texto plano la tabla simulando haber creado la alerta e incluyendo incluso una advertencia: *"esta alerta es un recordatorio interno... No publica automáticamente en la Página de Facebook"*.

---

## 4. Análisis Técnico Profundo — Caso B

**Prompt:** `"que cron jobs hay actualmente"`

### 4.1. Fase 1 y Fase 7: Por qué `schedule_list` quedó en `0.83726543` (< `0.84`)

1. **Recall Léxico (Fase 1):**
   - Términos de longitud `> 2`: `["que", "cron", "jobs", "hay", "actualmente"]`.
   - El término `"cron"` coincide en `schedule_update`, `schedule_list`, `schedule_task`, `schedule_cancel`, `schedule_delete`.
   - Sin embargo, el anglicismo compuesto **`"cron jobs"`** no figura como unidad fraseológica en `name`, `intentSummary`, `description` ni `tags` de `schedule_list` (`scheduler.mcp.ts:89-98`), donde solo se emplea `"tareas programadas"`, `"recordatorios"` y `"cron"`. La palabra `"jobs"` tiene `0` coincidencias en todo el catálogo del scheduler.
2. **Similitud Coseno E5 + Afinidad de Nombre (Fase 7):**
   - El pasaje canónico indexado en `tool_cache.rs:155-163` para `schedule_list` es:
     ```text
     passage: schedule_list | scheduler / runtime | Lista las tareas programadas del agente (recordatorios, seguimientos, reportes, delegaciones) | Lista las tareas programadas del usuario (recordatorios, seguimientos, reportes, tareas delegadas) con tipo, fecha de ejecución, cron y estado. Úsala cuando el usuario pregunte qué tareas tiene programadas, qué recordatorios hay agendados o qué hay pendiente. | tags: scheduler, cron, recordatorio, tareas programadas, listar tareas, agenda, estado, programado
     ```
   - Para la consulta `"query: que cron jobs hay actualmente"`:
     - Ningún término de más de 3 caracteres (`"cron"`, `"jobs"`, `"actualmente"`) es subcadena de `"schedule_list"`, por lo que **`name_boost = 0.0`**.
     - La similitud coseno pura entre `"query: que cron jobs hay actualmente"` y el pasaje de `schedule_list` es **`0.83726543`**.
     - Curiosamente, en el turno de las 20:12:05 UTC (`"tienes cron jobs?"`), la brevedad e interrogación directa produjeron un vector con similitud **`0.8653077`** (superando `0.84`), mientras que al añadir `"que ... hay actualmente"` el vector se desplazó ligeramente hacia consultas de estado general/agenda, haciendo que:
       - `schedule_update` obtuviera `0.84357864` (#1)
       - `agenda_get_today_summary` obtuviera `0.84206080` (#2)
       - `process_order_request_review` obtuviera `0.83923340` (#3)
       - **`schedule_list` obtuviera `0.83726543` (#4)**

### 4.2. Fase 8: Filtrado por Piso de Relevancia (`adaptive_min_score = 0.84`)

En `PredictionService::adaptive_threshold` (`src/core/prediction/service.rs:233-238`):
```rust
for i in 0..scored.len() {
    if scored[i].score < min_score || (top - scored[i].score) > band {
        cluster = i;
        break;
    }
}
```
Como `min_score = 0.8400`, el bucle corta exactamente en `i = 2` (`process_order_request_review` tiene `0.8392 < 0.84` y `schedule_list` tiene `0.8372 < 0.84`).
Por lo tanto, `remtk-prediction` devuelve únicamente **2 herramientas**:
```json
{
  "tools": ["schedule_update", "agenda_get_today_summary"]
}
```

### 4.3. Degradación a `DIRECT_CHAT` en `PlannerV8Service` y `AimcpServiceV8`

1. En `AimcpServiceV8.processPrompt`, el régimen de historial (`POST /history/regime`) detectó continuidad contextual y devolvió `PRESERVE_ALL` (los últimos 16 mensajes).
2. `PlannerV8Service.generatePlan` recibió:
   -Los últimos 6 mensajes del historial (`planner-v8.service.ts:102`), que contenían **dos tablas Markdown completas**: la tarea programada de las 8:00 AM (mostrada a las 20:12) y la supuesta "Alerta Halloween cada 10 min" (alucinada por el asistente 45 segundos antes, a las 20:36:35).
   - `HERRAMIENTAS RELEVANTES CANDIDATAS`: únicamente `schedule_update` y `agenda_get_today_summary`.
   - La Regla 5 y Regla 6 del prompt del planificador (`planner-v8.service.ts:38-44`):
     > *"5. CONVERSACIÓN PREVIA, DIÁLOGO Y GRÁFICOS (MERMAID): Las preguntas sobre la propia conversación, el diálogo previo... se resuelven DIRECTAMENTE en modo 'razonar'..."*
     > *"6. ...Si la tarea es de síntesis, explicación, análisis... de datos que ya están presentes en el diálogo, el modo DEBE ser 'razonar' y NO debe asignarse ninguna herramienta externa, ignorando las candidatas operativas no pertinentes."*
3. Al no tener `schedule_list` en las candidatas y ver que ambas tareas ya figuraban en el diálogo reciente, el planificador emitió:
   ```json
   {
     "objetivo": "Consultar los cron jobs (tareas programadas) actualmente existentes del usuario y del agente asignado",
     "tareas": [
       {
         "key": "task-0",
         "label": "Listar los cron jobs actuales del usuario y del agente",
         "modo": "razonar",
         "requiereHerramienta": false
       }
     ]
   }
   ```
4. En `aimcp-v8.service.ts:634`:
   ```ts
   if (!hasToolTasks) {
     dagSelection = {
       prediction: initialDagSelection.prediction,
       candidateTools: [],
       requiredActivations: [],
       skillContext: initialDagSelection.skillContext,
     };
   }
   ```
   Como `hasToolTasks` fue `false`, `candidateTools` se vació a `[]`.
5. `HarnessPromptV8Builder.detectSituation` recibió `candidateToolsCount: 0` y clasificó la situación como **`DIRECT_CHAT`** (`toolsCount: 0`).
6. Finalmente, el LLM respondió en modo conversacional sintetizando las dos tareas presentes en el historial de chat (la real de las 8:00 AM y la alucinada de Halloween).

---

## 5. Resumen de Hallazgos Arquitectónicos (Alineados con `AGENT.md`)

De acuerdo con la directiva `AGENT.md` (*Prohibición Estricta de Parches Léxicos* y *Protocolo ante Fallos de Predicción §3*), estos dos casos evidencian cuatro puntos estructurales en el pipeline actual:

1. **Desconexión de `moe_output.ranked_scores` en `PredictionService::predict` (`src/core/prediction/service.rs:116-155`):**
   - El motor ejecuta la inferencia pesada del cabezal `remtk-juicio-moe` (~105–153 ms, que consume `query_tokens`, `context_tokens` y `tools_tensor`), pero ignora su vector de salida `moe_output.ranked_scores` al construir el vector `ranked`, usando únicamente `cosine_similarity_normalized(&query_emb, &tool.embedding) + name_boost`.
2. **Compresión de Banda de `multilingual-e5-small` vs. Piso Rígido `adaptive_min_score = 0.84`:**
   - En catálogos de más de 100 herramientas, las similitudes coseno de `e5-small` sin `name_boost` para consultas coloquiales o con anglicismos (`"que cron jobs hay actualmente"`) se sitúan entre `0.832` y `0.843`. Un piso estático en `0.84` corta herramientas legítimas ubicadas en el Top-4 (`schedule_list` con `0.8372`).
3. **Ausencia de Clausura de Familia en el Dominio `scheduler` (`key-remtk` / Grafo DAG):**
   - Mientras que `ToolCatalogV8Service.predictAndResolveDagTools` aplica clausura estructural de hermanos para el dominio `knowledge` (`knowledge_add`, `knowledge_list`, etc.) y `document` (`document_generate` ↔ `document_convert`), cuando el predictor selecciona `schedule_update` (que fue Top-8 en el Caso A con `0.8502` y Top-1 en el Caso B con `0.8435`), no se incluyen sus herramientas hermanas del grupo `scheduler` (`schedule_task`, `schedule_list`).
4. **Representación Semántica en `scheduler.mcp.ts`:**
   - `schedule_list` no incluye en su descripción ni en `tags` la expresión habitual `"cron jobs"` / `"jobs programados"` / `"alertas activas"`, y `schedule_task` reserva la palabra `"Alerta"` únicamente dentro de un subcampo de `inputSchema` que no forma parte del texto canónico indexado en `ToolRegistry::upsert_tools`.
