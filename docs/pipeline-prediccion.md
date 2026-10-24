# Pipeline de predicción

## Predicción de tools (`POST /predict`)

Flujo completo orquestado por `PredictionOrchestrator.predict()` en
`src/predict/orchestrator.ts`. Devuelve un `GraphPredictionResult`: el resultado
clásico (`tools`, `complexity`, `modelSize`, `rankedScores`) más un bloque
`graph` con `nodes`, `edges` y `executionOrder`.

```
 text (user) + sessionId + tenant (+ history, priorPlan)
      │
      ▼
 [1] TurnClassifier.classify(text) ──► new_query | confirmation_empty | confirmation_nuance
      │
      ▼  (turn = confirmation_empty)
 [2] ¿Plan cacheado en ConfirmationCache[sessionId]?  ──SÍ──► EARLY-EXIT: devuelve el plan
      │ NO                                                       (confirmHit = true)
      ▼
 [3] Construir promptText
      · new_query ──────────────► text + últimos 4 mensajes de history (≤300 chars c/u)
      · confirmation_* / agent ─► priorPlan ?? text
      │
      ▼
 [4] Estado latente de sesión: z_t = SessionStateCache.resolve(sessionId, promptText)
      · embed query: del prompt (prioridad alta)
      · cosine(z_{t-1}, z_new) ≥ 0.86 ──► z_t = L2(0.7·z_{t-1} + 0.3·z_new)
      · cosine < 0.86 o sin estado previo ──► z_t = z_new (topicShift = true)
      │
      ▼
 [5] ¿Existe grafo del tenant (graphCache.get) y z_t disponible?
      │ SÍ                                          │ NO
      ▼                                            ▼
 [6A] RUTA TOPOLÓGICA (Graph Router)         [6B] FALLBACK PLANO (3 capas)
      rerank.graphFilter(zt, graph, edges,   keywords.reduce(tenant, prompt, catálogo)
      lexicalScores, catálogo)                · embed query: de keywords del prompt
      · s_i = cosine(z_t, E(T_i))             · embed passage: de keywords de cada tool
      · s_i += keywordBoost × BM25_norm       · top-K (default 20) por cosine
      · propagación pre-requisitos:           si falla/vacío ──► catálogo completo
        S = s + α·(Aᵀ·s), α = 0.2            rerank.filter(...): fusión BM25 + umbral
      · piso de relevancia · umbral
        adaptativo · resolución de mutexes
      · orden topológico (Kahn)
      │                                            │
      ▼                                            ▼
      { tools[], complexity, modelSize, rankedScores, graph{nodes,edges,executionOrder} }
      │
      ▼
 [7] new_query ──► guarda resultado en ConfirmationCache[sessionId]
      │
      ▼
      (+ trace en Debugger)
```

### 1 — Clasificador de turno (`TurnClassifier`)

Compara el embedding del texto con dos **arquetipos** embebidos con `query:`
(constantes en `shared/constants/messages/predict.constant.ts`):

- `"confirma y continúa con el plan previo"` (confirmación)
- `"nueva solicitud"` (consulta)

Regla de decisión (con `confirmScore` y `queryScore`), devuelve un valor del enum
`TurnType` (`shared/dictionary/turn.dictionary.ts`):

```
confirm < query + 0.03          → new_query
confirm < 0.55                  → confirmation_nuance
de lo contrario                 → confirmation_empty
```

### 2 — Early-exit por confirmación

Si el turno es `confirmation_empty` y existe un **plan cacheado** para la sesión
(seteado por la última `new_query`), devuelve directamente ese
`GraphPredictionResult` sin recomputar embeddings (ahorro de CPU/latencia).

### 3 — Contexto del prompt

- En `new_query` se incorporan los últimos **4 mensajes** de `history` (cada uno
  recortado a 300 caracteres, `MAX_HISTORY_MESSAGES`/`MAX_MESSAGE_CHARS`) y se
  concatenan con el texto actual separados por salto de línea. Sirve para
  resolver referencias del tipo "sí, hazlo" dando contexto a keywords y recall.
- En confirmaciones o turnos con `source: "agent"` se predice sobre `priorPlan`.

### 4 — Estado latente de sesión (`SessionStateCacheService`)

Por cada `sessionId` se mantiene un vector de estado `z_t` (384 dims):

- Se embebe el prompt con prefijo `query:` (prioridad **alta**, ver cola en
  [embeddings.md](embeddings.md)).
- Si no hay estado previo o el coseno con el anterior es
  `< TOPIC_SHIFT_THRESHOLD (0.86)`, hay **cambio de tema** y el estado se resetea
  a `z_new` (`topicShift = true`).
- Si el coseno es `≥ 0.86` (mismo tema), se aplica la combinación convexa
  `z_t = L2(0.7·z_{t-1} + 0.3·z_new)`.

Este `z_t` es el "espacio latente" contra el que se puntúan los nodos del grafo.

### 5 y 6A — Ruta topológica (Graph Router)

Se usa cuando el tenant tiene **grafo precomputado** (se construye en
`POST /tools`) y se obtuvo `z_t`. Implementada en `RerankService.graphFilter`
(`src/predict/services/rerank.service.ts`).

1. **Similitud base**: por cada nodo del grafo,
   `s_i = cosine(z_t, E(T_i))` donde `E(T_i)` es el embedding `passage:` de la
   firma de la tool, precacheado en el grafo.
2. **Confirmación léxica**: se suma `keywordBoost (0.15) × score BM25
   normalizado` de Qdrant (ver paso 5 del fallback plano).
3. **Propagación de pre-requisitos** por la matriz de adyacencia A (N×N):
   `S_i = s_i + α·Σⱼ A[j][i]·s_j` con `α = graphPropagationAlphaDefault = 0.2`.
   Los pre-requisitos (origen de la arista `PREREQUISITE`) transfieren masa a sus
   dependientes sin distorsionar el ranking principal.
4. **Piso de relevancia**: si el mejor score propagado no supera
   `adaptiveMinScore` (0.8) → 0 tools.
5. **Umbral adaptativo** sobre los scores propagados + recorte a
   `MAX_OUTPUT_TOOLS`.
6. **Resolución de mutexes**: si dos nodos seleccionados comparten una arista
   `MUTUALLY_EXCLUSIVE`, se elimina el de menor score propagado; si la poda deja
   el conjunto por debajo del mínimo adaptativo, se rellena con la siguiente
   mejor tool no conflictiva.
7. **Orden topológico** (algoritmo de Kahn) usando solo las aristas
   `PREREQUISITE` del subgrafo activado (empates y ciclos se desempatan por mayor
   score). La respuesta ordena `tools` y `graph.executionOrder` según ese orden.

Si el estado de sesión falla o el tenant no tiene grafo, se degrada al fallback
plano (abajo), que produce `graph = { nodes: orden, edges: [], executionOrder: orden }`.

### 6B — Fallback plano: capas 1+2 (keywords + match cross-idioma)

`KeywordService.reduce` es la reducción del catálogo cuando no hay grafo. Detalle
en [embeddings.md](embeddings.md).

1. `extractQueryKeywords(prompt)`: normaliza (minúsculas + quita diacríticos),
   descarta stopwords ES/EN y tokens ≤ 2 chars, ordena por frecuencia y luego por
   longitud.
2. Embed de la query con prefijo `query:` (keywords unidas o prompt crudo).
3. Por cada tool del catálogo del tenant: embedding `passage:` de sus **keywords
   canónicas** (`tags` + palabras de `intentSummary`), **cacheado** por sha1 del
   documento.
4. `score = cosine(query, keywords_tool)`; se ordena y se quedan las **top-K**
   (`keywordTopK`, default 20).

Notas:
- El modelo `multilingual-e5-small` mapea términos de idiomas distintos a vectores
  cercanos, así el match "correo ↔ email" funciona **sin traducción explícita**.
- En plena predicción los embeddings se computan con **prioridad alta**; el
  registro masivo usa prioridad baja.
- Si el embed degrada a hash o Qdrant no está, sigue funcionando con scores más débiles.

### 5 — Confirmación léxica (BM25)

En ambas rutas, `QdrantService.retrieveTools()` hace el recall léxico (sinónimos
+ RAKE + BM25, blend 60/40; ver [qdrant.md](qdrant.md)). Cada score se normaliza
por el máximo del ranking (0..1) para que sea comparable al coseno. Errores de
Qdrant no rompen el flujo: el mapa de scores queda vacío.

### Umbral adaptativo y complejidad (`RerankService` / utilidades)

- **Piso de relevancia:** si el mejor coseno (`topCos`) es menor que
  `adaptiveMinScore` (default **0.8**), no se devuelve ninguna tool
  (`rankedScores = [topCos]`).
- **Umbral adaptativo** (`adaptiveThreshold`): con el ranking ordenado desc:
  1. Garantiza al menos `adaptiveMinTools` (default **2**).
  2. Recorre posiciones hasta `adaptiveMaxTools` (default **30**) buscando el
     primer **gap natural** `Δ > adaptiveGapThreshold` (default **0.03**); si lo
     encuentra, corta ahí.
  3. Sin gap natural (ranking denso, prompts multi-paso) devuelve el **cluster
     superior** (tools dentro de `3×gap` del top), acotado al mínimo y al máximo,
     en vez de colapsar a 1–2 tools.
- Tope final opcional `maxOutputTools` (env `MAX_OUTPUT_TOOLS`, 0–50) recorta el
  resultado del umbral.
- **Complejidad estimada** (`estimateComplexity`):

| Condición | `complexity` |
|---|---|
| tools seleccionadas > 20 y promedio del ranking > 0.65 | `complex` |
| tools seleccionadas > 10 y promedio > 0.55 | `moderate` |
| resto | `simple` |

El resultado incluye `modelSize` = `"small"` (ONNX) u `"hash"` (fallback).

### 7 — Caché del plan

En `new_query` el resultado se guarda en `ConfirmationCache` por `sessionId` para
soportar el early-exit del paso 2 en turnos siguientes.

### Trazas

Cada predicción genera una `Trace` (sessionId, tenant, turnType, confirmHit,
recall, degraded, latencia, modelo, embeddings recomputados/cacheados,
complejidad…) que alimenta `Debugger` (endpoint `GET /debug`, anillo de 200,
snapshot de las últimas 20 invertidas).

## Pipeline de memorias (`POST /memory/predict`)

Flujo de `PredictionOrchestrator.predictMemory()`.

```
 text (+ sessionId, tenant como userId, limit)
      │
      ▼
 [1] Embed del prompt (prefijo query:)
      │
      ▼
 [2] Topic-shift: cosine contra el embedding previo de la sesión
      topicScore < 0.86 ──► topicShift = true (resetea estado)
      Estado por sesión: primera vez → guarda; shift → reinicia;
      sin shift → blend 70/30 + normalización L2
      │
      ▼
 [3] Recall Qdrant: searchMemories(tenant, text, limit)   (BM25 + filtro userId)
      fallo ──► candidatas = []
      │
      ▼
 [4] Re-rank por cosine: embed passage: del contenido de cada memoria
      (cacheado en memoria por contenido) y cosine contra el prompt
      │
      ▼
      { memories[], topicShift, topicScore, modelSize, rankedScores }
```

Detalles relevantes:

- `topicShift` indica que el usuario cambió de tema respecto al turno anterior.
- El `limit` default es **8**.
- El estado de tópico (`topics`) y el caché de embeddings de memorias
  (`memoryCache`) viven en memoria del orquestador.
- Igual que en tools, la degradación es suave: sin Qdrant devuelve listas vacías
  sin errores.

## Registro de tools (`POST /tools`)

`Orchestrator.registerTools(tenant, tools)`:

1. `qdrant.indexTools(tenant, tools)` — actualiza el **catálogo en memoria** del
   tenant (fuente de definiciones) e indexa points BM25 en Qdrant si está
   habilitado.
2. `keywords.upsertTools(tenant, tools)` — precomputa (prioridad **baja**) los
   embeddings `passage:` de las keywords de cada tool; se salta las que no
   cambiaron (hash sha1 del documento de keywords). Devuelve contadores
   recomputed/cached.
3. `graphCache.buildTenantGraph(tenant, tools)` — **construye el grafo**: un nodo
   por tool con el embedding `passage:` de su firma (prioridad baja) y las
   aristas inferidas (ver abajo), con su matriz de adyacencia.
4. Actualiza contadores de `Debugger` y devuelve `{ indexed: tools.length }`.

Esto hace que en la primera consulta el pipeline ya tenga embeddings y grafo, sin
recomputar en caliente.

### Construcción del grafo (`ToolGraphCacheService`)

- **Nodo**: `E(T_i) = L2(ONNX("passage: " + firma))` donde la firma es
  `name group category description tags… intentSummary {inputSchema}`.
- **Aristas inferidas** (`inferEdges`) con pesos fijos:
  - `PREREQUISITE` (0.85): relaciones declaradas en `prerequisites` + inferencia
    ligera: si una clave del `inputSchema` de una tool coincide (token
    normalizado) con el nombre o grupo de otra, esa otra es su pre-requisito.
  - `MUTUALLY_EXCLUSIVE` (1.0): relaciones declaradas en `conflicts`
    (simétricas, se guarda una sola arista con `from < to`).
  - `CO_OCCURRENCE` (0.4): pares de tools del mismo `group`.
- Matriz de adyacencia A aplanada N×N con `A[i][j]` = peso de `i → j` (i
  pre-requisito de j). `versionHash` (sha1 de firmas y relaciones) permite
  detectar cambios de catálogo.
