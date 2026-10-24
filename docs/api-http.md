# API HTTP

El proceso arranca **tres apps Nest autocontenidas** (`src/server.ts`), cada una
en su puerto, con contratos 1:1 respecto al server Go original:

| Puerto | App Nest | Controlador | Propósito |
|---|---|---|---|
| `6776` | `PredictAppModule` | `PredictV1Controller` | Predicción de tools MCP |
| `6777` | `EmbeddingAppModule` | `EmbeddingV1Controller` | Embeddings (e5-small) |
| `6775` | `QdrantAppModule` | `QdrantV1Controller` | API propia sobre el motor Qdrant |

Todas responden `{"status":"ok"}` en `/health`. Las tres montan el parser
`express.json({ limit: "4mb" })` (constante `LIMIT` en
`shared/constants/predict/data.predict.ts`, aplicado por `startHttp` en
`app.module.ts`). No hay prefijo global: cada app expone los paths raíz del
listener Express original.

## Puerto 6776 — Predicción (`PredictV1Controller`)

### `POST /predict`

Predice las herramientas MCP más relevantes para el texto. Devuelve
`GraphPredictionResult`: además de `tools`/`complexity`/`modelSize`/
`rankedScores` incluye el subgrafo activado (`graph`).

```bash
curl -X POST http://localhost:6776/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "chat-123",
    "tenant": "tenant-demo",
    "text": "envía un correo electrónico al equipo con el resumen",
    "source": "human",
    "history": [{"role": "user", "content": "hola"}],
    "priorPlan": "{\"objetivo\":\"...\"}"
  }'
```

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `sessionId` | string | sí | Identificador de conversación (caché de confirmación, tópico). |
| `tenant` | string | sí | Tenant cuyo catálogo se consulta. |
| `text` | string | sí | Texto del usuario. Vacío → `{tools:[],...}` sin pipeline. |
| `source` | `"human"` \| `"agent"` | no | Predice sobre `priorPlan` si es `agent`. |
| `history` | `ChatMessage[]` | no | Mensajes previos (solo turnos `new_query`, últimos 4). |
| `priorPlan` | string (JSON) | no | Plan previo para confirmaciones/agentes. |

**Respuesta 200** (ruta topológica — grafo del tenant + estado de sesión):

```json
{
  "tools": [
    {
      "id": "t1", "name": "send_email", "group": "communication",
      "category": "email", "description": "Envía un correo electrónico...",
      "tags": ["email"], "intentSummary": "Enviar correo", "inputSchema": {}
    }
  ],
  "complexity": "moderate",
  "modelSize": "small",
  "rankedScores": [0.91, 0.72],
  "graph": {
    "nodes": ["send_email", "log_email"],
    "edges": [
      { "from": "send_email", "to": "log_email", "type": "PREREQUISITE", "weight": 0.85 }
    ],
    "executionOrder": ["send_email", "log_email"]
  }
}
```

**Respuesta 200** (fallback plano — sin grafo ni estado de sesión):

```json
{
  "tools": [],
  "complexity": "simple",
  "modelSize": "hash",
  "rankedScores": [],
  "graph": { "nodes": [], "edges": [], "executionOrder": [] }
}
```

- `graph.nodes` = herramientas seleccionadas en orden de ejecución.
- `graph.edges` = aristas del subgrafo activado (ambos extremos seleccionados).
- `graph.executionOrder` = orden topológico (Kahn) por `PREREQUISITE`; igual a
  `nodes` salvo cuando hay mutexes o aristas no topológicas.
- En el fallback plano `edges` siempre es `[]` y `nodes`/`executionOrder` son el
  orden de salida del ranking.
- `modelSize`: `"small"` (ONNX) u `"hash"` (fallback determinístico).
- `rankedScores`: puntajes finales de las tools devueltas, en el mismo orden.
- Si el turno es confirmación vacía y hay plan cacheado → **early-exit** con el
  plan previo (sin recomputar embeddings).

**Errores:** `400` si faltan `sessionId`/`tenant`/`text`; `500` si falla el pipeline.

### `POST /tools`

Registra (indexa) las tools del tenant: indexa Qdrant, precomputa/cachea
embeddings de keywords y **construye el grafo topológico del tenant**
(`ToolGraphCacheService.buildTenantGraph`, embeddings de nodo sobre la firma de
cada tool). El grafo queda listo para que `/predict` use la ruta topológica.

```bash
curl -X POST http://localhost:6776/tools \
  -H "Content-Type: application/json" \
  -d '{
    "tenant": "tenant-demo",
    "tools": [{
      "id": "t1", "name": "send_email", "group": "communication",
      "category": "email", "description": "Envía un correo electrónico...",
      "tags": ["email"], "intentSummary": "Enviar correo",
      "inputSchema": {"to": "string"},
      "prerequisites": ["t0"],
      "conflicts": ["t9"]
    }]
  }'
```

`prerequisites`/`conflicts` son opcionales (`string[]` de nombres de tools):
declaran relaciones que el grafo refuerza/inferencia junto a las automáticas.

**Respuesta 200:** `{ "indexed": 12 }` · **Errores:** `400` sin `tenant`/`tools[]`, `500` interno.

### `GET /tools/count?tenant=tenant-demo`

```json
{ "count": 12 }
```

Cuenta el catálogo en memoria del tenant (no requiere Qdrant).

### `POST /memory/predict`

Recupera memorias contextuales del usuario (BM25 + re-rank por cosine) y detecta
cambio de tema. Independiente del grafo (sigue el pipeline plano de memoria).

```bash
curl -X POST http://localhost:6776/memory/predict \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"chat-123","tenant":"user-1","text":"qué hora es en Perú","limit":8}'
```

**Respuesta 200:**

```json
{
  "memories": [
    {
      "id": "m1", "content": "El usuario está en Perú (Piura)...",
      "chatId": "chat-123", "createdAt": "2026-08-29T22:19:00Z", "score": 0.88
    }
  ],
  "topicShift": false,
  "topicScore": 0.92,
  "modelSize": "small",
  "rankedScores": [0.88]
}
```

| Campo | Descripción |
|---|---|
| `topicShift` | `true` si el tema cambió respecto al turno previo (umbral 0.86). |
| `topicScore` | Coseno contra el estado de tópico de la sesión (0–1). |
| `limit` | Máximo de memorias (default `8`). |

**Errores:** `400` sin `sessionId`/`tenant`/`text`; `500` interno.

### `GET /debug`

Estado interno: motor de embeddings, estadísticas y trazas recientes.

```json
{
  "engine": { "size": "small", "modelPath": "../models/multilingual-e5-small-onnx" },
  "stats": {
    "tenants": 1,
    "confirmGets": 1, "confirmHits": 0, "confirmSets": 1,
    "embeddingsRecomputed": 12, "embeddingsCached": 2
  },
  "traces": [ { "sessionId": "...", "turnType": "new_query", "recall": 2, "latencyMs": 45 } ]
}
```

- `engine.size`: `"small"` (ONNX) u `"hash"` (fallback); `modelPath` vacío si el
  modelo ONNX no cargó.
- `traces` = últimas 20 trazas (orden inverso); internamente se conservan 200.

## Puerto 6777 — Modelos (`EmbeddingV1Controller`)

API del motor de embeddings. Sirve únicamente el tamaño `small` (e5-small, 384
dims); el `large` quedó desactivado en este proceso (solo uso in-process en
tests).

### `GET /models`

Estado del modelo por defecto del proceso (e5-small):

```json
{
  "small": { "model": "small", "dim": 384, "modelPath": "../models/multilingual-e5-small-onnx" }
}
```

- `model` = `"hash"` si el modelo ONNX no está disponible (carga fallida o inacabada).
- `modelPath` = carpeta del modelo cuando está listo; si no, `""`.

### `POST /embed`

Genera el embedding de un texto con el modelo `small` (e5-small). El campo
`size` del body se **ignora**: el API sirve siempre `small`. El caller debe
aplicar el prefijo `query: `/`passage: ` según el caso (contrato e5).

```bash
curl -X POST http://localhost:6777/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"query: enviar correo al equipo"}'
```

**Respuesta 200:**

```json
{
  "embedding": [0.0123, -0.0456, 0.0891, "..."],
  "model": "small",
  "dim": 384,
  "modelPath": "../models/multilingual-e5-small-onnx"
}
```

**Errores:** `400` si `text` no es string no vacío; `500` si la inferencia falla
(de todos modos el engine devuelve fallback hash y `model: "hash"`).

## Puerto 6775 — Qdrant API (`QdrantV1Controller`)

Capa HTTP propia sobre el motor Qdrant (recall BM25, indexado, sinónimos).

### `GET /status`

```json
{ "enabled": true, "qdrant": { "ok": true, "collections": ["mcp_tools", "tool_keywords", "query_synonyms", "contextual_memories"] } }
```

### `POST /tools/upsert`

Indexa tools en Qdrant (colecciones `mcp_tools` + `tool_keywords`). Body igual que
`POST /tools` de 6776. No construye grafo (eso lo hace el `POST /tools` de 6776).

**Respuesta 200:** `{ "indexed": 12 }` (cuenta el catálogo en memoria del tenant).

### `GET /tools/count?tenant=tenant-demo`

```json
{ "count": 12 }
```

### `POST /tools/search`

Recall BM25 de tools (sinónimos + RAKE + blend 60/40).

```json
{ "tenant": "tenant-demo", "text": "envía un correo", "limit": 20 }
```

**Respuesta 200:**

```json
{ "tools": [ { "name": "send_email", "score": 8.12 }, { "name": "web_search", "score": 2.4 } ] }
```

`limit` default `20`. **Errores:** `400` sin `tenant`/`text`; `500` interno.

### `POST /memories/search`

Recupera memorias por `userId` (BM25 con filtro).

```json
{ "userId": "user-1", "text": "hora de Perú", "limit": 8 }
```

**Respuesta 200:**

```json
{ "memories": [ { "id": "m1", "content": "...", "chatId": "chat-123", "createdAt": "...", "score": 0 } ] }
```

`limit` default `8`. `score` se deja en 0 aquí (el re-rank coseno ocurre en
`/memory/predict` de 6776).

### `POST /tools/synonyms`

Expande un texto con sinónimos de `query_synonyms` (utilidad/diagnóstico).

```json
{ "text": "enviar correo" }
```

**Respuesta 200:** `{ "expanded": "enviar correo email" }` (si no hay sinónimos, devuelve la query intacta).
