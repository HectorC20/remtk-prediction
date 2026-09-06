# remtk-prediction — Servicio unificado (TypeScript)

Servicio único que empaqueta **predicción + modelos ONNX + Qdrant** en un solo proceso
con tres puertos Express. Reemplaza al server Go legacy (mismos contratos).

```
┌──────────────────────────────────────────────────────────────┐
│                        PREDICTION (TS)                       │
│                                                              │
│  Puerto 6776  API de Predicción                              │
│    /predict · /tools · /tools/count · /memory/predict        │
│    /health · /debug                                          │
│                                                              │
│  Puerto 6777  API de Modelos (e5-large + e5-small)           │
│    POST /embed · GET /models · /health                       │
│                                                              │
│  Puerto 6775  API de Qdrant (BM25, keywording, indexado)     │
│    /tools/upsert · /tools/search · /memories/search          │
│    /tools/synonyms · /status · /health                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Arranque

```bash
cd prediction
pnpm install
pnpm dev          # desarrollo (tsx, hot reload, arranca los 3 puertos)
pnpm build        # compila a dist/
pnpm start        # producción: node dist/src/server.js
```

Al iniciar se loguean las 3 URLs:

```
[boot] → Predicción  http://localhost:6776
[boot] → Modelos     http://localhost:6777
[boot] → Qdrant      http://localhost:6775
```

> Qdrant externo (`localhost:6333`) se conecta igual que antes: el servicio 6775 es la
> capa HTTP propia sobre el motor Qdrant. Si Qdrant no está disponible, la predicción
> degrada automáticamente a catálogo completo (sin fallar).

---

## Uso como librería

El paquete exporta el núcleo **in-process** (sin abrir puertos) vía `src/index.ts`.
Requiere los modelos ONNX (en `./models` o `ONNX_MODELS_PATH`) y opcionalmente
Qdrant para el recall BM25 (si no está, la predicción degrada a catálogo completo).

```ts
import { createSystem, loadConfig } from "remtk-prediction";

const system = await createSystem(loadConfig());
await system.keywords.upsertTools(tenant, tools); // indexa keywords cross-idioma
const result = await system.orchestrator.predict({
  sessionId: "chat-1",
  tenant,
  text: "crea una herramienta para saber el precio del dólar en Perú",
  source: "human",
  history: [{ role: "user", content: "hola" }], // contexto previo
});
console.log(result.tools.map((t) => t.name)); // hasta MAX_OUTPUT_TOOLS (default 50, rango 0-50)
```

API exportada:

| Export | Descripción |
|---|---|
| `createSystem(cfg)` / `System` | Sistema completo (engine, qdrant, orchestrator, keywords…) |
| `loadConfig(env?)` / `AppConfig` | Config desde env vars (defaults listos) |
| `system.orchestrator.predict(input)` | Predicción de herramientas (pipeline 3 capas) |
| `system.orchestrator.memoryPredict(input)` | Predicción de memorias |
| `system.keywords.upsertTools(tenant, tools)` | Indexar keywords de tools |
| `EmbeddingEngine`, `QdrantService`, `RerankService`, `TurnClassifier`, `KeywordService`, `ConfirmationCache`, `Debugger` | Servicios reutilizables |
| `PredictionInput`, `PredictionResult`, `ToolDefinition`, `ChatMessage`, `Trace`, … | Tipos del contrato |

Para arrancar los servidores HTTP: `node dist/src/server.js` (o `import "remtk-prediction/server"`).

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `PORT_PREDICT` | `6776` | Puerto de predicción |
| `PORT_EMBED` | `6777` | Puerto de modelos |
| `PORT_QDRANT` | `6775` | Puerto Qdrant |
| `ONNX_ENABLED` | `true` | Cargar modelos ONNX (si `false` usa fallback hash) |
| `ONNX_MODELS_PATH` | `./models` → `../models` | Carpeta de los modelos |
| `ONNX_ADAPTIVE_MIN_TOOLS` | `2` | Mínimo de tools del umbral adaptativo |
| `ONNX_ADAPTIVE_MAX_TOOLS` | `30` | Máximo de tools del umbral adaptativo |
| `ONNX_ADAPTIVE_GAP_THRESHOLD` | `0.03` | Gap natural para cortar el ranking (escala de coseno e5-small) |
| `ONNX_ADAPTIVE_MIN_SCORE` | `0.8` | Score mínimo aceptado (coseno semántico) |
| `KEYWORD_BOOST` | `0.15` | Refuerzo sumado por confirm léxica de Qdrant (BM25, normalizada) |
| `KEYWORD_TOP_K` | `20` | Tamaño del top-K tras la reducción cross-idioma (capa 2) |
| `QDRANT_ENABLED` | `true` | Habilitar recall BM25 en Qdrant |
| `QDRANT_URL` | `http://localhost:6333` | URL del Qdrant |
| `QDRANT_API_KEY` | *(vacío)* | API key opcional |
| `RECALL_LIMIT` | `50` | Límite de recall por consulta |
| `MAX_OUTPUT_TOOLS` | `50` | *(Opcional)* Tope final de tools de salida tras el umbral adaptativo (rango `0-50`; `0` → nunca devolver herramientas) |

> Las colecciones de Qdrant (`mcp_tools`, `tool_keywords`, `query_synonyms`,
> `contextual_memories`) son **constantes fijas** del código
> (`shared/constants/qdrant/general.qdrant.ts`), no variables de entorno.

---

## Puerto 6776 — Predicción

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /predict`

Predice las herramientas MCP más relevantes para el texto del usuario (pipeline:
clasificador de turno → recall BM25 → re-rank con e5-small → umbral adaptativo).

**Body:**

```json
{
  "sessionId": "chat-123",
  "tenant": "tenant-demo",
  "text": "envía un correo electrónico al equipo con el resumen",
  "source": "human",
  "priorPlan": "{\"objetivo\":\"...\"}"
}
```

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `sessionId` | string | sí | Identificador de la conversación |
| `tenant` | string | sí | Tenant (catálogo de tools) |
| `text` | string | sí | Texto del usuario |
| `source` | `"human"` \| `"agent"` | no | Origen del turno |
| `priorPlan` | string (JSON) | no | Plan previo para turnos de confirmación |

**Respuesta 200:**

```json
{
  "tools": [
    {
      "id": "t1",
      "name": "send_email",
      "group": "communication",
      "category": "email",
      "description": "Envía un correo electrónico...",
      "tags": ["email", "correo"],
      "intentSummary": "Enviar correo",
      "inputSchema": {}
    }
  ],
  "complexity": "simple",
  "modelSize": "large",
  "rankedScores": [0.81, 0.8]
}
```

> `modelSize` = modelo usado por el pipeline (`large`/`small`/`hash` si hay fallback).
> Si el turno es de confirmación y hay plan en caché para la sesión, responde con
> early-exit sin re-computar embeddings.

**Errores:** `400` si faltan `sessionId`/`tenant`/`text`; `500` si falla el pipeline.

### `POST /tools`

Registra (indexa) herramientas del tenant en el caché y en Qdrant.

**Body:**

```json
{
  "tenant": "tenant-demo",
  "tools": [
    {
      "id": "t1",
      "name": "send_email",
      "group": "communication",
      "category": "email",
      "description": "Envía un correo electrónico al equipo",
      "tags": ["email"],
      "intentSummary": "Enviar correo",
      "inputSchema": {}
    }
  ]
}
```

**Respuesta 200:** `{ "indexed": 12 }`

### `GET /tools/count?tenant=tenant-demo`

```json
{ "count": 12 }
```

### `POST /memory/predict`

Servicio de **memoria de predicción**: recupera memorias relevantes del usuario.

**Body:**

```json
{
  "sessionId": "chat-123",
  "tenant": "tenant-demo",
  "text": "qué hora es en Perú",
  "limit": 8
}
```

**Respuesta 200:**

```json
{
  "memories": [
    {
      "id": "m1",
      "content": "El usuario está en Perú (Piura)...",
      "chatId": "chat-123",
      "createdAt": "2026-08-29T22:19:00Z",
      "score": 0.88
    }
  ],
  "topicShift": false,
  "topicScore": 0.92,
  "modelSize": "small",
  "rankedScores": [0.88]
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `topicShift` | boolean | `true` si el tema cambió respecto al turno anterior |
| `topicScore` | number | Similitud con el turno anterior (0–1) |
| `modelSize` | string | Modelo usado (`small`/`hash`) |

### `GET /debug`

Estado interno: motor de embeddings, estadísticas y trazas de las últimas predicciones.

```json
{
  "engine": { "size": "large", "modelPath": "../models/multilingual-e5-large-onnx/model.onnx" },
  "stats": {
    "tenants": 1,
    "confirmGets": 0,
    "confirmHits": 0,
    "confirmSets": 1,
    "embeddingsRecomputed": 12,
    "embeddingsCached": 2
  },
  "traces": [ { "sessionId": "...", "turnType": "new_query", "recall": 2, "latencyMs": 45 } ]
}
```

---

## Puerto 6777 — Modelos

### `GET /models`

Estado de los dos modelos empaquetados:

```json
{
  "large":  { "model": "large", "dim": 1024, "modelPath": "../models/multilingual-e5-large-onnx/model.onnx" },
  "small":  { "model": "small", "dim": 384,  "modelPath": "../models/multilingual-e5-small-onnx/model.onnx" }
}
```

> `model` = `"hash"` si el modelo no está disponible y se usa el fallback determinístico.

### `POST /embed`

Genera el embedding de un texto. **El caller aplica el prefijo** `query: ` o `passage: `
según el caso (contrato del modelo e5).

**Body:**

```json
{ "text": "query: envía un correo al equipo", "size": "large" }
```

| Campo | Tipo | Descripción |
|---|---|---|
| `text` | string | Texto a embeber (requerido) |
| `size` | `"large"` \| `"small"` | Modelo; si se omite usa el default (`large`) |

**Respuesta 200:**

```json
{
  "embedding": [0.0123, -0.0456, 0.0891, "..."],
  "model": "large",
  "dim": 1024,
  "modelPath": "../models/multilingual-e5-large-onnx/model.onnx"
}
```

---

## Puerto 6775 — Qdrant

Capa HTTP propia sobre el motor Qdrant (BM25 sparse + hash djb2, igual que el monolith).

### `GET /status`

```json
{
  "enabled": true,
  "ok": true,
  "collections": ["mcp_tools", "tool_keywords", "query_synonyms", "contextual_memories"]
}
```

### `POST /tools/upsert`

Indexa herramientas en Qdrant (colecciones `mcp_tools` + `tool_keywords`).

**Body:** igual que `POST /tools` de 6776 → **Respuesta:** `{ "indexed": 12 }`

### `GET /tools/count?tenant=tenant-demo`

```json
{ "count": 12 }
```

### `POST /tools/search`

Recall BM25 de herramientas (expansión de sinónimos + keywords RAKE + blend 60/40).

**Body:**

```json
{ "tenant": "tenant-demo", "text": "envía un correo", "limit": 20 }
```

**Respuesta 200:**

```json
{
  "tools": [
    { "name": "send_email", "score": 8.12 },
    { "name": "web_search", "score": 2.4 }
  ]
}
```

### `POST /memories/search`

Recupera memorias del usuario por filtro `userId`.

**Body:**

```json
{ "userId": "user-1", "text": "hora de Perú", "limit": 8 }
```

**Respuesta 200:**

```json
{
  "memories": [
    {
      "id": "m1",
      "content": "El usuario está en Perú (Piura)...",
      "chatId": "chat-123",
      "score": 0.88
    }
  ]
}
```

### `POST /tools/synonyms`

Expande un texto con sinónimos de la colección `query_synonyms`.

**Body:** `{ "text": "enviar correo" }` → **Respuesta:** `{ "expanded": "enviar correo email" }`

---

## Ejemplos rápidos (curl)

```bash
# Health de los 3 servicios
curl http://localhost:6776/health
curl http://localhost:6777/health
curl http://localhost:6775/health

# Predecir herramientas
curl -X POST http://localhost:6776/predict \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"c1","tenant":"demo","text":"envía un correo al equipo"}'

# Memoria de predicción
curl -X POST http://localhost:6776/memory/predict \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"c1","tenant":"demo","text":"qué hora es en Perú"}'

# Embedding con e5-large
curl -X POST http://localhost:6777/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"query: enviar correo","size":"large"}'

# Recall BM25
curl -X POST http://localhost:6775/tools/search \
  -H "Content-Type: application/json" \
  -d '{"tenant":"demo","text":"enviar correo"}'
```

---

## Tests

```bash
pnpm test         # 17 tests herméticos (sin modelos ni Qdrant)
pnpm test:model   # 5 tests con modelos ONNX reales (e5-large/e5-small)
pnpm test:all     # ambos
```
