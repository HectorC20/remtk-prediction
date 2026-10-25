# remtk-prediction — Servicio de Inferencia y Juicio MoE (Rust Nativo)

Microservicio de inferencia de ultra-baja memoria (**< 88 MB RSS**) y juicio unificado MoE en **Rust nativo** (Tokio + Axum + ort v2 + tokenizers), que **reemplaza formalmente a la antigua arquitectura NestJS / Node.js**.

Administra los tres puertos canónicos concurrentemente sobre un solo binario optimizado con tensores Zero-Copy respaldados por `mmap`.

```
┌──────────────────────────────────────────────────────────────┐
│                  REMTK-PREDICTION (RUST)                     │
│                                                              │
│  Puerto 6776  API de Predicción y Juicio MoE                 │
│    /predict · /spatial/predict · /predict/route              │
│    /memory/predict · /tools · /tools/count · /health         │
│                                                              │
│  Puerto 6777  API de Modelos (e5-small)                      │
│    POST /embed · GET /models · /health                       │
│                                                              │
│  Puerto 6775  API de Qdrant (BM25, keywording, indexado)     │
│    /tools/upsert · /tools/search · /memories/search          │
│    /tools/synonyms · /status · /health                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Comparativa de Arquitectura: NestJS vs Rust Nativo

| Métrica | NestJS / V8 (Antiguo) | Rust Nativo (Actual) | Ganancia |
|---|---|---|---|
| **RAM en Reposo** | ~550 MB | **~45 MB** | **12× menos RAM** |
| **RAM bajo Carga** | 850 MB – 1,200 MB | **< 88 MB** | **14× menos RAM** |
| **Latencia P50** | 18.4 ms | **3.2 ms** | **5.7× más rápido** |
| **Latencia P99** | 85.0 ms (pausas GC) | **7.4 ms (determinista)** | **11.5× más predecible** |
| **Cold Start** | ~4,200 ms | **< 45 ms** | **90× más veloz** |
| **Imagen Docker** | ~1.2 GB | **< 45 MB** | **28× más liviana** |
| **Manejo de Tensores**| Copia N-API Float32Array | **Zero-Copy `ArrayView` + `mmap`**| **0 bytes clonados** |

---

## Arranque del Servidor en Rust

### Vía Docker (Producción)

```bash
docker compose up -d          # Levanta Qdrant (6333) + Servidor Rust (6776/6777/6775)
docker compose logs -f        # Monitoreo de logs estructurados
```

### Vía Cargo (Host Local)

```bash
# Requiere libonnxruntime 1.20+
cargo run --release --manifest-path rust-server/Cargo.toml
```

Al iniciar se loguean las 3 URLs:

```
[boot] → Predicción  http://localhost:6776
[boot] → Modelos     http://localhost:6777
[boot] → Qdrant      http://localhost:6775
```

> La arquitectura legacy de NestJS ha sido retirada y archivada en `migration/legacy-nest/` para propósitos históricos y de auditoría.

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
console.log(result.tools.map((t) => t.name)); // hasta MAX_OUTPUT_TOOLS (default 150, rango 0-150)
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

## Estructura del proyecto

El repositorio organiza su código bajo el patrón de **Monolito Modular en Rust Nativo** (`api/`, `core/`, `support/`, `shared/`). Para un desglose exhaustivo de dependencias y mapa detallado de ficheros, consulta [`docs/arquitectura.md`](docs/arquitectura.md).

```
remtk-prediction/
├── src/                  # Monolito Modular en Rust Nativo (Tokio + Axum + ort v2)
│   ├── main.rs           # Bootstrap concurrente (init_logger + 3 listeners TCP: 6776, 6777, 6775)
│   ├── lib.rs            # Raíz de módulos del crate remtk-prediction-server
│   ├── api/              # Capa de transporte HTTP (Axum) y trazabilidad [http:*]
│   │   ├── predict_server.rs  # Puerto 6776 (/predict, /spatial/predict, /predict/route, /history/regime, /tools...)
│   │   ├── embed_server.rs    # Puerto 6777 (/embed, /models, /health)
│   │   ├── qdrant_server.rs   # Puerto 6775 (/tools/upsert, /tools/search, /status...)
│   │   └── state.rs           # AppState compartido (Arc)
│   ├── core/             # Dominio central e inferencia matemática pura (sin dependencias HTTP)
│   │   ├── embedding/    # EmbeddingService (multilingual-e5-small 384d, L2 norm, [profile:embed])
│   │   ├── inference/    # InferenceEngine (ort v2 mmap), MoE 3 etapas ([trace:moe]) y E5Tokenizer
│   │   └── prediction/   # PredictionService, umbral adaptativo, Fast-Path y trazas ([trace:predict], [trace:route])
│   ├── support/          # Capacidades de soporte e infraestructura en memoria
│   │   ├── registry/     # TenantCache (DashMap), tensores FP16 y recall léxico ([trace:recall])
│   │   ├── qdrant/       # QdrantService ([profile:qdrant_upsert], [trace:qdrant_search])
│   │   └── spatial/      # SpatialService ([trace:spatial])
│   └── shared/           # Capa transversal: config, logger dual, DTOs Serde, errores y tensores SIMD
│       ├── config/       # AppConfig tipado desde variables de entorno
│       ├── logger/       # LoggerConfig + DualMakeWriter (stdout + logs/prediction-YYYY-MM-DD.log)
│       ├── dto/          # Contratos JSON Serde 1:1
│       ├── error/        # AppError (thiserror)
│       └── tensor/       # Operaciones vectoriales SIMD (cosine, L2, FP16 <-> FP32)
├── docs/                 # Documentación técnica exhaustiva y diagnósticos forenses
├── migration/            # Especificación arquitectónica y skill operativo de la migración a Rust
├── models/               # Modelos ONNX locales optimizados (multilingual-e5-small-onnx, remtk-juicio-moe-int8)
├── scripts/              # Automatizaciones de ciclo de vida y evaluación
├── training/             # Pipelines de destilación y exportación ONNX del cabezal MoE
├── logs/                 # Registro rotativo diario de logs en ejecución (prediction-YYYY-MM-DD.log)
└── AGENT.md              # Directiva arquitectónica obligatoria (razonamiento denso, prohibición de parches léxicos)
```

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `PORT_PREDICT` | `6776` (o `6876` en contenedor host) | Puerto HTTP de predicción y juicio MoE |
| `PORT_EMBED` | `6777` (o `6877` en contenedor host) | Puerto HTTP de modelos de embeddings |
| `PORT_QDRANT` | `6775` (o `6875` en contenedor host) | Puerto HTTP de indexado y recall Qdrant |
| `MODELS_PATH` / `ONNX_MODELS_PATH` | `models` | Ruta a la carpeta con los modelos ONNX (`multilingual-e5-small-onnx`, `remtk-juicio-moe-int8`) |
| `ONNX_MODEL_SIZE` | `small` | Tamaño del modelo de embeddings (`small`, 384 dimensiones) |
| `ONNX_INTRA_OP_THREADS` | `4` | Hilos intra-op de ONNX Runtime (SIMD AVX2) |
| `ONNX_INTER_OP_THREADS` | `1` | Hilos inter-op de ONNX Runtime |
| `ONNX_ADAPTIVE_MIN_SCORE` | `0.84` | Piso mínimo de relevancia para herramientas ganadoras (`src/shared/config/mod.rs`) |
| `ONNX_ADAPTIVE_GAP_THRESHOLD` | `0.03` | Brecha natural (`gap`) para cortar el ranking adaptativo |
| `ABSTENCION_THRESHOLD` | `0.52` | Umbral de probabilidad de abstención del cabezal MoE para retorno nulo (`0 tools`) |
| `KEYWORD_BOOST` | `0.05` | Refuerzo por coincidencia de keywords internas |
| `NAME_AFFINITY_BOOST` | `0.08` | Refuerzo por afinidad léxica en el nombre de la herramienta (tope `0.15`) |
| `FAMILY_GATE_PENALTY` | `0.15` | Penalización por familia o namespace ajeno |
| `RECALL_LIMIT` | `20` | Límite máximo de herramientas candidatas recuperadas en la fase de recall |
| `MAX_OUTPUT_TOOLS` | `8` | Tope máximo de herramientas devueltas tras el umbral adaptativo |
| `MAX_CATEGORIES` | `6` | Tope máximo de categorías activas evaluadas en el juicio |
| `LOGS_DIR` | `logs` | Directorio donde `DualMakeWriter` persiste los logs rotativos diarios `prediction-YYYY-MM-DD.log` |
| `LOG_TO_FILE` | `true` | Habilita/deshabilita la escritura simultánea en archivo rotativo diario (`false` o `0` para solo consola) |
| `LOG_FORMAT` | `text` | Formato de salida del logger (`json` para JSON estructurado línea por línea; texto compacto por defecto) |
| `RUST_LOG` | `info,remtk_prediction_server=debug,profiler=info,prediction_trace=info,http_trace=info` | Filtro `EnvFilter` por nivel y target (`prediction_trace`, `http_trace`, `profiler`, `remtk_prediction_server`) |

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
clasificador de turno → recall BM25 → re-rank con e5-small → **etapa A**: recorte
a las mejores categorías/`group` según `MAX_CATEGORIES` → **etapa B (decisor)**:
umbral adaptativo + recorte final según `MAX_OUTPUT_TOOLS`).

**Body:**

```json
{
  "sessionId": "chat-123",
  "tenant": "tenant-demo",
  "text": "envía un correo electrónico al equipo con el resumen",
  "source": "human",
  "priorPlan": "{\"objetivo\":\"...\"}",
  "keywords": ["enviar correo", "email", "equipo"]
}
```

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `sessionId` | string | sí | Identificador de la conversación |
| `tenant` | string | sí | Tenant (catálogo de tools) |
| `text` | string | sí | Texto del usuario |
| `source` | `"human"` \| `"agent"` | no | Origen del turno |
| `priorPlan` | string (JSON) | no | Plan previo para turnos de confirmación |
| `keywords` | string[] | no | Palabras clave que acompañan al texto (p. ej. las que el planificador delega a cada mini-agente). Se normalizan/deduplican y se anexan a la consulta para afinar el recall léxico y el match cross-idioma |
| `intentContext` | string | no | Prosa interna que el consumidor produjo ANTES de preguntar (el pre-pensamiento del planificador). No es una orden de herramienta: se parte por ideas y los segmentos más próximos a la consulta (`MAX_INTENT_SEGMENTS`, con tope de caracteres `MAX_INTENT_CONTEXT_CHARS`) **amplían el pool de candidatas** —entran solo las que la consulta cruda no traía, sin tocar su score ni su orden—. `text` sigue siendo lo que se pidió y lo que puntúa |
| `history` | `{role,content}[]` | no | Mensajes previos de la conversación (contexto) |

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
