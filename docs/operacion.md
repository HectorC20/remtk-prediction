# Operación

## Requisitos

- **Node ≥ 20** (imagen de referencia: `node:22-slim`, glibc).
- **pnpm** (proyecto gestionado con pnpm; los binarios ONNX/tokenizers requieren
  glibc — **no** funcionan en Alpine/musl).
- Modelos ONNX en `./models` (ver [configuracion.md](configuracion.md)) para usar
  el pipeline real; sin ellos el servicio degrada a hash.
- Qdrant opcional en `QDRANT_URL` para el recall BM25.

## Instalación y arranque

```bash
pnpm install
pnpm dev          # desarrollo: predev (ensure-qdrant) + tsx src/server.ts (3 puertos)
pnpm build        # compila a dist/ (tsc)
pnpm start        # producción: node dist/src/server.js
```

`pnpm dev` dispara el hook **`predev`** → `scripts/ensure-qdrant.mjs`: si el puerto
de `QDRANT_URL` no responde levanta el contenedor con `docker compose up -d` (abre
Docker Desktop si el daemon está apagado). Se omite con `QDRANT_ENABLED=false`.

Al arrancar se loguea la configuración y las URLs:

```
[boot] config: predict=6776 embed=6777 qdrant=6775
[boot] models: path=... size=small onnx=true
[boot] qdrant: enabled=true url=http://localhost:6333
[boot] → Predicción  http://localhost:6776
[boot] → Modelos     http://localhost:6777
[boot] → Qdrant      http://localhost:6775
```

Apagado limpio con `SIGINT`/`SIGTERM`.

## Uso como librería (in-process, sin puertos)

```ts
import { createSystem, loadConfig } from "remtk-prediction";

const system = await createSystem(loadConfig());

// 1. Registrar el catálogo del tenant: indexa keywords/Qdrant y construye
//    el grafo topológico del tenant (necesario para la ruta por grafo).
await system.orchestrator.registerTools(tenant, tools);

// 2. Predecir
const result = await system.orchestrator.predict({
  sessionId: "chat-1",
  tenant,
  text: "envía un correo al equipo",
  source: "human",
  history: [{ role: "user", content: "hola" }],
});
console.log(result.tools.map((t) => t.name));
console.log(result.graph.executionOrder); // orden topológico (ruta grafo)
```

Exports principales (`src/index.ts`):

| Export | Descripción |
|---|---|
| `createSystem(cfg)` / `System` | Grafo completo (engine, qdrant, orchestrator, keywords, graphCache, sessionState, rerank…). |
| `loadConfig(env?)` / `AppConfig` | Config desde env vars. |
| `system.orchestrator.predict / predictMemory / registerTools / countTools` | API del pipeline. |
| `EmbeddingEngineService`, `QdrantService`, `KeywordService`, `TurnClassifier`, `ConfirmationCache`, `Debugger` | Servicios reutilizables (pipeline plano y estado). |
| `ToolGraphCacheService`, `SessionStateCacheService`, `RerankService` | Servicios del enrutador topológico (grafo, z_t, rerank). |
| `cosine`, `normalizeL2`, `adaptiveThreshold`, `estimateComplexity`, `topologicalSort` | Utilidades. |
| Tipos del contrato (`ToolDefinition`, `PredictionInput`, `Trace`, `GraphPredictionResult`, `TenantToolGraph`, …) | Tipado de entradas/salidas. |

Entry de servidor: `import "remtk-prediction/server"` o `node dist/src/server.js`.

## Tests

```bash
pnpm test          # herméticos: rerank + e2e + graph (sin ONNX real ni Qdrant)
pnpm test:graph    # solo graph.test.ts (grafo, topo-sort, session state)
pnpm test:e2e      # solo e2e (levanta los 3 listeners en puertos efímeros)
pnpm test:model    # e5-small real: embed-model.test.ts (requiere ./models)
pnpm test:compare  # e5-small vs e5-large real: embed-compare.test.ts (dims, L2, calidad, latencia)
pnpm test:bge      # benchmark BGE-M3 int8 (externo al engine): embed-bge.test.ts
pnpm test:all      # rerank + e2e + graph + embed-model
```

Detalles:

- **rerank.test.ts** — unitarios de `cosine`, `adaptiveThreshold` y `estimateComplexity`.
- **graph.test.ts** (`pnpm test:graph`) — unitarios del enrutador topológico:
  `buildTenantGraph`, orden topológico (Kahn), resolución de mutexes, blend
  `SessionStateCacheService` y latencia de propagación < 15 ms. Hermético (sin
  modelo ni Qdrant).
- **e2e.test.ts** — levanta los 3 listeners HTTP en puertos efímeros y valida
  `/predict` (respuesta con bloque `graph`), `/tools`, `/tools/count`,
  `/memory/predict`, `/embed`, `/models` (solo `small`), `/status`, `/debug` con
  `onnxEnabled=false` y `qdrantEnabled=false` (usa el ejemplo de 12 tools de
  `tests/example-tools.ts`).
- **embed-model.test.ts** — carga `multilingual-e5-small` real y verifica modelo
  `small`, dim 384 y vectores normalizados. Requiere `model.onnx`/`tokenizer.json`
  en `./models`.
- **embed-compare.test.ts** (`pnpm test:compare`) — carga e5-small y e5-large
  reales: dims 384/1024, normalización L2, calidad semántica (afines > disímiles)
  y latencia sobre texto de 512 tokens (large más lento que small).
- **embed-bge.test.ts** (`pnpm test:bge`) — benchmark standalone de
  `multilingual-bge-m3-int8-onnx` (`model_quantized.onnx`, dim 1024) con ONNX
  directo (no usa el engine): latencia corto/largo y separación semántica vs
  e5-small.

## Docker

`Dockerfile` multi-stage sobre `node:22-slim` (glibc; ONNX Runtime no funciona en
musl/Alpine):

- **Builder:** instala `python3 make g++`, `pnpm@9.15.9`, instala dependencias con
  lockfile, compila a `dist/`.
- **Runtime:** instala `libgomp1` (OpenMP de ONNX Runtime multi-hilo) y
  `ca-certificates` (evita warning de telemetría de ORT); copia `dist`, `node_modules`
  y `package.json`.
- **Modelos nunca van en la imagen:** se montan por bind mount en
  `ONNX_MODELS_PATH=/app/models`.
- Expone `6776 6777 6775`; `CMD ["node", "dist/src/server.js"]`.

```bash
# Build + run (con Qdrant y modelos montados)
docker build -t remtk-prediction .
docker run --rm -p 6776:6776 -p 6777:6777 -p 6775:6775 \
  -v "$PWD/models:/app/models" \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  remtk-prediction
```

`.dockerignore` excluye `node_modules`, `dist`, `models`, `tests`, `.git` y logs.

## Logs y observabilidad (`src/shared/logger/mod.rs`)

El servidor en Rust nativo incorpora un subsistema de observabilidad estructurada basado en `tracing` y `tracing-subscriber` (`init_logger()`):

- **Escritura Dual (`DualMakeWriter` / `DualWriter`)**: cada evento de log se emite simultáneamente a `stdout` (para inspección con `docker logs remtk_prediction_rust`) y al archivo rotativo diario `logs/prediction-YYYY-MM-DD.log` (directorio configurable vía `LOGS_DIR`, activo por defecto con `LOG_TO_FILE=true`).
- **Rotación Diaria sin Dependencias Pesadas**: `RotatingFileState` calcula la fecha UTC actual (`YYYY-MM-DD`) mediante el algoritmo civil de Howard Hinnant y rota automáticamente el descriptor de archivo al cambiar de día.
- **Formato y Filtros (`RUST_LOG` y `LOG_FORMAT`)**:
  - Por defecto usa el filtro `info,remtk_prediction_server=debug,profiler=info,prediction_trace=info,http_trace=info`.
  - Si se define `LOG_FORMAT=json`, emite objetos JSON estructurados por línea.
- **Trazas Estructuradas del Pipeline**:
  - `[trace:recall]` (`target: "prediction_trace"`): tamaño del catálogo del tenant, términos tokenizados de la consulta, cantidad de coincidencias léxicas no nulas y top-5 herramientas léxicas.
  - `[trace:moe]` (`target: "prediction_trace"`): cantidad de herramientas evaluadas, estado NLI dominante (`TRANSICION`, `CONTINUACION`, `CORRECCION`, `ABANDONO`), vector de probabilidades NLI, `abstencion_prob` frente a `threshold`, `topic_shift` y pesos de activación de los 6 expertos.
  - `[profile:predict]` (`target: "profiler"`): latencia total y desglose por sub-etapa (`recall`, `q_emb`, `query_tokens`, `ctx_tokens`, `assemble`, `moe`), tamaño del catálogo, candidatas evaluadas y herramientas ganadoras.
  - `[trace:predict]` (`target: "prediction_trace"`): consulta e `keywords`, `min_score`, `gap_thresh`, lista `selected` de herramientas ganadoras, `top_candidates` con sus scores exactos y `filtered_out` con las herramientas descartadas por umbral o tope. Si el tenant no tiene herramientas indexadas en memoria, emite un `WARN` con `status=COLD_CATALOG`.
  - `[trace:route]`, `[trace:memory_judge]`, `[trace:spatial]`, `[trace:qdrant_search]`: trazas detalladas de decisiones Fast-Path, juicio de memoria, predicción espacial y búsquedas en catálogo.
  - `[http:predict]`, `[http:spatial_predict]`, `[http:predict_route]`, `[http:history_regime]` (`target: "http_trace"`): trazas de transporte HTTP con código de estado, latencia extremo a extremo y resumen de respuesta.
- `GET http://localhost:6776/debug` (o `:6876/debug` en contenedor host) expone:
  - `status`: `"healthy"`,
  - `memory_rss_mb`: estimación de memoria RSS del proceso y tensores FP16,
  - `active_tenants`: cantidad de tenants con catálogo cargado en `DashMap`,
  - `total_cached_tools`: total de herramientas indexadas en memoria,
  - `onnx_runtime_version`: `"1.20.1"`.

## Degradación esperada (comportamiento, no error)

| Situación | Qué se ve |
|---|---|
| ONNX deshabilitado / modelo ausente | `model=hash`, dim 384, vectores determinísticos. Predicciones con scores más débiles. |
| Qdrant deshabilitado/caído | Recall vacío; predict degrada a catálogo completo; `GET /status` reporta `ok:false`. |
| Qdrant optimizando (timeout) | Reintentos con backoff; si persiste, degrade suave. |
| Sin grafo del tenant (no indexado) | `/predict` usa el fallback plano por keywords (sin propagación ni orden topológico). |
| `POST /tools` durante predicts activos | El recompute de embeddings es de prioridad baja: los predicts saltan en la cola de inferencia. |
