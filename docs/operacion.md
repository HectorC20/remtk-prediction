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

## Logs y observabilidad

- Logger compacto con timestamp ISO: `2026-09-05T12:00:00.000Z [tag] mensaje`.
- Cada línea se **persiste además en un archivo diario**
  `logs/prediction-YYYY-MM-DD.log` (carpeta `LOGS_DIR` o `<cwd>/logs`, mejor
  esfuerzo: si no puede escribir, sigue solo en consola). Ver
  [configuracion.md](configuracion.md).
- Tags principales: `[boot]`, `[embed]`, `[classifier]`, `[keywords]`, `[graph]`,
  `[rerank]`, `[rerank:graph]`, `[recall]`, `[memory]`, `[tools]`, `[pipeline]`,
  `[qdrant]`.
- `GET http://localhost:6776/debug` expone:
  - estado del engine (`size`, `modelPath` — `hash` si degradó),
  - estadísticas acumuladas (tenants, confirmGets/Hits/Sets, embeddings
    recomputados vs cacheados),
  - las últimas 20 trazas de predicción (latencia, turno, recall, modelo, etc.).
- En `dist/src/server.js` está el entry de producción; el paquete publica `dist`,
  `models`, `README` y `LICENSE`.

## Degradación esperada (comportamiento, no error)

| Situación | Qué se ve |
|---|---|
| ONNX deshabilitado / modelo ausente | `model=hash`, dim 384, vectores determinísticos. Predicciones con scores más débiles. |
| Qdrant deshabilitado/caído | Recall vacío; predict degrada a catálogo completo; `GET /status` reporta `ok:false`. |
| Qdrant optimizando (timeout) | Reintentos con backoff; si persiste, degrade suave. |
| Sin grafo del tenant (no indexado) | `/predict` usa el fallback plano por keywords (sin propagación ni orden topológico). |
| `POST /tools` durante predicts activos | El recompute de embeddings es de prioridad baja: los predicts saltan en la cola de inferencia. |
