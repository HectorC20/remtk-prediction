# Registro de actualizaciones

Bitácora de los cambios relevantes de **remtk-prediction**. Las entradas más
recientes van primero. Cada una referencia el hash de git, el alcance del cambio
y la documentación técnica que lo describe.

---

## 2026-10-25 · feat(logger,docs): subsistema de observabilidad estructurada en Rust y diagnóstico forense de cron jobs

Implementación del módulo transversal de logging estructurado en Rust nativo (`src/shared/logger/mod.rs`), instrumentación completa del pipeline de predicción y documentación forense de selección de herramientas `scheduler`. Ver [configuracion.md](configuracion.md), [operacion.md](operacion.md), [arquitectura.md](arquitectura.md) y [diagnostico-fallo-cron-jobs.md](diagnostico-fallo-cron-jobs.md).

- **Subsistema `src/shared/logger/mod.rs`**:
  - `LoggerConfig::from_env()`: soporte para `LOGS_DIR` (default `logs`), `LOG_TO_FILE` (`true`), `LOG_FORMAT` (`text` | `json`) y `RUST_LOG` (`info,remtk_prediction_server=debug,profiler=info,prediction_trace=info,http_trace=info`).
  - `DualMakeWriter` y `DualWriter`: multiplexación `io::Write` que emite simultáneamente en `stdout` y en archivos rotativos diarios `logs/prediction-YYYY-MM-DD.log` (cálculo de fecha UTC sin dependencias externas mediante algoritmo civil de Howard Hinnant).
  - Utilidades `preview_text()` (recorte seguro en fronteras de carácter UTF-8) y `format_scored_slice()`.
- **Instrumentación Estructurada del Pipeline**:
  - `[trace:recall]` (`src/support/registry/tool_cache.rs`): tamaño del catálogo, términos tokenizados y candidatos léxicos iniciales.
  - `[trace:moe]` (`src/core/inference/moe.rs`): estado NLI dominante (`TRANSICION`, `CONTINUACION`, `CORRECCION`, `ABANDONO`), probabilidades por clase, probabilidad de abstención frente a umbral, `topic_shift` y pesos de los 6 expertos.
  - `[profile:predict]` y `[trace:predict]` (`src/core/prediction/service.rs`): desglose de latencias por sub-etapa, herramientas ganadoras (`selected`), `top_candidates` con scores y `filtered_out` (herramientas descartadas con sus scores), además de alertas `WARN` ante `COLD_CATALOG`.
  - `[trace:route]`, `[trace:memory_judge]`, `[trace:qdrant_search]`, `[trace:spatial]` y `[http:*]` (`src/api/predict_server.rs`).
- **Diagnóstico Forense (`docs/diagnostico-fallo-cron-jobs.md`)**:
  - Investigación empírica de la sesión `df08bb63-cf1e-41c1-9a2b-8c68100ed00c` documentando por qué `schedule_task` (`0.8336`, posición #16) y `schedule_list` (`0.8373`, posición #4) quedaron bajo el piso `adaptive_min_score = 0.84` y cómo interactuaron con `PlannerV8Service` y `AimcpV8Service`.

---

## 2026-10-22 · docs: actualización exhaustiva de la estructura y arquitectura

Actualización completa de la documentación estructural en [`docs/arquitectura.md`](arquitectura.md) y [`README.md`](../README.md) reflejando el estado actual del repositorio:

- **Estructura global de carpetas**: Documentación formal de `bench/`, `docs/`, `models/`, `scripts/`, `tests/` y `logs/`.
- **Módulo de Juicio Cognitivo (`src/juicio/`)**: Incorporación al mapa de `JuicioService` y sus 6 subsistemas neuronales (`AbstencionService`, `CrossEncoderService`, `EspecificidadService`, `EstadoNliService`, `MaxSimService`, `PuertaContextoService`).
- **Nuevos servicios en `src/predict/services/`**: Mapeo de `calibration.service.ts`, `lexical-profile.service.ts`, `remtk-memory-skills.adapter.ts`, `skill-memory.service.ts`, `spatial-predict.service.ts` y `variante-c.service.ts`.
- **Grafo de dependencias y composición (`src/main.ts`)**: Diagrama actualizado reflejando la inyección de la capa de juicio, adaptadores de memoria y perfiles léxicos.
- **Submódulo `src/shared/`**: Detalle de los nuevos dominios de constantes (`juicio/`, `spatial/`) y los 18 contratos tipados en `interfaces/`.

---

## 2026-09-06 · `323a1e8` — feat(rerank,logger): mejora del re-rank y logging a archivo

Implementa el **enrutador topológico (Graph Router)** sobre el espacio latente y
añade persistencia de logs a archivo. Ver [pipeline-prediccion.md](pipeline-prediccion.md),
[embeddings.md](embeddings.md) y [arquitectura.md](arquitectura.md).

- `RerankService` gana la ruta `graphFilter`: puntúa el catálogo completo contra
  el estado latente de sesión `z_t`, refuerza con BM25, propaga pre-requisitos
  (`S_propagado = S + α·Aᵀ·S`, α = 0.2), aplica umbral adaptativo, resuelve
  mutexes y ordena topológicamente (Kahn). La ruta plana (`filter`) se conserva
  como fallback.
- `ToolGraphCacheService` (nuevo): grafo G = (V, E) por tenant construido en
  `POST /tools`. Nodo = embedding `passage:` de la firma de la tool (e5-small);
  aristas `PREREQUISITE` (0.85, declaradas + inferidas por `inputSchema`),
  `MUTUALLY_EXCLUSIVE` (1.0) y `CO_OCCURRENCE` por `group` (0.4); matriz de
  adyacencia aplanada N×N.
- `SessionStateCacheService` (nuevo): mantiene `z_t` por sesión; si el coseno
  con el prompt entrante ≥ 0.86 hace blend 70/30 y si < 0.86 resetea
  (topic-shift).
- `POST /predict` devuelve `GraphPredictionResult` (bloque `graph` con
  `nodes`/`edges`/`executionOrder`); `POST /tools` ahora también indexa el grafo.
- Tests: nuevo `tests/graph.test.ts` (`pnpm test:graph`, incluido en `pnpm test`
  y `test:all`).
- Logger: cada línea se persiste en `logs/prediction-YYYY-MM-DD.log`
  (env `LOGS_DIR` o `<cwd>/logs`) además de la consola. `Debugger` queda con
  trazas en anillo para `GET /debug`.

## 2026-09-06 · `286935d` — refactor: rename embedding service, centralizar turn types

- `embedding-engine.ts` → `embedding.service.ts` (`EmbeddingEngineService`), con
  helpers `embedQuery`/`embedPassage` (prefijos `query:`/`passage:`) y
  `getInfo()`. Ver [embeddings.md](embeddings.md).
- Tipos de turno centralizados en el enum `TurnType`
  (`src/shared/dictionary/turn.dictionary.ts`): `new_query`,
  `confirmation_empty`, `confirmation_nuance`.
- Limpieza de residuos de la estructura anterior (servicios duplicados, archivos
  de listeners Express).

## 2026-09-06 · `2422f19` — refactor: restructure codebase a NestJS

- Los tres listeners Express pasan a **apps Nest autocontenidas**
  (`PredictAppModule` → 6776, `EmbeddingAppModule` → 6777, `QdrantAppModule` →
  6775), con controladores `PredictV1Controller`, `EmbeddingV1Controller`,
  `QdrantV1Controller`. El bootstrap (`src/app.module.ts`) monta el parser JSON
  (`express.json`, límite 4mb) y `src/server.ts` arranca las tres en paralelo.
- Reorganización de `shared/`: `constants/`, `interfaces/`, `dictionary/`.
  `src/main.ts` conserva `createSystem` (uso como librería in-process). Ver
  [arquitectura.md](arquitectura.md).
- Mantiene los contratos HTTP 1:1 con el server Go original
  ([api-http.md](api-http.md)).

## 2026-09-05 · `fd3f173` — first commit: servicio unificado TypeScript

Migración 1:1 desde el server Go legacy (`remtk-prediction` TS):

- Un solo proceso Node con 3 listeners: predicción, modelos e5 ONNX y API Qdrant.
- Pipeline de 3 capas: keywords cross-idioma → Qdrant BM25 sparse (hash djb2
  client-side, IDs uuidv5 DNS) → re-rank por cosine e5-small + umbral adaptativo.
- Modelos ONNX: `multilingual-e5-small-onnx` (384 dims, default) y
  `multilingual-e5-large-onnx` (1024 dims, bajo demanda); fallback determinístico
  `hashEmbedding` si ONNX no está disponible.
- Fallback suave ante degradación: sin ONNX → hash; sin Qdrant → catálogo
  completo. Docs base de arquitectura, pipeline, embeddings, Qdrant, API,
  configuración y operación.

---

> Este registro reemplaza el antiguo "plan de implementación" del espacio latente
> directo: ese plan quedó **implementado** en `323a1e8` y la descripción de cómo
> funciona hoy está en [pipeline-prediccion.md](pipeline-prediccion.md).
