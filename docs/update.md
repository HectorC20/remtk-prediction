# Registro de actualizaciones

Bitácora de los cambios relevantes de **remtk-prediction**. Las entradas más
recientes van primero. Cada una referencia el hash de git, el alcance del cambio
y la documentación técnica que lo describe.

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
