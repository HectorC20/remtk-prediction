# Arquitectura

## Visión de conjunto

`remtk-prediction` es un **servicio unificado en un solo proceso Node**. Sustituye
al antiguo server Go y a los tres servicios separados del monolith (embeddings,
Qdrant y predicción) manteniendo **contratos HTTP 1:1**. Puede ejecutarse de dos
formas:

- **Como librería in-process** (sin abrir puertos): `import { createSystem } from "remtk-prediction"`.
- **Como servidor HTTP** de 3 puertos: `node dist/src/server.js` (entry `src/server.ts`).

```
                  ┌────────────────────────────────────────────────────────┐
   NestJS/otros   │                     Proceso Node                       │
   (consumidores) │                                                        │
       │  HTTP    │   ┌──────────────────────────────────────────────┐     │
       ├─────────►│   │  createSystem(cfg)  (src/main.ts)            │     │
       │          │   │                                              │     │
       │          │   │  EmbeddingEngineService (e5-small / large)   │    │  ┌─────────────┐
       │          │   │  QdrantService (BM25 sparse) ────────────────┼───►│ Qdrant HTTP  │
       │          │   │  TurnClassifier · CrossEncoderService        │    │ (6333)       │
       │          │   │  KeywordService · LexicalProfileService      │    └─────────────┘
       │          │   │  ToolGraphCacheService (grafo DAG)           │
       │          │   │  SessionStateCacheService (z_t blend 70/30)  │
       │          │   │  RerankService (fusión BM25 + topo Kahn)     │
       │          │   │  JuicioService (NLI, NOOP, MaxSim, CtxGate)  │
       │          │   │  SkillMemoryService · SpatialPredictService  │
       │          │   │  VarianteCService · ConfirmationCache        │
       │          │   │  Debugger · PredictionOrchestrator           │
       │          │   └──────────────────────────────────────────────┘     │
       │          │                                                        │
       │          │  Express 6776 (predicción)  PredictAppModule           │
       │          │  Express 6777 (modelos)     EmbeddingAppModule         │
       │          │  Express 6775 (Qdrant API)  QdrantAppModule            │
       └──────────┼────────────────────────────────────────────────────────┘
```

## Puntos de entrada

| Archivo | Rol |
|---|---|
| `src/index.ts` | API pública de librería (in-process). Exporta `createSystem`, `loadConfig`, servicios núcleo, adaptadores y tipos de contrato. |
| `src/main.ts` | Composition Root (`createSystem(cfg)`): construye el grafo de dependencias de todos los servicios. Define la interfaz `System`. |
| `src/server.ts` | Entry del servidor de producción y desarrollo: carga configuración y arranca en paralelo los 3 listeners HTTP con apagado limpio (`SIGINT`/`SIGTERM`). |
| `src/app.module.ts` | Bootstrap NestJS: `startPredictServer` (6776), `startEmbedServer` (6777) y `startQdrantServer` (6775) con parser `express.json({ limit: "4mb" })`. |
| `src/config.ts` | `loadConfig(env?)`: carga, normaliza y valida fuertemente tipadas las variables de entorno (`AppConfig`). |
| `src/logger.ts` | Logger estructurado dual con timestamp ISO (`log`/`warn`): escribe simultáneamente a stdout y a un archivo rotativo diario (`logs/prediction-YYYY-MM-DD.log` o `LOGS_DIR`). |
| `src/shared/scope.ts` | Utilidades de scope y multitenancy: resolución canónica de identificadores de tenant, agente (`resolveScopeKey`) y parseo seguro (`parseKeyRemtk`). |
| `src/shared/interfaces/` | Contratos TypeScript de dominio, grafo DAG, juicio neuronal, memorias, predicción y mensajería. |
| `src/shared/constants/` | Constantes globales fijas de puertos, colecciones Qdrant, umbrales de juicio, calibración, dimensiones y herramientas espaciales. |

## Grafo de dependencias (`src/main.ts`)

```
cfg ──► EmbeddingEngineService ──────────┐
 │                                       ├─► CrossEncoderService ─► TurnClassifier
 │     QdrantService ────────────────────┤
 │      (asegura colecciones)            ├─► KeywordService, LexicalProfileService
 │                                       ├─► ToolGraphCacheService, SessionStateCacheService
 │                                       ├─► JuicioService (EstadoNLI, Abstención,
 │                                       │   PuertaContexto, MaxSim, Especificidad)
 │                                       ├─► RerankService, VarianteCService
 │                                       ├─► SkillMemoryService, SpatialPredictService
 │                                       └─► ConfirmationCache, Debugger
 │                                                   │
 └───────────────────────────────────────────────────┴─► PredictionOrchestrator
```

`createSystem` construye los componentes en estricto orden de dependencia:
1. `engine` (`EmbeddingEngineService`): lanza `warmup()` asíncrono para precargar la sesión ONNX del modelo `small`.
2. `qdrant` (`QdrantService`): ejecuta `ensureCollections()` si Qdrant está activo.
3. Servicios auxiliares y modelos profundos: `crossEncoder`, `classifier`, `confirmCache`, `keywords`, `lexical`, `graphCache`, `sessionState`, `debugger_`, `rerank`, `varianteC`.
4. Capa de juicio cognitivo (`JuicioService`): integra `EstadoNliService`, `AbstencionService`, `PuertaContextoService`, `MaxSimService`, `EspecificidadService` y `CrossEncoderService`.
5. Orquestador (`PredictionOrchestrator`): ensamblado mediante `createOrchestrator` con todos los servicios inyectados.
6. `skillMemory` (`SkillMemoryService`): memoria de destrezas y parámetros anclados.

`PredictionOrchestrator` expone la superficie principal de predicción:
- `predict(input)`: pipeline completo de selección de herramientas; devuelve `GraphPredictionResult` (herramientas seleccionadas + bloque `graph` con `nodes`, `edges` y `executionOrder` topológico).
- `predictMemory(input)`: pipeline de memorias contextuales con detección de topic-shift.
- `registerTools(tenant, tools)`: registro e indexación; además de keywords y Qdrant, **reconstruye el grafo topológico DAG del tenant** (`graphCache.buildTenantGraph`).
- `countTools(tenant)`: conteo de herramientas registradas por tenant.

---

## Estructura global del repositorio

El repositorio se divide en módulos funcionales y directorios de soporte:

```
remtk-prediction/
├── bench/         # Benchmarks de rendimiento, precisión de intención y pruebas de estrés
├── docs/          # Documentación técnica exhaustiva por subsistema
├── models/        # Modelos neuronales ONNX (e5-small, e5-large) y tokenizers
├── scripts/       # Automatizaciones de verificación de infraestructura, docker y métricas
├── src/           # Código fuente modular en TypeScript (librería y servidor)
├── tests/         # Batería de pruebas unitarias, grafos DAG, e2e, calibración y casos adversariales
├── logs/          # Archivos de log diarios generados en runtime (prediction-YYYY-MM-DD.log)
├── AGENT.md       # Directiva arquitectónica obligatoria (prohibición de parches léxicos)
├── package.json   # Dependencias, scripts de prueba/benchmark y metadatos
├── tsconfig.json  # Configuración del compilador TypeScript
└── README.md      # Resumen general del servicio, arranque, contratos HTTP y uso como librería
```

| Directorio | Propósito y Contenido |
|---|---|
| `src/` | Núcleo de la aplicación: controladores, módulos NestJS, servicios de espacio latente, juicio neuronal y utilidades compartidas. |
| `bench/` | Suites de benchmark y análisis empírico (`run.ts`, `cases.ts`, `intent-complex.ts`, `juicio-stress.ts`, `stock-stress.ts`, comparativas bi-encoder vs cross-encoder y resultados JSON históricos en `results/`). |
| `scripts/` | Automatizaciones de ciclo de vida: `ensure-qdrant.mjs` (arranque automático de contenedor Docker), `ensure-onnx-models.mjs`, `bench-embeddings.ts`, scripts de medición de catálogos y manifold (`_measure-*.ts`). |
| `tests/` | Batería de pruebas exhaustivas: unitarias (`rerank.test.ts`, `graph.test.ts`), integración E2E (`e2e.test.ts`), pruebas de calibración y flujo continuo (`continuous-flow-intent.test.ts`, `calibration-context.test.ts`), escenarios adversariales (`casos-estres-adversariales.test.ts`, `retorno-nulo-escenarios.test.ts`) y fixtures de catálogos reales (`mitumbes-production-tools.ts`). |
| `models/` | Binarios ONNX optimizados (INT8) para inferencia CPU local: `multilingual-e5-small` (384 dims) y `multilingual-e5-large` (1024 dims) junto a sus archivos `tokenizer.json`. |
| `docs/` | Documentación técnica detallada del sistema: arquitectura, pipeline, juicio cognitivo, embeddings, Qdrant, API HTTP, operación, changelog y notas de competencia. |
| `logs/` | Directorio de salida del logger donde se persisten los eventos diarios (`prediction-YYYY-MM-DD.log`). |

---

## Mapa detallado por directorio (`src/`)

```
src/
├── app.module.ts              Bootstrap modular NestJS (instancia listeners en 6776, 6777 y 6775)
├── config.ts                  Carga, parseo y validación tipada de variables de entorno (AppConfig)
├── index.ts                   API pública exportada para uso in-process como librería
├── logger.ts                  Logger estructurado dual (consola + rotación diaria en logs/)
├── main.ts                    Composition Root (createSystem): instanciación y cableado de dependencias
├── server.ts                  Entry ejecutable del servidor HTTP (arranca los 3 listeners en paralelo)
│
├── embedding/                 Subsistema de Modelos de Lenguaje & Embeddings (Puerto 6777)
│   ├── embedding.controller.ts  API REST 6777 (POST /embed, GET /models, GET /health)
│   ├── embedding.module.ts      Módulo NestJS de embeddings (EmbeddingAppModule)
│   └── embedding.service.ts     EmbeddingEngineService: motor ONNX e5-small (384d, default) y
│                                e5-large (1024d), colas con prioridad, deduplicación y hash fallback
│
├── juicio/                    Capa Cognitiva de Juicio y Verificación Neuronal (docs/juicio.md)
│   ├── juicio.service.ts        JuicioService: árbitro pre y post pipeline (decisiones de veredicto)
│   └── services/
│       ├── abstencion.service.ts      Abstención __NOOP__ y filtro de energía para descartar small-talk
│       ├── cross-encoder.service.ts   Atención cruzada token a token (polaridad, negaciones y núcleo ejecutivo)
│       ├── especificidad.service.ts   Evaluador de especificidad semántica respecto al catálogo de tools
│       ├── estado-nli.service.ts      NLI de estado: evalúa hipótesis sobre propuestas pendientes
│       ├── maxsim.service.ts          Similitud fina de interacción tardía token a token (estilo ColBERT)
│       └── puerta-contexto.service.ts Puerta de coherencia y relevancia contextual
│
├── predict/                   Pipeline de Predicción de Herramientas y Memorias (Puerto 6776)
│   ├── orchestrator.ts          PredictionOrchestrator: orquesta etapa A (categorías), etapa B (decisor) y memorias
│   ├── predict.controller.ts    API REST 6776 (/predict, /tools, /tools/count, /memory/predict, /health, /debug)
│   ├── predict.module.ts        Módulo NestJS de predicción (PredictAppModule)
│   ├── turn-classifier.ts       Clasificador de turno conversacional (new_query, confirmation_empty, nuance)
│   ├── keywords.ts              Extracción, tokenización y normalización léxica
│   ├── helper/
│   │   └── debugger.helper.ts   Buffer circular de trazas y telemetría de decisiones (/debug)
│   └── services/
│       ├── calibration.service.ts         Calibración dinámica de umbrales y márgenes de confianza
│       ├── confirm-cache.service.ts       Caché de planes previos por sesión (early-exit en confirmaciones)
│       ├── graph-cache.service.ts         ToolGraphCacheService: grafo DAG por tenant (prerequisitos, mutexes, adyacencia)
│       ├── keyword.service.ts             KeywordService: reducción cross-idioma léxica (capas 1-2 fallback)
│       ├── lexical-profile.service.ts     LexicalProfileService: perfiles léxicos adaptativos por canal/scope
│       ├── remtk-memory-skills.adapter.ts Adaptador de integración con el almacén externo remtk-memory
│       ├── rerank.service.ts              RerankService: fusión BM25 + dense, umbral adaptativo y orden topológico (Kahn)
│       ├── session-state-cache.service.ts SessionStateCacheService: estado latente z_t por sesión (blend 70/30)
│       ├── skill-memory.service.ts        SkillMemoryService: persistencia y anclaje semántico de parámetros para skills
│       ├── spatial-predict.service.ts     SpatialPredictService: enrutamiento y predicción para herramientas visuales/espaciales
│       └── variante-c.service.ts          VarianteCService: pipeline alternativo de autonomía de herramientas
│
├── qdrant/                    Integración con Motor Vectorial Qdrant (Puerto 6775)
│   ├── qdrant.controller.ts     API REST 6775 (/tools/upsert, /tools/search, /memories/search, /tools/synonyms, /status)
│   ├── qdrant.module.ts         Módulo NestJS para Qdrant (QdrantAppModule)
│   ├── qdrant-service.ts        Orquestador de negocio Qdrant (BM25 sparse, RAKE, blend 60/40 y catálogo por tenant)
│   └── qdrant-client.ts         Cliente HTTP nativo: cálculo de hash djb2 e IDs deterministas uuidv5
│
└── shared/                    Contratos transversales, constantes y utilidades
    ├── constants/
    │   ├── index.ts             Re-exports centralizados de constantes
    │   ├── juicio/              Constantes de umbrales, márgenes y arquetipos del subsistema de juicio
    │   ├── ports/               Puertos TCP por defecto (6776, 6777, 6775)
    │   ├── predict/             Parámetros de calibración, límites, timeouts, dimensiones y versión
    │   ├── qdrant/              Nombres de colecciones fijas y dimensiones vectoriales
    │   └── spatial/             Identificadores y configuración de herramientas espaciales
    ├── dictionary/
    │   ├── stopwords.dictionary.ts Stopwords multilenguaje (español, inglés, portugués)
    │   └── turn.dictionary.ts      Tipos y arquetipos de turnos conversacionales
    ├── factory/
    │   └── orchestrator.factory.ts Fábrica constructora de PredictionOrchestrator
    ├── interfaces/              Contratos TypeScript tipados: calibration, chat, classifier, domain,
    │                            embedding, graph, juicio, keyword, lexical, memory, orchestrator,
    │                            qdrant, rerank, session, skill, spatial, system, topic
    └── scope.ts                 Parseo de claves multitenant y resolución de scope de agentes

## Servidores HTTP (ver [api-http.md](api-http.md))

Cada puerto es una **app Nest autocontenida** arrancada por `src/app.module.ts`
con el parser JSON `express.json({ limit: "4mb" })` (constante `LIMIT` en
`shared/constants/predict/data.predict.ts`):

| Puerto | App (arranque) | Roles |
|---|---|---|
| `6776` | `PredictAppModule` (`startPredictServer`) | Orquesta predicción y registro de tools; expone `/health`, `/debug`. |
| `6777` | `EmbeddingAppModule` (`startEmbedServer`) | Expone embeddings del modelo por defecto e5-small; `/models`, `/health`. |
| `6775` | `QdrantAppModule` (`startQdrantServer`) | Capa HTTP sobre el motor Qdrant; `/status`, `/health`. |

`server.ts` arranca los tres listeners en paralelo y registra apagado limpio con
`SIGINT`/`SIGTERM`.

## Arranque del proceso

```ts
const cfg = loadConfig();                       // env vars → defaults listos
const system = await createSystem(cfg);         // warmup ONNX + ensure colecciones
startPredictServer(...); startEmbedServer(...); startQdrantServer(...);
```

Durante `createSystem`:

1. `engine.warmup()` lanza en background la carga de la sesión ONNX de `small`
   (no bloquea). El clasificador carga sus arquetipos de forma perezosa en el
   primer `classify`.
2. Si `qdrantEnabled`, `qdrant.ensureCollections()` crea las 4 colecciones si no
   existen.

El grafo de cada tenant **no** se construye en el boot: se arma en `POST /tools`
(`registerTools`), cuando llega el catálogo.

## Degradación (principio de diseño)

Ninguna dependencia externa es crítica:

| Fallo | Comportamiento |
|---|---|
| ONNX no disponible / modelo incompleto | Embeddings por **hash determinístico** (djb2), `model = "hash"`. |
| Qdrant caído o deshabilitado | Recall BM25 vacío → la puntuación queda solo semántica (grafo o keywords). |
| Grafo del tenant ausente (tenant no registrado vía `POST /tools`) | Reduce **plano** por keywords cross-idioma sobre el catálogo completo. |
| Keyword reduce falla | Se usa el catálogo completo como candidatas. |
| Estado de sesión (`z_t`) falla | Se salta la ruta de grafo y se degrada a la reducción plana. |
| Juicio deshabilitado o motor en fallback hash | `JuicioService` opera como pass-through neutro (evaluación transparente sin vetos de abstención). |

Esto replica el comportamiento "soft-degrade" del sistema original: **el servicio
nunca falla por ausencia de modelos, de Qdrant o del grafo**.
