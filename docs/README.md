# remtk-prediction — Documentación

Documentación de funcionamiento del servicio **remtk-prediction**: predicción de
herramientas MCP y memorias contextuales en TypeScript, migrado 1:1 desde el
server Go legacy. Un solo proceso Node con **3 listeners HTTP** (apps Nest
autocontenidas) que empaqueta predicción, modelos de embeddings ONNX y la capa de
recall sobre Qdrant. Desde la última actualización, la reducción de herramientas
usa además un **enrutador topológico** en espacio latente (grafo por tenant +
estado de sesión), manteniendo el fallback plano por keywords.

```
┌──────────────────────────────────────────────────────────────┐
│                      PREDICTION (TS)                          │
│                                                              │
│  Puerto 6776  API de Predicción                              │
│    /predict · /tools · /tools/count · /memory/predict        │
│    /health · /debug                                          │
│                                                              │
│  Puerto 6777  API de Modelos (e5-small ONNX)                 │
│    POST /embed · GET /models · /health                       │
│                                                              │
│  Puerto 6775  API de Qdrant (BM25, keywording, indexado)     │
│    /tools/upsert · /tools/search · /memories/search          │
│    /tools/synonyms · /status · /health                       │
└──────────────────────────────────────────────────────────────┘
```

## Índice

| Documento | Contenido |
|---|---|
| [arquitectura.md](arquitectura.md) | Visión de conjunto: proceso único, módulos, doble uso (librería y servidor), mapa de ficheros. |
| [pipeline-prediccion.md](pipeline-prediccion.md) | Cómo funciona la predicción: enrutador topológico (grafo) y fallback plano de 3 capas para tools, y flujo de memorias con topic-shift. |
| [juicio.md](juicio.md) | Capa de juicio (opcional): NLI de estado, abstención `__NOOP__`, puerta de coherencia, cross-encoder, MaxSim e índice dual — emite un veredicto, no solo similitud. |
| [embeddings.md](embeddings.md) | `EmbeddingEngineService`: ONNX e5-small (default, 384 dims) y e5-large (bajo demanda, 1024), prefijos `query:`/`passage:`, cola con prioridad, dedupe y fallback hash. |
| [qdrant.md](qdrant.md) | Integración con Qdrant: colecciones, BM25 sparse (djb2 client-side), IDs uuidv5, sinónimos, RAKE y blend 60/40. |
| [api-http.md](api-http.md) | Referencia de endpoints HTTP de los puertos 6776, 6777 y 6775. |
| [entrenamiento-prediccion.md](entrenamiento-prediccion.md) | **Propuesta**: capa de aprendizaje léxico incremental por canal (scope), disparada al registrar herramientas y aislada por usuario. |
| [configuracion.md](configuracion.md) | Variables de entorno (`AppConfig` y `LoggerConfig`), targets `RUST_LOG` y resolución de modelos. |
| [operacion.md](operacion.md) | Arranque, Docker, tests, logger dual (`src/shared/logger/mod.rs`) y trazas de observabilidad (`[trace:*]`, `[profile:*]`, `[http:*]`). |
| [diagnostico-fallo-cron-jobs.md](diagnostico-fallo-cron-jobs.md) | **Diagnóstico forense**: análisis de causa raíz de fallos de predicción en herramientas `scheduler` (`schedule_task` y `schedule_list`) y planificador V8. |
| [update.md](update.md) | Registro de las actualizaciones recientes (logger estructurado en Rust, diagnóstico forense de cron jobs, monolito modular). |

## En 30 segundos

1. Se arranca un único proceso (`node dist/src/server.js`) que abre 3 puertos.
2. El monolith NestJS (u otro consumidor) llama a `POST http://localhost:6776/predict`
   con `{ sessionId, tenant, text }`.
3. La predicción clasifica el turno, mantiene un **estado latente de sesión**
   (`z_t`, blend 70/30 cuando no hay cambio de tema) y puntúa el catálogo contra
   el **grafo precomputado** del tenant (embeddings `passage:` de cada firma +
   aristas de dependencias): refuerzo BM25 de Qdrant, propagación de
   pre-requisitos, umbral adaptativo, resolución de mutexes y **orden
   topológico**.
4. Si el tenant aún no tiene grafo, degrada a la reducción plana por keywords
   cross-idioma (capas 1–2) + confirmación léxica BM25 + umbral adaptativo.
5. Si ONNX o Qdrant no están disponibles, el sistema **degrada** (hash + catálogo
   completo) en lugar de fallar.
