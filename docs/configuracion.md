# Configuración

Toda la configuración del servidor en Rust nativo se lee desde **variables de entorno** en Dos estructuras complementarias al arrancar (`src/main.rs`):

1. **`LoggerConfig::from_env()` (`src/shared/logger/mod.rs`)**: configura la observabilidad estructurada dual (`stdout` + archivo rotativo diario `logs/prediction-YYYY-MM-DD.log`).
2. **`AppConfig::from_env()` (`src/shared/config/mod.rs`)**: carga y valida con fallbacks tipados los puertos TCP, rutas de modelos ONNX, hilos de inferencia y umbrales matemáticos del pipeline.

## 1. Variables de Entorno de Negocio e Inferencia (`AppConfig`)

| Variable | Default (`AppConfig`) | Descripción |
|---|---|---|
| `PORT_PREDICT` | `6776` (`6876` en contenedor host) | Puerto HTTP del servicio de predicción de herramientas y juicio MoE. |
| `PORT_EMBED` | `6777` (`6877` en contenedor host) | Puerto HTTP del servicio de embeddings vectoriales (`e5-small`). |
| `PORT_QDRANT` | `6775` (`6875` en contenedor host) | Puerto HTTP del servicio de recall e indexación de catálogo. |
| `MODELS_PATH` / `ONNX_MODELS_PATH` | `models` (o `../models`) | Ruta al directorio que contiene `multilingual-e5-small-onnx` y `remtk-juicio-moe-int8`. |
| `ONNX_MODEL_SIZE` | `small` | Tamaño del modelo ONNX por defecto (`small`, 384 dimensiones). |
| `ONNX_INTRA_OP_THREADS` | `4` | Hilos intra-op de ONNX Runtime para aceleración SIMD AVX2. |
| `ONNX_INTER_OP_THREADS` | `1` | Hilos inter-op de ONNX Runtime. |
| `ONNX_ADAPTIVE_MIN_SCORE` | `0.84` | Piso mínimo de relevancia (`adaptive_min_score`): ninguna herramienta con score inferior es seleccionada. |
| `ONNX_ADAPTIVE_GAP_THRESHOLD` | `0.03` | Umbral de brecha natural (`gap_threshold`) entre candidatos sucesivos. |
| `ABSTENCION_THRESHOLD` | `0.52` | Umbral de probabilidad de abstención del cabezal MoE para retorno nulo (`0 tools`). |
| `KEYWORD_BOOST` | `0.05` | Refuerzo por coincidencia de palabras clave internas. |
| `NAME_AFFINITY_BOOST` | `0.08` | Refuerzo por afinidad léxica en el nombre de la herramienta (con tope acumulado de `0.15`). |
| `FAMILY_GATE_PENALTY` | `0.15` | Penalización por familia o namespace ajeno. |
| `RECALL_LIMIT` | `20` | Límite máximo de candidatas recuperadas en la fase de recall (`ToolRegistry::retrieve_candidates`). |
| `MAX_OUTPUT_TOOLS` | `8` | Tope máximo de herramientas devueltas en la respuesta final de `/predict`. |
| `MAX_CATEGORIES` | `6` | Tope máximo de categorías activas evaluadas en el juicio. |

---

## 2. Variables de Observabilidad y Logger Dual (`LoggerConfig`)

Definidas en `src/shared/logger/mod.rs` e inicializadas mediante `init_logger()` al inicio de `src/main.rs`:

| Variable | Default (`LoggerConfig`) | Descripción |
|---|---|---|
| `LOGS_DIR` | `logs` (`/app/logs` en Docker) | Directorio donde `DualMakeWriter` crea y rota automáticamente el archivo diario `prediction-YYYY-MM-DD.log` (fecha UTC calculada con el algoritmo civil de Howard Hinnant). |
| `LOG_TO_FILE` | `true` | Habilita la escritura simultánea en archivo rotativo diario además de `stdout`. Desactivar con `false` o `0`. |
| `LOG_FORMAT` | `text` | Formato de salida: texto estructurado legible por defecto, o JSON línea por línea si `LOG_FORMAT=json`. |
| `RUST_LOG` | `info,remtk_prediction_server=debug,profiler=info,prediction_trace=info,http_trace=info` | Filtro `tracing_subscriber::EnvFilter` por nivel y target de instrumentación. |

### Targets de Trazabilidad (`RUST_LOG`)

| Target | Nivel | Prefijos Emitidos | Propósito |
|---|---|---|---|
| `prediction_trace` | `INFO` / `WARN` | `[trace:recall]`, `[trace:moe]`, `[trace:predict]`, `[trace:route]`, `[trace:memory_judge]`, `[trace:qdrant_search]`, `[trace:spatial]` | Trazabilidad forense de cada etapa del pipeline: términos de recall, probabilidades NLI/abstención/expertos del MoE, herramientas seleccionadas (`selected`), `top_candidates` y herramientas descartadas (`filtered_out`) con sus scores exactos, así como alertas `COLD_CATALOG`. |
| `profiler` | `INFO` | `[profile:predict]`, `[profile:embed]`, `[profile:qdrant_upsert]`, `[profile:interest]`, `[profile:history_regime]` | Desglose milimétrico de tiempos de cómputo en CPU (`recall`, `q_emb`, `query_tokens`, `ctx_tokens`, `assemble`, `moe`). |
| `http_trace` | `INFO` | `[http:predict]`, `[http:spatial_predict]`, `[http:predict_route]`, `[http:history_regime]` | Resumen de entrada/salida HTTP en la capa Axum (`status`, `tenant`, `session`, `elapsed`, `tools_out`, `complexity`, `nli`). |
| `remtk_prediction_server` | `INFO` / `DEBUG` | `boot ...`, `API Predict: Registradas ...` | Eventos de ciclo de vida del servidor, carga `mmap` de sesiones ONNX y apagado elegante (`SIGINT`/`SIGTERM`). |

### Capa de juicio (opcional)

Todas tienen default en `shared/constants/juicio/juicio.constants.ts`. Ver
[juicio.md](juicio.md) para qué hace cada señal y su calibración medida.

| Variable | Default | Descripción |
|---|---|---|
| `JUICIO_ENABLED` | `true` | Interruptor maestro. `false` → `pre/post` son pasa-through. |
| `JUICIO_NLI_MARGIN` | `0.03` | Margen mínimo entre pools rechazo/confirmación del NLI. |
| `JUICIO_NLI_MODEL_PATH` | *(vacío)* | Carpeta del cross-encoder NLI ONNX. Vacío → fallback por arquetipos. |
| `JUICIO_NOOP_MARGIN` | `0.1` | Margen del prototipo `__NOOP__` sobre el mejor score de herramienta. |
| `JUICIO_ENERGY_T` | `0.05` | Temperatura T de la energía libre E = −T·ln Σ exp(sim/T). |
| `JUICIO_ENERGY_TAU` | `0` | Umbral de energía. `0` = desactivada (la banda plana de e5-small no la separa). |
| `JUICIO_GATE_BETA` | `8` | β (pendiente) de la puerta de coherencia `λ = σ(cos·β − γ)`. |
| `JUICIO_GATE_GAMMA` | `5` | γ (desplazamiento) de la puerta de coherencia. |
| `JUICIO_GATE_TEXT_MIN` | `0.5` | λ mínima para concatenar el historial al prompt. |
| `JUICIO_RERANKER_MODEL_PATH` | *(vacío)* | Carpeta del cross-encoder reranker ONNX (componente 4). |
| `JUICIO_RERANKER_ALPHA` | `0.6` | Peso del logit del reranker en la fusión. |
| `JUICIO_MAXSIM_WEIGHT` | `0.3` | Peso del MaxSim token-level en la fusión del re-rank. |
| `JUICIO_MAXSIM_MAX_TOKENS` | `48` | Tope de tokens por texto en el MaxSim (cota O(m·n)). |
| `JUICIO_SPECIFICITY_RHO` | `0.5` | ρ de la penalización por difusión semántica. |
| `JUICIO_PROBLEM_MARGIN` | `0.08` | Margen que debe superar el `problemSpace` para reemplazar al score fusionado. |
| `JUICIO_POST_TOP_K` | `10` | Tope de candidatas re-puntuadas en el post de juicio. |

> **Modelo por defecto `small`.** No existe variable que cambie el modelo del
> pipeline: `onnxModelSize` es siempre `"small"` (e5-small, 384 dims). El motor
> (`EmbeddingEngineService`) también soporta `"large"` (e5-large, 1024 dims)
> **bajo demanda** con `embed(text, "large")`, sin calentarlo en el boot. El
> modelo `multilingual-bge-m3-int8-onnx` de `./models` solo lo usan los tests de
> benchmark (`pnpm test:bge`); no es parte del engine ni del pipeline.

## Resolución de la ruta de modelos (`resolveModelsPath`)

Orden de búsqueda:

1. `ONNX_MODELS_PATH` explícita.
2. Rutas relativas al paquete según dónde se ejecute:
   - desde `src/…` → `../models`
   - desde `dist/src/…` → `../../models`
3. Fallbacks `./models` y `../models` (según el CWD).
4. Si nada existe, usa la primera ruta del paquete (la carga fallará y se
   degradará a hash con un warning).

La estructura esperada dentro de la carpeta (cada subcarpeta lleva su
`tokenizer.json` y `tokenizer_config.json`):

```
models/
├── multilingual-e5-small-onnx/     model.onnx                — default del pipeline
├── multilingual-e5-large-onnx/     model.onnx + model.onnx_data — bajo demanda
└── multilingual-bge-m3-int8-onnx/  model_quantized.onnx      — solo benchmark (test:bge)
```

Si `model.onnx` de un e5 falta, el warmup registra `modelo no encontrado` y el
sistema usa el fallback hash (no falla). BGE-M3 int8 no lo lee el engine: existe
en disco únicamente para `tests/embed-bge.test.ts`.

## Configuración típica

```bash
# Ejemplo: Qdrant local sin API key, modelos en ./models
PORT_PREDICT=6776
PORT_EMBED=6777
PORT_QDRANT=6775
ONNX_ENABLED=true
ONNX_MODELS_PATH=/app/models
QDRANT_URL=http://localhost:6333
# Opcional: tope final de tools de salida (rango 0-50, default 50).
# MAX_OUTPUT_TOOLS=50
```

## Valores clave del pipeline (constantes en código)

Además de la config por entorno, el código define constantes internas:

| Constante | Valor | Dónde |
|---|---|---|
| `MAX_HISTORY_MESSAGES` | 4 | Mensajes previos incorporados en `new_query`. |
| `MAX_MESSAGE_CHARS` | 300 | Recorte por mensaje de historial. |
| `TOPIC_SHIFT_THRESHOLD` | 0.86 | Umbral de cambio de tema en memorias y en el estado latente de sesión. |
| `CLASSIFICATION_MARGIN` | 0.03 | Margen del clasificador de turno. |
| `NUANCE_THRESHOLD` | 0.55 | Separación empty/nuance en confirmaciones. |
| `onnxMaxTokens` | 512 | Truncación del tokenizer en el engine (`shared/constants/predict/embedding.constants.ts`). |
| `graphPropagationAlphaDefault` | 0.2 | Alpha de propagación de pre-requisitos: S_propagado = S + α·Aᵀ·S (`general.predict.ts`). |
| Pesos de aristas del grafo | 0.85 / 0.4 / 1.0 | `PREREQUISITE` / `CO_OCCURRENCE` / `MUTUALLY_EXCLUSIVE` (`graph-cache.service.ts`). |
| `BLEND_PREV` / `BLEND_NEW` | 0.7 / 0.3 | Combinación convexa 70/30 del estado latente de sesión z_t (`session-state-cache.service.ts`). |
| `REQ_TIMEOUT_MS` / `REQ_RETRIES` | 15000 / 2 | Timeout y reintentos del cliente Qdrant. |
| `MAX_SEARCH_LIMIT` | 50 | Tope de resultados por búsqueda BM25. |
| `MAX_KEYWORDS` | 8 | Keywords RAKE máximas por query. |
| `MAX_SYNONYM_EXPANSIONS` | 8 | Sinónimos nuevos máximos por expansión. |
| `MAX_TRACES` | 200 | Trazas conservadas por el Debugger. |
| `toolsCollectionDefault` / `keywordsCollectionDefault` / `synonymsCollectionDefault` / `memoriesCollectionDefault` | `mcp_tools` / `tool_keywords` / `query_synonyms` / `contextual_memories` | Colecciones Qdrant fijas (`shared/constants/qdrant/general.constant.ts`). |

> Los defaults de entorno (`KEYWORD_BOOST`…`RECALL_LIMIT`) se aplican en
> `loadConfig`; las constantes de arriba viven hardcodeadas en cada módulo.
