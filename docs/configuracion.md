# Configuración

Toda la configuración se lee de **variables de entorno** en `src/config.ts`
(`loadConfig(env?)`). Si una variable falta o está vacía, se usa su **default**.
Los helpers `int/float/bool` validan y caen al default si el valor no es numérico
/ booleano válido.

## Variables y valores por defecto

| Variable | Default | Descripción |
|---|---|---|
| `PORT_PREDICT` (o `PORT`) | `6776` | Puerto del API de predicción. |
| `PORT_EMBED` | `6777` | Puerto del API de modelos. |
| `PORT_QDRANT` | `6775` | Puerto del API de Qdrant. |
| `ONNX_ENABLED` | `true` | Cargar modelos ONNX. `false` → fallback hash siempre. |
| `ONNX_MODELS_PATH` | resolución automática | Carpeta con los modelos ONNX (ver árbol abajo). |
| `KEYWORD_BOOST` | `0.15` | Refuerzo sumado por confirmación léxica BM25 en la fusión (capa 3). |
| `KEYWORD_TOP_K` | `20` | Tamaño del top-K tras la reducción cross-idioma (capa 2). |
| `ONNX_ADAPTIVE_MIN_TOOLS` | `2` | Mínimo de tools garantizadas por el umbral adaptativo. |
| `ONNX_ADAPTIVE_MAX_TOOLS` | `30` | Techo de exploración del gap adaptativo. |
| `ONNX_ADAPTIVE_GAP_THRESHOLD` | `0.03` | Gap natural para cortar el ranking (escala coseno e5-small). |
| `ONNX_ADAPTIVE_MIN_SCORE` | `0.8` | Piso de relevancia: si el mejor coseno no lo supera → 0 tools. |
| `QDRANT_ENABLED` | `true` | Habilitar la capa Qdrant (recall BM25). |
| `QDRANT_URL` | `http://localhost:6333` | URL del Qdrant externo. |
| `QDRANT_API_KEY` | *(vacío)* | API key opcional (cabecera `api-key`). |
| `RECALL_LIMIT` | `50` | Límite de recall BM25 usado por el orquestador en `/predict`. |
| `PREDICT_WARMUP_WAIT_MS` | `15000` | Tiempo máximo que `/predict` espera el warm-up en vuelo (embeddings + grafo) de un catálogo recién registrado antes de degradar a la ruta plana. Rango permitido `0–60000`; `0` → nunca espera. El consumidor bien comportado llama `GET /tools/ready` después de `POST /tools`, que no tiene esta cota. |
| `MAX_OUTPUT_TOOLS` | `50` | *(Opcional)* Tope final de tools de salida tras el umbral adaptativo. Rango permitido `0–50`; `0` → nunca devolver herramientas. |
| `LOGS_DIR` | `<cwd>/logs` | *(Opcional)* Carpeta de logs diarios `prediction-YYYY-MM-DD.log`. **No pasa por `loadConfig`**: el logger la lee directo en `src/logger.ts`. |

> **Colecciones de Qdrant fijas.** Los nombres `mcp_tools`, `tool_keywords`,
> `query_synonyms` y `contextual_memories` ya **no** son configurables por
> entorno: viven como constantes en `shared/constants/qdrant/general.constant.ts`.

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
