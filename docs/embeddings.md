# Servicio de embeddings (`EmbeddingEngineService`)

`src/embedding/embedding.service.ts` empaqueta el motor de embeddings del sistema
(antes `embedding-engine.ts`). Usa **onnxruntime-node** (CPU) +
**@huggingface/tokenizers**.

## Modelos

| Propiedad | Valor |
|---|---|
| Modelo por defecto | `multilingual-e5-small-onnx` (`small`, default del pipeline) |
| Modelo bajo demanda | `multilingual-e5-large-onnx` (`large`, soportado por el motor: `embed(text, "large")` lo carga al pedirlo) |
| Dimensiones | **384** (`small`) / **1024** (`large`) — `onnxEmbedDim` |
| Runtime | ONNX Runtime, `executionProviders: ["cpu"]` |
| Truncación | 512 tokens (`onnxMaxTokens`) |
| Pooling | mean-pooling por attention mask sobre `[1, seq, dim]` |
| Normalización | L2 |
| Entradas del sesión | `input_ids`, `attention_mask`, `token_type_ids` (int64) |
| Salida usada | `sentence_embedding` (o `last_hidden_state` si no existe) |
| Archivos en disco | `multilingual-e5-{size}-onnx/` con `model.onnx`, `tokenizer.json`, `tokenizer_config.json` |

> El pipeline y el API HTTP (`POST /embed`) trabajan con `small`. El tamaño
> `large` está soportado por el motor (`embed(text, "large")`) pero no se
> calienta en el boot para no penalizar el arranque. El modelo
> `multilingual-bge-m3-int8-onnx` de `./models` es **externo al engine**: solo lo
> usan los tests de benchmark (`pnpm test:bge`), nunca el sistema en runtime.

## Prefijos del contrato e5

El caller aplica el prefijo según el rol del texto (regla del modelo e5). El
motor expone dos helpers que lo hacen por él:

- `embedQuery(text, size?, opts?)` → `query: {text}` — prompts de predicción.
- `embedPassage(text, size?, opts?)` → `passage: {text}` — contenido indexable
  (tools/memorias/nodos del grafo).

Ambos aceptan `{ high?: boolean }` para la cola de prioridad (ver abajo).

## Ciclo de vida

1. **Warmup** (`warmup()`): al crear el sistema se lanza en background la carga
   del tamaño por defecto (`small`) sin bloquear el arranque. Los demás tamaños
   (`large`) se cargan bajo demanda en el primer `embed(...)`. Si
   `onnxEnabled=false` solo se loguea y se usará fallback hash.
2. **Carga perezosa** (`ensure`/`doLoad`): la sesión y el tokenizer se crean la
   primera vez que se necesitan. Los intentos fallidos se recuerdan para no
   reintentar (`attempts`), y las cargas en curso se comparten (`loading`).
3. La sesión lee `model.onnx`, `tokenizer.json` y `tokenizer_config.json` desde
   la carpeta del modelo (ver [configuracion.md](configuracion.md)).

## Cola de inferencia con prioridad

La inferencia ONNX es **CPU-bound y síncrona**, de modo que todas las llamadas
pasan por una **cola FIFO con prioridad** (2 niveles) procesada en serie:

| Nivel | Uso |
|---|---|
| `high: true` | Emplazamientos **en plena predicción**: estado de sesión, clasificador, reduce de keywords, re-rank de memorias. |
| `high: false` | **Recompute masivo de registro** (`POST /tools` → `upsertTools` y `buildTenantGraph`). |

- El drenador toma primero los items `high`, así una predicción entrante no espera
  detrás de cientos de recomputes de catálogo.
- **Dedupe** (`inflight`): un embedding idéntico (`size\0text`) en curso se
  reutiliza (misma promesa) en vez de computarse dos veces.
- Si la inferencia falla, se loguea un warning y se devuelve el **fallback hash**.

## Fallback hash (determinístico)

Si ONNX está deshabilitado, el modelo no existe o la inferencia falla:

- Tokeniza en minúsculas separando por no-alfanuméricos (incluye `áéíóúüñ`),
  descarta tokens de 1 carácter (se guardan los de longitud ≥ 2).
- Acumula frecuencia por token: `vec[hashDjb2(token) % dim] += 1` (dim según el
  tamaño solicitado: 384 en `small`, 1024 en `large`).
- Normaliza L2. El resultado es **determinístico** (mismos textos → mismos vectores).
- `model = "hash"`, `modelPath = ""`.

> El hash del fallback del engine NO coincide con el hash sparse de Qdrant: cada
> uno replica el esquema de su sistema original (ver [qdrant.md](qdrant.md)).

## Utilidades

- `embedQuery(text)` / `embedPassage(text)` — helpers con prefijo e5.
- `getInfo(size)` — `{ model: "small"|"hash", dim, modelPath }` según si la
  sesión está cargada (lo usa `GET /models` y `/debug`).
- `hashDjb2(token)` — hash del fallback (absoluto, por `charCodeAt`).
- Estáticas `EmbeddingEngineService.normalizeL2(vec)` y
  `EmbeddingEngineService.cosine(a, b)` — normalización L2 y producto punto con
  **clamp a [0,1]** (equivalente al `Cosine` del sistema original).
- `EmbedResult` = `{ embedding, model: "small" | "large" | "hash", dim, modelPath }`.

## Integración con el resto del sistema

| Consumidor | Uso |
|---|---|
| `TurnClassifier` | Embed `query:` del texto contra 2 arquetipos (ver pipeline). |
| `SessionStateCacheService` | Embed `query:` del prompt → `z_t` por sesión (prioridad alta). |
| `KeywordService.reduce` | Embed `query:` de keywords del prompt + `passage:` de keywords de cada tool. |
| `KeywordService.upsertTools` | Precomputa `passage:` de keywords de tools al registrarlas (prioridad baja). |
| `ToolGraphCacheService` | Embed `passage:` de la firma de cada tool → nodo del grafo (prioridad baja). |
| `PredictionOrchestrator.predictMemory` | Embed `query:` del prompt y `passage:` del contenido de memorias (cacheado). |
| Puerto 6777 — API de modelos | Expone `POST /embed` (siempre `small`) y `GET /models` (ver [api-http.md](api-http.md)). |
