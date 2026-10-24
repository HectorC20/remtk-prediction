# Capa Qdrant (BM25 sparse)

La integración con Qdrant se divide en dos piezas:

- `src/qdrant/qdrant-client.ts` — cliente HTTP mínimo (fetch nativo) contra el Qdrant externo.
- `src/qdrant/qdrant-service.ts` — orquestación de negocio: colecciones, sinónimos, RAKE, blend y catálogo por tenant.
- `src/qdrant/qdrant.controller.ts` — API de Qdrant (puerto 6775) registrada por `qdrant.module.ts`.

## Cliente (`QdrantClient`)

### Conectividad

- URL base desde `QDRANT_URL` (default `http://localhost:6333`).
- API key opcional enviada en cabecera `api-key`.
- Timeout de **15 s** por request (`REQ_TIMEOUT_MS`) con **2 reintentos** y backoff
  creciente 300/600 ms ante fallos transitorios (red/timeout). Los errores HTTP no
  se reintentan (se devuelven con su status).
- `enabled` refleja `QDRANT_ENABLED`. Todo método devuelve vacío / no-op si está
  deshabilitado.

### BM25 sparse client-side

Qdrant se usa con **vectores sparse** `bm25_text` (modifier `idf`). La tokenización
se replica **client-side** para que los índices coincidan con los del sistema
original (hash djb2 por code point, semilla 0):

1. Lowercase; chars permitidos: `a-z`, `0-9` y `áéíóúüñ`.
2. Se descartan tokens de longitud ≤ 1.
3. Por token: `índice = abs(hashDjb2(token))`, `valor = frecuencia del token`.

```ts
// Ejemplo conceptual
{ bm25_text: { indices: [382104, 90812, ...], values: [2, 1, ...] } }
```

### Puntos e IDs determinísticos

- **Upsert** con `PUT /collections/{c}/points?wait=true` (nunca `POST`), re-upsert
  sobre el mismo ID en vez de duplicar.
- **Búsqueda** con `POST /collections/{c}/points/query` usando `using: "bm25_text"`,
  `with_payload: true`, `with_vector: false`, filtros opcionales y
  `payload_selector.include` para traer solo los campos que se leen (reduce mucho
  el payload; los puntos de tools incluyen el documento completo).
- **IDs UUIDv5** (RFC 4122, namespace DNS) replicados desde el monolith:
  - `mcp-tool:{group||"general"}:{name}` → colección de tools.
  - `tool-kw:{group||"general"}:{name}` → colección de keywords.
- Límite por búsqueda: `MAX_SEARCH_LIMIT = 50`.

### Colecciones

Se crean con `PUT /collections/{c}` si no existen (no destructivo), declarando
`sparse_vectors: { bm25_text: { modifier: "idf" } }`.

## Servicio (`QdrantService`)

### Catálogo por tenant (memoria)

`QdrantService` guarda el **catálogo en memoria** `tenant → (name → ToolDefinition)`
como fuente autoritativa de definiciones completas. Es lo que usan count/recall y
el orquestador para devolver las tools (Qdrant solo devuelve `toolName` + score).
`setCatalog`/`catalog`/`countTools` viven de esta estructura: **funcionan incluso
sin Qdrant**.

### Indexado de tools (`indexTools`)

Por cada tool se escriben dos points:

**Colección `mcp_tools`** (1 point por tool):

```json
{
  "id": "uuidv5(\"mcp-tool:{group}:{name}\")",
  "payload": {
    "tenant": "...", "toolName": "...", "group": "...", "category": "...",
    "description": "...", "tags": ["..."], "intentSummary": "...",
    "toolDocument": "name group category description tags... intentSummary {schema}"
  },
  "vector": { "bm25_text": sparse(toolDocument) }
}
```

**Colección `tool_keywords`** (1 point por tool con keywords):

```json
{
  "id": "uuidv5(\"tool-kw:{group}:{name}\")",
  "payload": { "tenant": "...", "toolName": "...", "keywords": ["..."] },
  "vector": { "bm25_text": sparse(keywords.join(" ")) }
}
```

`keywords` = tokens de `tags` + palabras de `intentSummary`, en minúsculas,
deduplicados y recortados a 12. Si una tool no genera keywords, no se indexa point.

### Recall de tools (`retrieveTools`)

Es la capa léxica del pipeline de predicción (ver [pipeline-prediccion.md](pipeline-prediccion.md)):

```
 prompt ─► [1] expandSynonyms ─► [2] RAKE keywords
                │                        │
                ▼                        ▼
        [3] BM25 mcp_tools ◄────── BM25 tool_keywords (paralelo)
                │                        │
                ▼                        ▼
        [4] merge por toolName + blend 60/40 + cruce con catálogo
                ▼
        top-limit ordenado por blend
```

1. **Expansión de sinónimos** (`expandSynonyms`): por cada token de la query
   busca en `query_synonyms` (lookups en paralelo, caché por token en memoria) y
   agrega sinónimos a la query. Topes: `MAX_SYNONYM_EXPANSIONS = 8` términos
   nuevos. Filtro de contaminación: se descartan sinónimos de > 5 tokens o que
   colisionen con nombres de tools del catálogo.
2. **Keywords RAKE** (`extractKeywordsRake`, ligero): frases → frecuencia/grado
   por palabra → score `grado/frecuencia`, top `MAX_KEYWORDS = 8`. Replica el
   extractor del sistema original.
3. **BM25 en paralelo**:
   - `mcp_tools` con la query expandida (limit `min(max(limit*3,10), 50)`, include `toolName`).
   - `tool_keywords` con las keywords RAKE si existen (limit `min(max(limit*2,10), 50)`).
4. **Merge + blend 60/40**: si hubo matches de keywords, el score final es
   `0.6 × score_mcp_tools + 0.4 × score_keywords`; sin keywords, se usa el score
   crudo. Solo se conservan tools presentes en el catálogo del tenant. Orden
   desc y corte en `limit`.

### Memorias (`searchMemories`)

BM25 en `contextual_memories` con **filtro `userId`** (match exacto sobre el campo
`userId` del payload). Devuelve candidatas con payload `memoryId`, `content`,
`chatId`, `metadata`, `createdAt`.

### Sinónimos (`expandSynonyms` público) y status

- `/tools/synonyms` usa la misma función de expansión (endpoint de prueba/utilidad).
- `status()` consulta `GET /collections` y reporta `{ enabled, qdrant: { ok, collections } }`.

## Conservación de compatibilidad

| Recurso | Compatibilidad |
|---|---|
| IDs uuidv5 DNS | Coinciden con los del monolith: re-upsert actualiza sin duplicar. |
| Hash djb2 de tokens | Igual esquema de índices del Qdrant original. |
| Payloads | Mismos campos que `ToolRetrievalService.syncTools` y `MemoryRecallService`. |
| Endpoints HTTP 6775 | Contratos 1:1 con los servicios anteriores (ver [api-http.md](api-http.md)). |
