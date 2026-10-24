# Arquitectura 02: Presupuesto de Tokens, Ejes Dinámicos y Gestión de Caché

## 1. Deducción Matemática de la Capacidad de Tokens en `remtk-prediction`

El sistema procesa entradas que provienen de múltiples campos del contrato (`text`, `keywords`, `intentContext`, `history`, `priorPlan`) y las cruza contra catálogos de hasta $200$ herramientas por tenant.

Gracias a la separación axial del modelo `remtk-juicio-moe.onnx`, los tokens se distribuyen en **tres tensores de entrada con ejes dinámicos independientes**:

| Nombre del Tensor en ONNX | Forma Simbólica (`shape`) | Límite de Secuencia por Ítem | Capacidad de Tokens por Llamada | Origen de los Datos |
| :--- | :--- | :--- | :--- | :--- |
| **`query_tokens`** | `(1, q_len, 384)` | $16 \le \text{q\_len} \le 256$ *(extensible a $512$)* | **$256 - 512$ tokens** | `text` + `keywords` normalizadas y deduplicadas. |
| **`context_tokens`** | `(1, ctx_len, 384)` | $16 \le \text{ctx\_len} \le 512$ | **$512$ tokens** *(o $4 \times 128$ condensados)* | Últimos turnos de `history` + segmentos relevantes de `intentContext`. |
| **`tools_tokens`** | `(num_tools, tool_len, 384)` | $1 \le \text{num\_tools} \le 200$<br>$32 \le \text{tool\_len} \le 128$ | **$3,840 - 25,600$ tokens** | Definiciones de herramientas (`name`, `group`, `category`, `description`, `intentSummary`, `tags`). |

### Capacidad TotalSimultánea por Petición (`/predict`)
En una consulta de estrés con un párrafo extenso ($256$ tokens), contexto e `intentContext` ($512$ tokens) y un catálogo de $150$ herramientas ($150 \times 96 = 14,400$ tokens), **el modelo evalúa simultáneamente $\sim 15,168$ tokens (con techo en $26,624$ tokens para $200$ tools de $128$ tokens)** en menos de $8\text{ ms}$.

---

## 2. Manejo de Párrafos Extensos (`parrafos-contexto-estres.test.ts`)

Cuando el campo `text` o `intentContext` supera los $256$ tokens (por ejemplo, cuando un usuario pega un bloque de código, un log de error o un documento largo):

1. **En `intentContext` (Pre-pensamiento del planificador):**
   * Tal como establece el contrato de `remtk-prediction`, `intentContext` se divide por oraciones/ideas hasta `MAX_INTENT_SEGMENTS` (respetando `MAX_INTENT_CONTEXT_CHARS`).
   * Cada segmento amplía el pool de candidatas en la fase de recall sin contaminar el vector principal de `text`.
2. **En `text` (Consulta principal extensa $> 256$ tokens):**
   * Si `text` tiene entre $256$ y $512$ tokens, `e5-small` lo procesa nativamente en una sola ventana (su límite físico `max_position_embeddings` es $512$).
   * Si `text` supera los $512$ tokens, `EmbeddingService` aplica una ventana deslizante (*sliding window*) con solapamiento de $32$ tokens y concatena los tokens de mayor norma de atención antes de entregarlos al tensor `query_tokens`.

---

## 3. Diseño de la Caché de Tensores de Herramientas en RAM

El secreto para que evaluar $150\text{ a }200$ herramientas tome $\sim 4\text{ ms}$ en lugar de $400\text{ ms}$ es que **las herramientas nunca se tokenizan ni pasan por `e5-small` durante `POST /predict`**.

### Flujo de Indexación en `POST /tools`
Cuando un tenant registra o actualiza sus herramientas:
1. Se formatea el texto canónico de cada herramienta:
   ```text
   passage: {name} | {group} / {category} | {intentSummary} | {description} | tags: {tags}
   ```
2. Se pasa en lote (*batch*) por `e5-small` extrayendo la última capa oculta antes del *mean pooling*: tensor de forma $(L_t, 384)$ con $L_t = 96$ (truncado o *padded*).
3. Se almacena además una máscara booleana `tool_mask` de longitud $96$ para ignorar los tokens de `<pad>` durante la operación `MaxSim` dentro del MoE.
4. En `POST /predict`, cuando la Etapa A (`MAX_CATEGORIES`) selecciona las $K$ herramientas candidatas (ej. $K = 75$), simplemente se copian las $75$ rebanadas de memoria contigua (`TypedArray.subarray`) para armar el tensor `tools_tokens` de forma $(75, 96, 384)$ en **$0.05\text{ ms}$**.