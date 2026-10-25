# Arquitectura 02: Presupuesto de Tokens, Ejes Dinámicos, Máscaras Implícitas y Caché RAM (Escala 300M)

## 1. Ejes Dinámicos y Detección Automática de Padding sin Romper el Contrato ONNX

El modelo `remtk-juicio-moe-300m` conserva exactamente los **3 tensores de entrada** del contrato de TypeScript (`query_tokens`, `context_tokens`, `tools_tokens`), pero resuelve internamente el problema de contaminación por relleno (*zero-padding*):

| Nombre del Tensor en ONNX | Forma Simbólica (`shape`) | Límite de Secuencia por Ítem | Capacidad de Tokens por Llamada | Detección Interna de Máscara en el Grafo |
| :--- | :--- | :--- | :--- | :--- |
| **`query_tokens`** | `(1, q_len, 384)` | $16 \le \text{q\_len} \le 512$ | **$256 - 512$ tokens** | `q_mask = (norm(query_tokens, dim=-1) > 1e-5)` |
| **`context_tokens`** | `(1, ctx_len, 384)` | $16 \le \text{ctx\_len} \le 512$ | **$512$ tokens** | `c_mask = (norm(context_tokens, dim=-1) > 1e-5)` |
| **`tools_tokens`** | `(num_tools, tool_len, 384)` | $1 \le \text{num\_tools} \le 200$<br>$32 \le \text{tool\_len} \le 128$ | **$3,840 - 25,600$ tokens** | `t_mask = (norm(tools_tokens, dim=-1) > 1e-5)` |

### Por qué la máscara automática por norma L2 corrige los fallos de decisión
En `EmbeddingService`, los tokens reales extraídos por `multilingual-e5-small` están normalizados en L2 ($\Vert{}\mathbf{v}\Vert{}_2 \approx 1.0$), mientras que las posiciones vacías de los buffers pre-alocados en RAM (`Float32Array`) contienen ceros exactos ($\Vert{}\mathbf{0}\Vert{}_2 = 0.0$).
* En la arquitectura anterior (18.5M), el operador `MaxSim` promediaba sobre `Lt = 96` fijo. Una herramienta concisa de 18 tokens tenía 78 ceros que arrastraban su score de especificidad (`mean_max_bwd`) hacia abajo.
* En la arquitectura **300M**, el grafo ONNX calcula la máscara exacta a partir de la norma del vector en $< 0.05\text{ ms}$ y divide únicamente entre el número de tokens válidos $N_{t, k} = \max(1, \sum m_{t, k})$.

---

## 2. Manejo de Párrafos Extensos y Contexto Multi-Turno

1. **Párrafos Extensos en `text` ($> 256$ tokens):**
   * Con la pila de **Convoluciones Dilatadas (`dilation = 1, 2, 4`)** y el **Attentive Pooling** de la Etapa 1, el modelo puede recibir hasta $512$ tokens sin que el texto informativo o el código pegado opaque la instrucción principal del usuario.
2. **Preservación del Objetivo en `context_tokens`:**
   * `context_tokens` concatena el resumen de `intentContext` (pre-pensamiento) seguido de los últimos turnos de `history`. Gracias al *Attentive Pooling* cruzado, incluso si el último turno del usuario es monosilábico (*"sí, con ese RUC"*), la Etapa 1 recupera la entidad y el verbo del turno anterior.

---

## 3. Presupuesto de Memoria RAM en Producción (Modelo 300M INT8)

| Componente en el Proceso Node.js | Tamaño en Disco | Consumo en RAM (Runtime) |
| :--- | :--- | :--- |
| **`multilingual-e5-small` (ONNX)** | ~118 MB | ~165 MB |
| **`remtk-juicio-moe-300m-int8` (`model_quantized.onnx`)** | **~295 MB** | **~340 MB** |
| **Caché de Tensores de Herramientas por Tenant ($200 \times 96 \times 384 \times 4\text{ B}$)** | 0 MB (En memoria) | **~29.4 MB por tenant** |
| **Total del Subsistema de Predicción + Juicio** | **~413 MB** | **~535 MB de RAM** |

> **Nota de Rendimiento:** A pesar de escalar de $18.5\text{M}$ a **$302.4\text{M}$ de parámetros**, la cuantización dinámica a `INT8` y el despacho de matrices con instrucciones vectoriales **AVX2 / AVX-512** en `onnxruntime-node` mantienen el consumo total del servidor Node.js muy por debajo del antiguo stack de 3 modelos Transformer ($1.2\text{ GB}$ RAM) y con una latencia de inferencia de **$14\text{ ms a } 22\text{ ms}$** en CPU.