# Plan 01: Resolución de Necesidades — Unificación de Modelos y Optimización de Recursos

## 1. Diagnóstico del Cuello de Botella Actual en `remtk-prediction`

Actualmente, el servicio unificado en TypeScript carga y coordina tres modelos Transformer independientes durante el ciclo de vida de `src/juicio/` y `src/embedding/`:

1. **`Xenova/multilingual-e5-small` (~118M parámetros totales):** Utilizado en `EmbeddingService` (Puerto `6777`), `PuertaContextoService` y `MaxSimService`.
2. **`Xenova/ms-marco-MiniLM-L-6-v2` (~22.7M parámetros):** Utilizado en `RerankService` y `CrossEncoderService`. Entrenado primordialmente sobre corpus en inglés (MS-MARCO), lo que degrada la comprensión semántica fina en español y obliga a compensar con heurísticas en `EspecificidadService` y `VarianteCService`.
3. **`Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (~278M parámetros):** Utilizado en `EstadoNliService`. Ejecuta 4 pasadas secuenciales o en paralelo (`Promise.all`) evaluando las hipótesis `TRANSICION`, `CONTINUACION`, `CORRECCION` y `ABANDONO` por cada consulta que requiere validación de estado.

### Impacto Operativo Actual
* **Consumo de Memoria RAM:** Entre $850\text{ MB}$ y $1.2\text{ GB}$ en el proceso de Node.js.
* **Latencia Acumulada en CPU:** Entre $65\text{ ms}$ y $180\text{ ms}$ cuando concurren la puerta de contexto, el estado NLI y el re-ranking de múltiples herramientas candidatas.
* **Desalineación con `AGENT.md`:** La debilidad del Cross-Encoder en español empuja a usar reglas léxicas compensatorias que violan el principio de razonamiento denso puro.

---

## 2. Solución Arquitectónica: El Cabezal Decisor `remtk-juicio-moe.onnx`

En lugar de reemplazar `e5-small` (que es indispensable para el contrato del Puerto `6777` y la indexación vectorial), **reutilizamos los tensores a nivel de token de `e5-small`** como entrada directa a un único modelo neuronal propio de toma de decisiones: **`remtk-juicio-moe-int8`**.

### Qué se elimina del servidor
* Se elimina `Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` ($0\text{ MB}$ en RAM).
* Se elimina `Xenova/ms-marco-MiniLM-L-6-v2` ($0\text{ MB}$ en RAM).
* Se eliminan las 4 inferencias redundantes de NLI por turno.

### Qué entra en su lugar
Un único grafo computacional ONNX cuantizado en `INT8` de **$\sim 18.5\text{ MB}$** ($18.5\text{M}$ de parámetros puros de decisión sobre las representaciones de `e5-small`) que resuelve en **una sola ejecución (`session.run()`)**:
1. Estado conversacional NLI (`TRANSICION`, `CONTINUACION`, `CORRECCION`, `ABANDONO`) y detección de `topicShift`.
2. Puntuación semántica y de especificidad de hasta $K = 200$ herramientas candidatas en paralelo (`rankedScores`).
3. Probabilidad calibrada de abstención (`abstencion_prob`) para escenarios de retorno nulo (`tools: []`).

---

## 3. Presupuesto Comparativo de Rendimiento y Costos

| Métrica del Sistema | Estado Actual (3 Modelos ONNX) | Nuevo Estado (`e5-small` + `remtk-juicio-moe`) | Mejora Neta |
| :--- | :--- | :--- | :--- |
| **Memoria RAM en Producción** | $950\text{ MB} - 1,200\text{ MB}$ | $280\text{ MB} - 340\text{ MB}$ | **$-70\%$ RAM** |
| **Llamadas a ONNX Runtime por `/predict`** | $1 + K + 4$ ($15\text{ a }45$ llamadas) | **$2$ llamadas fijas** ($1$ `e5-small` + $1$ `moe`) | **Reducción $10\times - 20\times$** |
| **Latencia de Juicio en CPU (AVX2)** | $60\text{ ms} - 150\text{ ms}$ | **$3\text{ ms} - 8\text{ ms}$** | **$\sim 15\times$ más rápido** |
| **Costo de Entrenamiento en CPU Local** | N/A | **$0.00\text{ USD}$** ($25 - 40\text{ min}$ en CPU) | Autonomía total |

---

## 4. Preservación Estricta de Contratos HTTP e In-Process

La migración es **100% transparente** para cualquier consumidor del paquete `remtk-prediction`:
* **Puerto `6776` (`/predict`, `/memory/predict`, `/tools`, `/debug`):** Mantiene exactamente el mismo esquema JSON de entrada y salida (`tools`, `complexity`, `modelSize`, `rankedScores`).
* **Puerto `6777` (`/embed`, `/models`):** Sigue sirviendo `e5-small` ($384\text{ dims}$) y `e5-large` ($1024\text{ dims}$) sin alteraciones.
* **Puerto `6775` (Qdrant BM25):** Continúa ejecutando el recall inicial y la expansión de sinónimos.