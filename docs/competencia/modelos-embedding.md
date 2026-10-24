# Laya AI (Convai Innovations) — investigación de competencia

> Documento de inteligencia competitiva. Recopila **cómo funciona** Laya, el
> motor de decisiones tipadas "System One" de Convai Innovations, y lo contrasta
> con el enfoque de `remtk-prediction`. Fecha de recopilación: 2026-09-29.
> Fuentes al final.
>
> **Advertencia sobre las cifras**: todas las métricas de rendimiento que se citan
> provienen del **propio autor** (README, `BENCHMARKS.md`, model cards de Hugging
> Face). Los números de Jev que aparecen en la comparación son **publicados por
> terceros y no medidos por Laya**. Se marcan explícitamente como tales.

---

## 0. Aclaración previa: tres nombres que se confunden

Antes de cualquier análisis hay que separar tres cosas distintas que circulan bajo
nombres casi idénticos:

| Nombre | Qué es | Tipo | Relevancia |
|---|---|---|---|
| **"Layla"** (app) | App móvil de rol/compañía ("Layla Network Pty Ltd", dev `l3utterfly`, Australia). Distribuye GGUF/llama.cpp, LiteRT, PTE. De pago (~$19.99) | App / LLM generativo | **No es un modelo de embeddings ni un modelo abierto** |
| **"Laya"** | Motor de decisiones **no autorregresivo** "System 1" de **Convai Innovations** (`convaiinnovations/laya`). Apache 2.0, `pip install laya` | Modelo de **decisión/clasificación tipada** | **Este** es el objeto del documento |
| **Modelo de embeddings "LAYLA"** | **No existe** | — | — |

**Hallazgo clave (verificado contra la API de Hugging Face y npm):**

- No existe ningún modelo de embeddings llamado *Layla* o *Laya*. La búsqueda
  `pipeline_tag=feature-extraction` para "laya/layla" devuelve vacío; toda la
  familia `convaiinnovations/laya*` está registrada como **`text-classification`**.
- Laya **no es un bi-encoder** y **no produce vectores de embedding**. Es un
  encoder (ModernBERT / mmBERT) con un *head* de decisión que devuelve
  **respuestas tipadas con probabilidad calibrada**, no una representación
  semántica reutilizable.

> Por tanto, este documento **no** compara "Laya" contra modelos de embeddings
> (no son comparables); documenta a Laya como **competidor conceptual** en la
> capa de "System One" (el mismo punto de la arquitectura que ocupa Jev). La
> comparación de modelos de embeddings para `remtk-prediction` es un asunto
> separado y se resuelve con el benchmark `scripts/bench-embeddings.ts`.

---

## 1. Qué es Laya

- **Producto**: Laya, "multilingual, non-autoregressive **System 1 decision
  engine**".
- **Autor / compañía**: **Convai Innovations**; autor principal Nandakishor
  Mukkunnoth (Kerala, India).
- **Licencia**: **Apache 2.0** (código y pesos). Publicado en PyPI (`laya`) y en
  Hugging Face (`convaiinnovations/laya`).
- **Versión observada**: paquete `laya` **0.3.22** (PyPI, instalada y verificada
  en este repo). Ojo: el `pyproject.toml` de la rama `main` del repo (v0.3.5) está
  **desactualizado** respecto a lo publicado en PyPI — el metadato real de la
  distribución manda.
- **Repositorio**: `github.com/NandhaKishorM/laya`.

### La idea central

Un LLM devuelve **texto** (que hay que parsear y puede alucinar). Laya **no genera
texto**: se le da un `state` (texto, email, ticket o JSON) y un conjunto de
**preguntas tipadas**; devuelve **respuestas tipadas con probabilidades
calibradas** en **una sola pasada forward**.

> Como el espacio de salida son solo probabilidades y números, el texto libre y
> las violaciones de esquema son **imposibles por construcción**.

Es un competidor directo de **Jev (TypeSafe AI)** — ver
[jev-ai.md](file:///c:/Users/LENOVO/Desktop/ULTIMA%20DE%202026/remtk-prediction/docs/competencia/jev-ai.md) —
y la diferencia de fondo es que **Laya es abierto y autoalojable** (Apache 2.0,
pesos en HF, coste 0), mientras Jev es una API cerrada.

---

## 2. Arquitectura

### 2.1 Codificadores base (backbones)

Tres checkpoints, consolidados en el repo HF `convaiinnovations/laya`:

| Checkpoint | Backbone | Parámetros | Contexto | Para qué |
|---|---|---|---|---|
| `laya` | ModernBERT-large | **421M** | 512 | Inglés |
| `laya-multilingual` | mmBERT-base (vocab 256k) | **322M** | 1024 (hasta 8 192 con RoPE) | 100+ idiomas; ~2× más rápido |
| `laya-typed-decisions` | ModernBERT-large | **421M** | 1024 | Los workflows de decisiones tipadas (fine-tune) |

- `mmBERT-base` usa un vocabulario de 256k y ventana nativa de 1 024 tokens
  ampliable a **8 192** por interpolación posicional RoPE.
- Tamaño en disco de `laya-typed-decisions`: **~843 MB** (safetensors, 421.293.830
  parámetros exactos).

### 2.2 Formato de secuencia y marcadores

La secuencia de entrada se ensambla internamente así:

```
[CLS] instrucciones [SEP] [MASK] opción_0 [MASK] opción_1 ... [SEP] estado [SEP]
```

Cada `[MASK]` actúa como **delimitador posicional** (*marker*) de una opción
candidata: el *scorer* lee la representación en esas posiciones para puntuar cada
alternativa.

### 2.3 Presupuesto de tokens dividido

- **`head_max_len`**: cuota reservada para las instrucciones y los criterios de las
  opciones (**192 tokens** en el checkpoint inglés; **256** en multilingüe y
  typed-decisions).
- **`max_len − head_max_len`**: margen restante para el documento evaluado.
- Es ajustable en runtime: `agent.cfg["head_max_len"] = 512`.

> Esta partición es la causa del principal límite de Laya (ver §6): con muchas
> opciones, cada etiqueta recibe muy pocos tokens y se vuelve indistinguible.

### 2.4 Cabezal de decisión (`DecisionModel`)

Sobre el encoder bidireccional se monta un head con:

- **`type_emb`**: un embedding tipado que codifica la primitiva ejecutada
  (`choice: 0`, `score: 1`, `noul: 2`).
- **Transformer de 2 capas pre-norm**.
- **`scorer`** (`LayerNorm → Linear → GELU → Linear`): se aplica sobre las
  posiciones de los *markers* para computar **un logit por opción**; el softmax se
  acota **exclusivamente al catálogo de alternativas** recibido.
- **`act_head`**: procesa la representación de `[CLS]` combinada con métricas
  estadísticas (probabilidad máxima, margen entre las dos primeras opciones,
  entropía normalizada y cardinalidad) para derivar `act_probability`.

> La implementación de referencia está confirmada de forma independiente por el
> port a Rust `laya-decision` (crate en docs.rs), que reproduce
> `DecisionModel` = encoder ModernBERT/mmBERT + `type_emb` + transformer de 2
> capas + `scorer` + `act_head`, y las utilidades de calibración.

### 2.5 Calibración de confianza

- Los logits se escalan con **temperaturas calibradas por tipo de consulta y
  cardinalidad**, minimizando el **ECE** (Expected Calibration Error) hasta ~**0.081**
  (post-temperature fitting).
- Las probabilidades se entrenan con **RLCD** (ver §3), por lo que —a diferencia de
  un LLM que "dice" `confidence: 0.95`— aquí la confianza es **estadísticamente
  significativa en agregado** y sirve para *gating* automático.

### 2.6 Router submilisegundo

`Router` es el punto de entrada recomendado. Antes de cualquier forward pass,
detecta el **script y el idioma** en **< 0.5 ms** (Python puro, tablas Unicode
sobre 22 alfabetos) y despacha al checkpoint óptimo:

- Texto latino en inglés → checkpoint inglés (`laya`).
- Otros alfabetos / idiomas → `laya-multilingual`.

Reglas de precedencia: `model=` explícito > `task=` > detección de workflow
tipado (solo si `auto_task_detection=True`) > `lang=` explícito > script detectado
> `default=`.

Cada resultado incluye metadatos de enrutamiento con la razón de la decisión, y se
puede inspeccionar sin ejecutar el modelo (`router.route(state, questions).reason`).

**Gestión de memoria / latencia:**

| Modo | Latencia por request | Recargas de modelo |
|---|---|---|
| `Router()` (lazy, `max_loaded=1`) | **7–10 s en cada cambio de idioma** | 1 por cambio |
| `Router(preload=True)` | **32.8 ms (GPU) / 193–464 ms (CPU)** | **ninguna** |

`preload=True` sube `max_loaded` para evitar la evicción LRU; `attach()` reutiliza
un checkpoint ya construido; `unload()` libera memoria. Los tres checkpoints juntos
son **~1.16B parámetros (~4.6 GB en fp32)**, por eso el default mantiene **uno**
residente.

---

## 3. Entrenamiento: RLCD

El método propio es **RLCD — Reinforcement Learning for Calibrated Decisions**.
La diferencia clave con RLHF/RLVR es **de dónde viene la recompensa**:

| Método | Señal de recompensa | Qué optimiza |
|---|---|---|
| RLHF | Preferencia humana | Que "guste" al humano → sobreconfianza, *mode dropping* |
| RLVR | Correctitud verificable (mates/código) | Que la respuesta sea correcta |
| **RLCD** | **Reglas de scoring estrictamente propias** (log + spherical; **RPS** para ordinal) | **Calibración** de la probabilidad de decisión |

El algoritmo de optimización es **REINFORCE con baseline estilo GRPO**. La
consecuencia práctica: Laya está optimizado para decir "0.8" **solo si ocurre ~el
80 % de las veces**, no para sonar seguro.

> Nota: `laya-typed-decisions` es un **fine-tune** sobre el benchmark
> independiente `LocalLLaMA/typed-decisions` (1 200 casos de entrenamiento =
> 6 000 decisiones; test de 400 casos = 2 000 decisiones).

---

## 4. Las tres primitivas de pregunta

Toda la superficie del API son **tres tipos**:

| Primitiva | Qué pregunta | Qué devuelve | Casos de uso |
|---|---|---|---|
| **`choice`** | Elige 1 opción de un diccionario que tú defines | clave elegida + **distribución completa** + `confidence` | enrutado de departamento, clasificación de intención, categorización |
| **`score`** | Ubica el `state` en una **rúbrica ordinal** (0,1,2,…) | nivel esperado (puede caer **entre** niveles) + distribución + `confidence` | frustración, urgencia de ticket, severidad de daño |
| **`noul`** | ¿Es verdadera esta afirmación? | un único número **P(true) 0.0–1.0** calibrado | phishing, spam, jailbreak, riesgo de churn |

Además incluye **presets** listos para producción: `router_questions()`,
`guard_questions()`, `moderation_questions()`, `triage_questions()`,
`email_questions()`.

---

## 5. Instalación (verificada)

Requisito: **Python 3.10 o superior**.

```bash
pip install laya
```

Dependencias reales declaradas por el paquete publicado (0.3.22):

```
torch>=2.0.0
transformers>=4.48.0
safetensors>=0.4.0
huggingface_hub>=0.20.0
numpy>=1.20.0
```

> El README menciona pisos más altos de facto (`huggingface_hub` 1.x,
> `transformers` 5.x, `torch` 2.14) que "fijan" el mínimo en 3.10, pero los pisos
> **declarados** son los de arriba. Verificar en el entorno real antes de fijar
> versiones.

### Uso básico (Router, modo recomendado)

```python
from laya import Router

router = Router(preload=True)  # checkpoints residentes → enrutado "gratis"

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund today or we cancel.",
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}

res = router.predict(state, questions)
print(res["answers"]["department"]["choice"])   # -> billing
print(res["answers"]["urgency"]["score"])        # -> 1.84
print(res["answers"]["churn_risk"]["noul"])      # -> 0.892
print(res["routing"]["model"])                   # -> english
```

### Carga directa de un checkpoint

```python
import laya

agent = laya.load("convaiinnovations/laya")                            # inglés
agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual")
agent_td = laya.load("convaiinnovations/laya", subfolder="typed-decisions")

result = agent.predict(state, questions)
```

### Extras y comandos de consola (verificados en la instalación)

El paquete publicado **sí define extras opcionales** y **comandos de consola** (a
diferencia del `pyproject.toml` de `main`, que está desactualizado):

| Extra | Añade |
|---|---|
| `laya[serve]` | `fastapi`, `uvicorn`, `python-multipart` (servidor HTTP) |
| `laya[fast]` | `tilelang` (kernels GPU de baja latencia) |
| `laya[mcp]` | `mcp` (Model Context Protocol) |
| `laya[structured]` | `pydantic` |
| `laya[onnx]` | `onnx`, `onnxruntime`, `onnxscript` (export/serving ONNX) |
| `laya[langchain]` / `laya[langgraph]` | `langchain-core`, `langgraph` |
| `laya[llamaindex]` | `llama-index-core` |
| `laya[crewai]` | `crewai` |

Comandos instalados (entry points):

```
laya               # CLI (laya.cli)
laya-serve         # servidor HTTP (laya.serve)
laya-mcp-server    # servidor MCP (laya.mcp.server)
laya-evals         # runner de evals
```

> **Nota de integración con `remtk-prediction`**: el extra `[onnx]` y el módulo
> `predict_shortlist` (junto a `embed_fn_from_agent` / `cached_embed_fn`) son
> relevantes porque permiten darle a Laya **embeddings del llamante** para
> preseleccionar candidatos — exactamente el patrón que encajaría con el catálogo
> de 232 tools de `remtk-prediction`.

---

## 6. Soporte en TypeScript / otros lenguajes

- **Port oficial `laya-ts/`**: el repositorio de Laya contiene un subdirectorio
  `laya-ts/` (con su propio `package.json`, tests y un
  `.github/workflows/release-ts.yml` que empaqueta y **publica a npm** al empujar
  un tag `laya-ts-v*`, bajo el nombre de paquete `laya-ts`). También existe el
  script de export `laya-ts/scripts/export_onnx.py`, referenciado por el port a
  Rust.
  - **No confirmado**: no pude verificar que `laya-ts` esté efectivamente
    publicado y disponible en el registro npm en la fecha de este documento
    (la consulta directa al registry no devolvió el paquete). Tratar como
    "existe en el repo, publicación por confirmar".
- **Sin empaquetado oficial desde Python**: no hay `pip install "laya[...]"` que
  instale tooling de servidor; el patrón de producción es levantar un proceso
  Python con `Router(preload=True)` y exponerlo, o usar un port de terceros.
- **Ports de la comunidad (verificados en npm)** — útiles si se quiere consumir
  Laya desde el stack Node/TS de `remtk-prediction`:
  - `@receptron/laya` — corre `convaiinnovations/laya` desde Node/TS vía
    **ONNX Runtime** (el más alineado con el stack ONNX de `remtk-prediction`).
  - `@r4ai/laya-web`, `laya-sdk` — decisiones tipadas en navegador/Node con ONNX.
  - `laya-system-one` — motor INT8 servido por binario nativo / WASM, wire-compatible
    con Jev (`/v1/systemone`).
  - `laya-http-client`, `@felipenmoura/laya-node`, `maclaya` (MLX/Apple Silicon),
    `kevala` (WebGPU/WASM), `laya-mcp`.
- **Port a Rust**: crate `laya-decision` (inferencia nativa sobre `candle`).

---

## 7. Rendimiento (cifras del autor)

### Velocidad (Tesla T4, medidas por el autor)

| Preguntas por llamada | `laya` | `laya-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

Throughput batcheado: **103–332 preguntas/s** en una sola T4.

### Laya (con routing) vs Jev

Los números de **Laya son medidos** por el autor; los de **Jev son de terceros, no
medidos por el autor** (sin acceso a la API de TypeSafe):

| | Jev 1.13.0 | Laya (routed) |
|---|---|---|
| typed-decisions, 2 000 decisiones | 0.727 | **0.766** |
| AG News, 4 labels | 0.910 | **0.950** |
| DAIR Emotion, 6 labels | 0.480 | **0.595** |
| Banking77 (72 vs 77 labels) | **0.870** | 0.425 |
| ECE *(menor = mejor)* | 0.246 | **0.081** (post-temperature) |
| p50 latencia, 1 pregunta | 236–276 ms | **32.8 ms** |
| Idiomas usables (>3× azar) | *sin benchmark publicado* | **45 de 51** |
| Pesos | API cerrada | **Apache 2.0** |
| Coste | $0.042 / 1M tokens | **$0 autoalojado** |

### `laya-typed-decisions` (400 casos, 2 000 decisiones, 4 workflows)

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| **`laya-typed-decisions`** | **0.766** | 0.471 | **0.062** | 0.213 | **0.242** |
| `laya` | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (publicado)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *techo teacher self-agreement* | *0.735* | | | | |
| *majority class por pregunta* | *0.461* | | | | |
| *azar* | *0.318* | | | | |

Por primitiva en el checkpoint typed-decisions: **`noul` 0.857**, `choice` 0.733,
`score` 0.723. `score` es la primitiva más débil.

### Enrutado (17 416 preguntas, una T4)

| Tarea | `english` | `multilingual` | `Router` |
|---|---|---|---|
| MASSIVE intent, inglés | **0.783** | 0.657 | **0.783** |
| MASSIVE intent, otros 13 idiomas | 0.306 | **0.451** | **0.451** |
| XNLI, inglés | **0.860** | 0.843 | **0.860** |
| XNLI, otros 14 idiomas | 0.521 | **0.731** | **0.731** |

El checkpoint inglés **colapsa** fuera del inglés (en Khmer: 0.000 de accuracy con
0.952 de confianza → la confianza alta no protege). De ahí que la detección de
script sea la señal primaria de enrutado.

---

## 8. Límites estructurales (honestos)

1. **No genera texto** → inútil para redactar, resumir o razonar abierto.
2. **Alta cardinalidad degrada**: por encima de ~20 opciones a ajustes por defecto
   la accuracy cae (Banking77: 0.425 con 77 labels). Causa: el presupuesto fijo
   `head_max_len` deja ~3–4 tokens por etiqueta. Mitigación: subir
   `head_max_len`, o `predict_shortlist` (embeddings del llamante + una pasada).
3. **Soft distribution matching** inferior a Jev (0.471 vs 0.580) aunque gane en
   argmax.
4. **Calibración cruda** peor que Jev antes de escalar (ECE 0.213 vs 0.144); el
   0.081 se logra **después** del temperature fitting sobre el dominio.
5. **Inglés únicamente** en el checkpoint inglés (colapsa en otros scripts sin
   degradación elegante) → usar `Router`/`multilingual`.
6. **`typed-decisions` es un fine-tune estrecho** (4 workflows sintéticos); no debe
   ser fallback silencioso.
7. **Coste de recarga**: sin `preload`, cambiar de idioma cuesta 7–10 s por request.

---

## 9. Comparación con `remtk-prediction`

| Dimensión | **Laya (Convai)** | **`remtk-prediction`** |
|---|---|---|
| Naturaleza | Modelo + librería **local**, Apache 2.0, self-host | Librería + servidor **local**, código propio |
| Tarea | **Decisiones tipadas** (`choice`/`score`/`noul`) sobre un `state` | **Predicción de tool/intención** en 3 capas (keywords cross-idioma → Qdrant BM25 → umbral adaptativo) |
| Modelo base | **ModernBERT-large / mmBERT-base** (encoder + head de decisión) | **Bi-encoder e5-small/e5-large/gte ONNX**, CPU, local |
| Salida | Tipada + **probabilidad calibrada** (RLCD) | Lista ordenada de tools + scores semánticos/keywords |
| Cardinalidad | Se degrada >~20 opciones (presupuesto `head_max_len`) | Catálogo real de **232 tools** con recuperación en 2 etapas (elegir tool ≠ elegir de 232 en una pasada) |
| Calibración | **Núcleo** (RLCD + temperature fitting, ECE 0.081) | Fusión heurística de capas + umbral adaptativo |
| Enrutado multilingüe | Router por script/idioma (<0.5 ms) → checkpoint | Embeddings multilingües + keywords cross-idioma |
| Latencia | 33–464 ms según backend/GPU | Depende del modelo de embeddings (CPU) |
| Coste | **$0** autoalojado | **$0** por inferencia (local, CPU) |
| Privacidad | Todo local | **Todo local** |
| Self-host | **Sí** | **Sí** |

### Lecturas estratégicas

- **Compite en la misma capa que Jev**, no en la capa de embeddings: es un "System
  One" local que decide sin llamar a un LLM grande. Frente a Jev su ventaja
  decisiva es que es **abierto, autoalojable y con pesos Apache 2.0**.
- **Ventaja de Laya**: calibración explícita y *gating* por `confidence`;
  cero-parseo; **router multilingüe** gratuito; coste 0.
- **Ventaja de `remtk-prediction`**: **recuperación a escala de catálogo** (232
  tools) sin la caída de alta cardinalidad de Laya; pipeline de selección de
  *herramientas* de primera clase; `e5`/`gte` ONNX ya integrados.
- **Riesgo/oportunidad a vigilar**: Laya demuestra el valor de **probabilidades
  calibradas** para *gating* automático. La capa de confianza de
  `remtk-prediction` hoy es **heurística** (umbral adaptativo), no calibrada →
  posible línea de mejora (reforzar calibración/confianza del pipeline).
- **Complementariedad posible**: Laya (o un port ONNX como `@receptron/laya`)
  podría actuar como **head de decisión tipada** sobre el catálogo recuperado por
  `remtk-prediction`, resolviendo su debilidad de alta cardinalidad mediante la
  etapa previa de recuperación de candidatos.

### 9.1 Benchmark real de embeddings: `gte-multilingual-base` vs alternativas

Laya **no es un modelo de embeddings** (es `text-classification`), por lo que no
es comparable cabeza a cabeza con `gte-multilingual-base`. La comparación
relevante se hizo, por tanto, **entre `gte` y las alternativas reales de
embeddings** del catálogo de `remtk-prediction`, ejecutando el banco A/B propio:

```bash
npx tsx scripts/bench-embeddings.ts
```

- **Corpus**: catálogo real de MiTumbes (`TOTAL_ACTIVE_TOOLS` = **232 tools**),
  codificado con el formato *query/doc* declarado por cada modelo.
- **Consultas**: **15** intenciones reales en español con su tool objetivo.
- **Métrica**: ranking por similitud coseno → `top1`/`top3`/`top5` y **MRR**;
  además latencia de query y tiempo de indexación en **CPU/ONNX**.

**Resultado (ranking por MRR, CPU/ONNX, ≤10 GB, sin GPU):**

| # | modelo | top1 | top3 | top5 | **MRR** | hero top5 | index 232 (s) | query (ms) |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | **embeddinggemma-300m** | 87% | 100% | 100% | **0.933** | 7/7 | 102.3 | 289 |
| 2 | **gte-multilingual-base** | 80% | 100% | 100% | **0.900** | 7/7 | **15.4** | **24** |
| 3 | e5-large | 73% | 100% | 100% | 0.856 | 7/7 | 77.9 | 112 |
| 4 | qwen3-0.6b | 67% | 100% | 100% | 0.833 | 7/7 | 112.8 | 247 |
| 5 | e5-large-instruct | 67% | 100% | 100% | 0.833 | 7/7 | 86.5 | 208 |
| 6 | arctic-embed-l-v2.0 | 67% | 100% | 100% | 0.833 | 7/7 | 31.8 | 43 |
| 7 | bge-m3 | 67% | 93% | 100% | 0.817 | 7/7 | 32.6 | 39 |
| 8 | e5-small | 53% | 87% | 87% | 0.707 | 7/7 | 5.5 | 13 |

> Todos los modelos aciertan el "hero" objetivo en el top-5 (7/7). Las 15
> consultas son duras por diseño: muchas comparten vocabulario con tools
> vecinas (`…_hero_obtener` vs `…_hero_configurar`), de ahí que `top3`/`top5`
> sature al 100% y el discriminante real sea `top1`/MRR.

**Conclusión — ¿cuál es mejor para `remtk-prediction`?**

- **Referencia de precisión (histórico)**: `embeddinggemma-300m` ganaba en MRR (0.933)
  y `top1` (87%), pero exige un presupuesto de CPU que un portátil no tiene (289 ms/query,
  102 s de indexado por 232 tools). **Se retiró del servicio**: `ONNX_MODEL_SIZE=gemma`
  ya no existe; el pipeline solo admite `gte`/`small`/`large`.
- **Modelo del servicio**: **`gte-multilingual-base`** (único ONNX del pipeline). Sacrifica
  solo **3.3 puntos de MRR** (0.900 vs 0.933) y **1 query de top1** (80% vs 87%, ya dentro
  del top-3), a cambio de ser **~12× más rápido por query** (24 ms vs 289 ms) y **~6.6× más
  rápido indexando** (15.4 s vs 102.3 s) con **~2.4× menos peso en disco/servidor**. Para un
  servidor CPU que atiende en línea —y más aún para desarrollo en portátil—, ese es el
  *trade-off* ganador.
- **Resto de la cola**: `e5-large` (0.856) es sólido pero pesado y lento;
  `qwen3-0.6b`, `e5-large-instruct` y `arctic-embed-l-v2.0` empatan en 0.833 con
  peor latencia; `bge-m3` cae en `top3` (93%); `e5-small` queda último (0.707).
  Ninguno domina a `gte` en la frontera precisión/velocidad.
- **Recomendación**: **`gte-multilingual-base` como modelo del servicio** (mejor relación
  precisión/rendimiento, viable en CPU de portátil). `gemma` queda solo como resultado de
  este banco (evidencia de la decisión), no como opción de configuración.

> **Nota metodológica**: el tamaño en disco de los modelos con pesos externos
> (`model.onnx_data`, p. ej. `embeddinggemma-300m` ≈ 300 MB, `e5-large` ≈ 560 MB)
> no se refleja en este banco porque mide solo el tamaño del grafo `.onnx`; las
> cifras de MRR/latencia sí son comparables (misma sesión, mismo corpus).

### 9.2 Prueba real de Laya como *selector de tool* (mismo banco, 232 tools)

Como Laya no es un bi-encoder, para compararla de forma justa con el pipeline de
recuperación de `remtk-prediction` se evaluó como **clasificador de tool de alta
cardinalidad**: la primitiva `choice` con las **232 tools** del catálogo real como
criterios, sobre **las mismas 15 consultas** del banco de embeddings. Máquina:
**CPU** (`torch 2.13.0+cpu`), checkpoint `convaiinnovations/laya` (subfolder
`multilingual`, mmBERT). Script reproducible: `scripts/bench-laya-decisions.py`.

**Modos evaluados**

- **A) Directo**: una sola pregunta `choice` con las 232 opciones.
- **B) Shortlist**: recuperación coseno con el propio encoder de Laya
  (`predict_shortlist` + `embed_fn_from_agent` + `cached_embed_fn`) para top-k, y
  luego la cabeza `choice` sobre esas k opciones.

**Resultados (15 consultas):**

| modo | **top1** | recall@k | latencia/query |
|---|:--:|:--:|:--:|
| A) directo (232 opciones) | **0%** (0/15) | — | 6.8 s |
| B) shortlist k=20 | **7%** (1/15) | **27%** (4/15) | ~2.5 s (74 s el 1.º query) |
| B) shortlist k=5 | **0%** (0/15) | **0%** (0/15) | ~2.3 s |

**Causa raíz del modo directo — colapso de opciones.** `usage.options.tool =
{total: 232, distinct: 9, tokens_per_option: 4}`: con `head_max_len=256` los 232
criterios se comprimen a **9 secuencias distintas de 4 tokens**, así que la cabeza
no puede distinguir la mayoría de tools y responde casi al azar (confianza
0.010–0.116). Es el límite del §8.2 medido sobre el catálogo real.

**Interpretación**

- Laya **no escala a un catálogo de 232 tools** en una sola pasada de decisión:
  `top1 = 0%`.
- Incluso con su patrón recomendado de shortlist, **su propio encoder recupera
  peor** que los bi-encoders dedicados: `recall@20 = 27%` frente al `top1` del
  banco de embeddings (`gte` 80%, `embeddinggemma` 87%). El `recall@5 = 0%` indica
  que su ranking *coarse* es débil en este corpus.
- Latencia: el 1.º query paga codificar las 232 descripciones (~70 s); después
  ~2.5 s/query. En CPU no compite con `gte` (24 ms).

**Conclusión — Laya vs `gte-multilingual-base` (para `remtk-prediction`).** Para la
tarea real del repo (**elegir 1 de 232 tools**), los **embeddings con recuperación
(`gte`/`embeddinggemma`) superan a Laya de forma clara**: Laya queda en **0–7% de
`top1`** frente a **80–87%**. Laya conserva su valor en su terreno propio (§9:
decisiones de **baja cardinalidad** con probabilidad calibrada), **no** en
recuperación masiva de tools.

> **Caveats de la prueba**: (1) ejecutada en **CPU**; (2) consultas en **español**
> sobre el checkpoint `multilingual`; (3) el encoder usado para el shortlist es
> *mean-pool* del encoder de la cabeza de decisión, **no** un bi-encoder entrenado
> para retrieval (la propia docstring de `embed_fn_from_agent` lo advierte), por lo
> que un bi-encoder dedicado shortlistaría mejor — lo que refuerza la conclusión
> de usar un bi-encoder real. Reproducir: `python -u scripts/bench-laya-decisions.py`.

---

## 10. Fuentes

- [Laya — Non-autoregressive System 1 decision engine (GitHub, NandhaKishorM/laya)](https://github.com/NandhaKishorM/laya)
- [laya — PyPI (paquete Python, autor Convai Innovations)](https://pypi.org/project/laya/)
- [Model card `convaiinnovations/laya` (Hugging Face)](https://huggingface.co/convaiinnovations/laya)
- [Model card `convaiinnovations/laya-multilingual` (Hugging Face)](https://huggingface.co/convaiinnovations/laya-multilingual)
- [Model card `convaiinnovations/laya-typed-decisions` (Hugging Face)](https://huggingface.co/convaiinnovations/laya-typed-decisions)
- [Demo Space `convaiinnovations/laya-demo` (Hugging Face)](https://huggingface.co/spaces/convaiinnovations/laya-demo)
- [BENCHMARKS.md (repo oficial)](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
- [laya-decision — Pure-Rust port (docs.rs)](https://docs.rs/laya-decision/0.1.0/laya/)
- [@receptron/laya — port Node/TS vía ONNX Runtime (npm)](https://www.npmjs.com/package/@receptron/laya)
- [Artículo: "I built non-autoregressive decision models…" (dev.to, autor)](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
- [Jev (TypeSafe AI) — investigación de competencia del repo](file:///c:/Users/LENOVO/Desktop/ULTIMA%20DE%202026/remtk-prediction/docs/competencia/jev-ai.md)
