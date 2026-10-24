# JEV AI (TypeSafe AI) — investigación de competencia

> Documento de inteligencia competitiva. Recopila **cómo funciona** Jev, el modelo
> "System One" de TypeSafe AI, y lo contrasta con el enfoque de `remtk-prediction`.
> Fecha de recopilación: 2026-09-29. Fuentes al final.
>
> **Advertencia sobre las cifras**: casi todas las métricas de rendimiento
> (latencia, precio, "0% de errores de esquema") son **declaraciones del propio
> vendedor**, no benchmarks independientes. Se marcan explícitamente como tales.

---

## 1. Qué es

- **Producto**: Jev, el primer modelo de la categoría que TypeSafe AI llama
  **"System One Models"**.
- **Compañía**: TypeSafe AI (San Francisco). Salió del modo sigiloso el
  **15 de septiembre de 2026** con una ronda semilla de **$40M liderada por DCVC**.
- **Fundadores**: **Diogo Almeida** (ex-OpenAI, coautor de la investigación de
  RLHF / InstructGPT que derivó en ChatGPT), **Erik Gafni** y **Sasha Sheng**.
- **Nombre**: "Jev" por **William Stanley Jevons** (paradoja de Jevons: al bajar
  el coste sube el consumo). "System One" por la dicotomía de Kahneman
  (pensamiento 1 = rápido/intuitivo vs. 2 = lento/deliberado).
- **Versión observada**: `jev-1.13` / `jev-1.13.0` (id en OpenRouter `typesafe/jev-1.13`,
  Cloudflare Workers AI `typesafe/jev`).

### La idea central

Un LLM devuelve **texto** (una string que puede ser cualquier cosa: respuesta,
negativa, prosa, JSON roto). Jev **no devuelve texto**: declaras por adelantado
la forma de la respuesta (qué campos y qué valores admite cada campo) y el modelo
te devuelve esos campos **rellenos, cada uno con una probabilidad**.

> **"Non-structured state in → typed probabilistic decisions out."**
> Es esencialmente un *function call* de inteligencia, no un chatbot.

Su manifiesto ("Decisions, not strings" / "Machine Native Intelligence") sostiene
que la IA actual fue entrenada asumiendo que hay un humano al otro lado; Jev se
entrena asumiendo que el consumidor es **software**.

---

## 2. Cómo funciona

### 2.1 Entrada y salida

| | Entrada | Salida |
|---|---|---|
| Contenido | `state` (texto plano o **JSON anidado**: ticket, pedido, registro) + mapa de **preguntas tipadas** | una respuesta tipada por pregunta, **cada una con probabilidad/confianza** |
| Formato | `POST /v1/systemone` (docs oficiales) con `{ model, state, questions }` | JSON nativo, **sin prosa ni markdown** que parsear |

- Ventana de contexto: **~32K tokens** (≈150k caracteres en inglés).
- No acepta **imágenes, audio ni vídeo**.
- Las preguntas se evalúan **en paralelo y en aislamiento** contra el mismo `state`;
  añadir preguntas apenas cambia el tiempo de respuesta.

### 2.2 Las tres primitivas de pregunta

Toda la superficie del API son **tres tipos** de pregunta:

| Primitiva | Qué pregunta | Qué devuelve |
|---|---|---|
| **Choice** | Elige 1 opción de una lista que tú defines (**hasta 255**) | opción elegida + **distribución de probabilidad completa** + `confidence` |
| **Score** | Puntúa el `state` sobre una rúbrica ordenada (**2–10 niveles**) | posición ponderada por probabilidad (puede caer **entre** niveles) + probs por nivel + `confidence` |
| **Noul** | ¿Es verdadera esta afirmación? | un único número **0–1** ("la probabilidad *es* la creencia"; **sin** campo de confianza separado) |

Sobre las 255 opciones de `Choice`: por encima de ese tope TypeSafe usa un enfoque
de dos etapas (puntuar candidatos y luego elegir), que reconoce **más lento**.

### 2.3 Arquitectura de inferencia: sampler paralelo

- Es **transformer-based**, pero **no autoregRegenera token a token**.
- **Sampling paralelo**: todos los campos se producen en **una sola pasada** sobre
  el estado → de ahí la velocidad.
- Conformidad de esquema **por construcción**: como las salidas posibles están
  enumeradas de antemano, un error de tipo "no es improbable: es **imposible**
  emitirlo" (TypeSafe: "falsificable con un solo contraejemplo").

### 2.4 Entrenamiento: RLCD

El método propio es **RLCD — Reinforcement Learning for Calibrated Decisions**.
La diferencia clave respecto a RLHF/RLVR es **de dónde viene la recompensa**:

| Método | Señal de recompensa | Objetivo que optimiza |
|---|---|---|
| RLHF | Preferencia humana | Que el humano lo vea "bien" → produce *chat*, **sobreconfianza** y *mode dropping* |
| RLVR | Correctitud verificable (mates/código) | Que la respuesta sea correcta |
| **RLCD** | **Verdad de referencia + reglas de scoring propias (tipo Brier Score)** | **Calibración** de la probabilidad de decisión (*epistemic honesty*) |

El **Brier Score** (métrica de calibración de pronósticos meteorológicos: si dices
"80%", ¿ocurrió el 80% de las veces?) es la base matemática del argumento de que
Jev da "probabilidades honestas". Según los reportes, Jev se entrena **solo con
datos sintéticos**.

### 2.5 Arquitectura de sistema recomendada

TypeSafe y terceros describen el patrón así:

> **"code controls, Jev decides, the LLM writes"** (3 capas)

1. **Código** controla el flujo (orquesta, enruta, aplica reglas de negocio).
2. **Jev** decide (clasifica, enruta, puntúa, verifica, gatekeepa).
3. **LLM** escribe (solo cuando hace falta prosa/reasoning abierto).

Jev **complementa** al LLM, no lo reemplaza: LLM = System Two (razonamiento abierto),
Jev = System One (juicios estructurados de alta frecuencia).

### 2.6 Patrón operativo clave: enrutado con umbral de confianza

Como la confianza es **significativa en agregado** (a más confianza, más acierto),
se ponen **umbrales distintos por acción** según el coste de equivocarse:

- `confidence` alto → **actuar** (p. ej. acción de solo-lectura, barra baja).
- `confidence` medio → **revisar**.
- `confidence` bajo → **derivar a humano** o a un modelo mayor.

---

## 3. Rendimiento y precio (cifras del vendedor)

| Métrica | Valor | Nota |
|---|---|---|
| Latencia extremo a extremo | **70–500 ms** (≈100 ms típico) | declarado por TypeSafe |
| Precio de entrada | **$0.042 / M tokens** (~$42 / 1B) | público |
| Precio de salida | **gratis** ("too cheap to meter") | público |
| Coste por decisión (demo) | **$0.000081** vs **$0.013880** del LLM comparado | demo propia; **444.6×** más barato |
| Latencia por decisión (demo) | **0.114 s** vs **8.566 s** | demo propia; **193.6×** más rápido |
| Errores de salida estructurada | **0%** | significa *conformidad de esquema*, **no** correctitud |

### Letra pequeña (lo que el propio vendedor admite o se verifica externamente)

- Los "193.6× / 444.6×" salen de **los evals de workflow de TypeSafe**
  (`evals.typesafe.ai`), con la **respuesta de referencia definida por el propio
  equipo** y los flujos **escritos por el equipo de TypeSafe**. La empresa dice
  que no puede probar que el precio no esté subvencionado.
- En su **propio dashboard de evals**, la exactitud agregada de Jev es **67.8%**
  frente a **74.1%** del mejor modelo de comparación → **más rápido y barato,
  pero menos exacto** en esas tareas (peor aún en procesamiento de facturas).
- **"0% de alucinaciones" = correspondencia de esquema garantizada**, no verdad
  empírica. Un "Noul" puede ser tipológicamente correcto y estar **equivocado**.
- Benchmarks independientes citados por terceros (p. ej. Vercel, Bryo AI) son
  mixtos: mejor/más rápido en unos casos, "Gemini algo más exacto pero 10–20×
  más caro" en otros.
- **No hay pesos publicados, ni recuento de parámetros, ni opción self-host.**
  La arquitectura **no fue divulgada**.

---

## 4. Dónde se usa

Casos que atacan exactamente los "micro-juicios" de alta frecuencia dentro de software:

- **Soporte al cliente**: clasificar tickets, medir urgencia, enrutar a cola/equipo.
- **Enrutado de LLM / selección de tool-handler / Agent workflows**: "¿qué modelo,
  herramienta o agente debe manejar esto?".
- **Recomendación y ranking / filtrado de relevancia**: "¿qué tan relevante es este
  candidato para la petición actual?".
- **RAG**: relevancia de pasajes, cobertura de respuesta, **reranking**, detección
  de *prompt injection*.
- **LLM as a Judge**: calidad de respuesta con **rúbricas propias** (groundedness,
  answer quality).
- **Content safety / guardrails**: "¿viola esta regla?" con gate a humano si hay duda.
- **Compresión de contexto**: keep / truncate / drop en el contexto de un agente.
- **Extracción y matching de entidades**; **juegos/NPC** (elegir acción legal por estado).

### Formas de acceso

Hosted API (acceso anticipado con waitlist) + gateways: **OpenRouter**,
**Vercel AI Gateway**, **Netlify AI Gateway**, **Cloudflare Workers AI**, **AIMLAPI**.
SDKs: `typesafe-sdk` (Python ≥3.10), `@typesafe-ai/sdk` (JS), integración
`langchain-typesafe`, y un *agent skill* para Claude Code.

---

## 5. Límites estructurales

1. **No genera texto** → inútil para redactar, resumir o razonar abierto.
2. **No planifica multi-paso abierto** (no es un agente de razonamiento largo).
3. **No multimodal**: sin imágenes, audio ni vídeo.
4. **Choice ≤ 255 opciones**; por encima, dos etapas (más lento).
5. **Puede equivocarse**: la conformidad de esquema no garantiza que la opción
   elegida sea la correcta (el error sigue siendo posible, solo que dentro del esquema).
6. **Cerrado**: sin pesos, sin self-host, sin cnt. de parámetros; un único endpoint
   hosted (con espejos de terceros de fiabilidad variable).
7. **Exactitud vs. frontier**: en los propios evals queda por debajo de modelos
   frontier en tareas agregadas.

---

## 6. Comparación con `remtk-prediction`

| Dimensión | **Jev (TypeSafe)** | **`remtk-prediction`** |
|---|---|---|
| Naturaleza | Modelo hospedado (API), propietario | Librería + servidor **local**, código propio |
| Tarea | Juicios tipados (Choice/Score/Noul) sobre un `state` | **Predicción de tool/intención** en 3 capas (keywords cross-idioma → Qdrant BM25 → umbral adaptativo) |
| Modelo base | Transformer propio "System One" (no divulgado) | **bi-encoder e5-large / e5-small ONNX**, CPU, local |
| Salida | Typed + probabilidad calibrada | Lista ordenada de tools + scores semánticos/keywords |
| Coste | $0.042/M in, out gratis (hosted) | **0** por inferencia (local, CPU) |
| Latencia | 70–500 ms (declarada) | Depende del modelo (e5-small/large, CPU, local) |
| Datos | Sintéticos, en la nube del proveedor | Catálogo real de tools del tenant (**232 tools** de prueba) |
| Calibración | **Núcleo del producto** (RLCD + Brier) | Fusión heurística de capas + umbral adaptativo |
| Privacidad | `state` sale a la nube | **Todo local** (RAM/CPU), sin fuga de datos del tenant |
| Enrutado por confianza | Patrón de primera clase | Umbral adaptativo (`adaptiveGapThreshold`, `adaptiveMinScore`) |
| Self-host | No | **Sí** |

### Lecturas estratégicas

- **Compite en otra capa**: Jev es un *modelo hospedado de decisiones*; `remtk-prediction`
  es un *motor local de selección de herramientas* con pipeline propio. No es
  un reemplazo directo, pero sí un **competidor conceptual** en el mismo punto de
  la arquitectura: el "System One" que decide qué hacer sin llamar a un LLM grande.
- **Ventaja de Jev**: calibración explícita (puedes *gatear* por `confidence`) y
  cero-parseo; **coste de salida 0**; latencia baja.
- **Ventajas de `remtk-prediction`**: **100% local/privado**, sin coste por token,
  sin dependencia de un proveedor, catálogo del tenant de primera clase, e5 ONNX
  ya integrante del pipeline y **control total del código**.
- **Riesgos de Jev a vigilar**: (a) exactitud agregada por debajo de frontier en sus
  propios evals; (b) "0% de errores" es conformidad, no verdad; (c) sin self-host
  → dependencia de proveedor y de un solo endpoint.
- **Movimiento defensivo natural**: reforzar la **calibración/confianza** de la capa
  de predicción (hoy heurística) y la **comprensión de intención** del bi-encoder
  local — de ahí la evaluación de un modelo de embeddings mejor que `e5-large`
  (ver `docs/competencia/modelos-embedding.md`).

---

## Fuentes

- [Jev API — TypeSafe System One Model API Access, Docs & Code Examples](https://jevapi.org/)
- [TypeSafe AI's Jev and "System One Models": What Actually Shipped (TrueFoundry, 2026-09-18)](https://www.truefoundry.com/blog/typesafe-ai-jev)
- [TypeSafe AI Releases Jev: A System One Model That Returns Typed, Calibrated Decisions Instead of Text (Meta AI Labs, 2026-09-19)](https://metaailabs.com/typesafe-ai-releases-jev-a-system-one-model-that-returns-typed-calibrated-decisions-instead-of-text/)
- [What Is Jev? The 2026 Complete Guide to TypeSafe's System One Model (a2aprotocol.ai, 2026-09-19)](https://a2aprotocol.ai/insights/2026-jev-system-one-model-guide)
- [What Is Jev? TypeSafe AI's Decision Model Explained (Layer3Labs, 2026-09-23)](https://www.layer3labs.io/guides/jev-explained)
- [Jev Review: Is It Any Good? (Layer3Labs, 2026-09-22)](https://www.layer3labs.io/guides/jev-review)
- [Jev AI — Playground & API (jev-ai.pro)](https://jev-ai.pro/)
- [Jev：TypeSafe AI 的 RLCD 决策模型 (blog.mushroom.cv, 2026-09-18)](https://blog.mushroom.cv/blog/typesafe-ai-jev-system-one-model-rlcd-decision-ai-enterprise/)
- [Jev AI Model — 开发者指南与社区 (jevai.dev)](https://jevai.dev/zh-Hans/)
- [Jev by TypeSafe AI — 试用 System One 模型与 API (jevtypesafeai.com)](https://jevtypesafeai.com/zh)
