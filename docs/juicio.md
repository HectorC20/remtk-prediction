# Arquitectura de Juicio

Capa opcional que convierte al enrutador de tools de un buscador por
similitud en un sistema que emite un **juicio**:

> «Dado el estado S y la evidencia E, ¿se sostiene lógicamente la hipótesis H
> o se refuta?»

Todo se apoya en embeddings neuronales (e5-small ONNX); no introduce regexes ni
diccionarios de palabras clave. Los "arquetipos" son **semillas semánticas**
comparadas por coseno (e5 proyecta cualquier idioma al mismo espacio), igual
que los arquetipos del `TurnClassifier`.

Se desactiva en cascada: sin `JUICIO_ENABLED` o con el motor degradado (hash),
`pre()`/`post()` son pasa-through y el pipeline queda exactamente como antes
(sin la capa: 7/12; con ella: 11/12 en `bench/`).

---

## Los 6 componentes

Cada componente ataca un fallo del benchmark base (`bench/cases.ts`).

### 1. NLI de estado condicionado → `rechazo-propuesta`, `confirmacion-plan`

Decide si el turno **autoriza** o **refuta** la propuesta pendiente del
asistente, sin regexes de afirmación/negación.

- Hipótesis fija: `NLI_HYPOTHESIS` = «El usuario aprueba y autoriza proceder
  con la acción propuesta.»
- Premisa: el último mensaje del asistente en el historial; si no hay, el
  resumen del plan cacheado (`confirmCache`).

Veredicto `entailment / contradiction / neutral`:

- **Con modelo cross-encoder NLI** (`JUICIO_NLI_MODEL_PATH`): entrada de dos
  segmentos `[premisa SEP hipótesis]`, softmax sobre las 3 etiquetas.
- **Sin modelo (default)**: fallback por coseno contra dos pools de arquetipos
  (`REJECT_ARCHETYPES` vs. pool de confirmación del clasificador). Emite
  `reject` si el pool de rechazo supera al de confirmación por
  `JUICIO_NLI_MARGIN`; si no, `neutral`.

`reject` → se elimina el plan cacheado y `/predict` devuelve `∅` (intención
`unknown`, la hipótesis queda **descartada**). `confirm` con plan cacheado →
early-exit reutilizando el plan.

### 2. Abstención `__NOOP__` + energía → `small-talk`

Dos señales para no devolver herramientas cuando la entrada es ajena al
catálogo (saludo, reflexión, agradecimiento):

- **Prototipo NOOP**: coseno máximo del turno contra `NOOP_ARCHETYPES`. Se
  abstiene si `noop > top_pipeline + JUICIO_NOOP_MARGIN`.
- **Energía libre de Helmholtz** `E(u) = −T·ln Σᵢ exp(simᵢ/T)` sobre los scores
  del ranking: distribuciones planas y bajas → energía alta → out-of-distribution.

**Calibración medida:** la señal de energía está **desactivada por defecto**
(`JUICIO_ENERGY_TAU = 0`; τ ≥ 0 es un no-op porque `E < 0` siempre). En la banda
plana de cosenos de e5-small (todo cae en 0.79–0.91), `E` queda dominada por
`−max`: una consulta legítima con top 0.83 da `E ≈ −0.86` y un small-talk plano
da `E ≈ −0.96` — no existe τ que las separe. Se conserva la fórmula para
modelos con mayor rango dinámico. La abstención la carga el prototipo NOOP con
margen amplio (`JUICIO_NOOP_MARGIN = 0.10`): small-talk real da `noop=1.0` vs.
`top=0.845`, mientras turnos anafóricos legítimos llegan a `noop=0.912` vs.
`top=0.842` (dentro del margen → no se abstienen).

### 3. Puerta de coherencia semántica → `cambio-tema`

Aísla el turno actual del historial cuando el tema cambió, para que el estado
latent `z_t` no contamine la predicción.

```
λ = σ( cos(u_actual, z_previo) · β − γ )      (β=JUICIO_GATE_BETA, γ=JUICIO_GATE_GAMMA)
```

- `λ < JUICIO_GATE_TEXT_MIN` → **cambio de tema**: se aísla el turno (no se
  concatena history, `z_t` no se mezcla con el previo).
- `λ` alta → continuidad: se concatena history y `sessionState.resolve` mezcla
  el estado previo con peso `λ` en vez del 70/30 fijo.

### 4. Cross-encoder de juicio sintáctico → `negacion-borrador` (opcional)

Reranker ONNX de pareja `[consulta SEP descripción]` que capta la **negación**
donde la similitud léxica falla («no envíes… guárdalo como borrador»). Activo
solo con `JUICIO_RERANKER_MODEL_PATH` (`model.onnx` + `tokenizer.json`). Sin él,
`negacion-borrador` se resuelve por el resto de señales (gate + MaxSim).
Fusión: `score = (1−α)·score + α·sigmoid(logit)`, `α = JUICIO_RERANKER_ALPHA`.

### 5. MaxSim de interacción tardía (ColBERT) → `multi-intencion`

Multi-intención sin segmentar por conjunciones. Se conservan los embeddings por
token `{e₁..e_m}` de la consulta y cada herramienta puntúa por alineación máxima
token a token sobre su firma (`name + intentSummary + description + tags`):

```
Score(tool) = (1/|J|) Σⱼ maxᵢ (eᵢ · tⱼ)
```

Fusión con el score del pipeline: `score = (1−w)·pipeline + w·maxsim`,
`w = JUICIO_MAXSIM_WEIGHT`. **Calibración:** `w = 0.3` (con 0.5 el ruido
token-level reordena turnos anafóricos sin contexto léxico). Tope de tokens por
texto: `JUICIO_MAXSIM_MAX_TOKENS` (cota de cómputo O(m·n)). Los vectores por
token salen de `EmbeddingEngineService.embedTokens()` (expone
`last_hidden_state` y normaliza L2 cada token).

### 6. Índice dual + especificidad → `intencion-implicita`

- **Difusión semántica** `d(t)`: coseno medio de la herramienta contra el resto
  del catálogo (desde los nodos del grafo, sin inferencia extra). Herramientas
  generalistas (p. ej. `web_search`) colisionan con muchos temas → `d` alto →
  se castigan:
  ```
  score' = score / (1 + ρ·max(0, d(t) − d̄))      ρ = JUICIO_SPECIFICITY_RHO
  ```
- **Índice dual capacidad / problema**: si la herramienta declara
  `problemSpace` (cómo un usuario expresaría la necesidad, en su voz), se embebe
  y puntúa en paralelo. El score de `problemSpace` **solo reemplaza** al score
  fusionado propio si lo supera por `JUICIO_PROBLEM_MARGIN` (**0.08**). El
  reemplazo es contra la MISMA tool, no contra el líder; sin margen suficiente
  cualquier `problemSpace` que roce la consulta salta al top (bench: `save_draft`
  superaba a `send_email` con «quiero mandar el resumen por correo»). 0.08 exige
  cuasi-paráfrasis, que es el caso objetivo («mi jefe me pide el balance de
  ventas» → `download_report`).

---

## Integración en `/predict`

`JuicioService` (`src/juicio/juicio.service.ts`) actúa en dos momentos del
`PredictionOrchestrator.predict()`:

```
 text + history
      │
      ▼  (classifier → early-exit meta-pregunta)
 [pre]  ──► (1) NLI estado → reject: ∅ | confirm: plan cacheado
      │        (3) puerta λ  → decide aislar/continuar historial
      ▼
 [pipeline normal: recall → graph/flat rerank → threshold adaptativo]
      │
      ▼  [post] ──► (2) abstención (NOOP/energía) → ∅
      │             (4+5+6) re-rank top-K: maxsim + cross-encoder + dual + especificidad
      ▼
 GraphPredictionResult  (tools re-ordenadas; context.intent.primaryAction = top)
```

`post()` clona superficialmente el resultado; preserva `executionOrder` del
grafo y reconstruye `tools`/`rankedScores`/`graph.nodes` según el nuevo orden.
`pre()` reusa el embedding del turno (`ctx.uText`) para no recalcular en `post()`.
El cacheo de plan solo ocurre si el resultado trae herramientas (`tools.length>0`),
para no propiciar falsas confirmaciones.

Los servicios se construyen en `createSystem()` (`src/main.ts`) y llegan al
orquestador vía `createOrchestrator()` (`src/shared/factory/orchestrator.factory.ts`).
Al re-registrar catálogo (`POST /tools`) se invalidan las cachés del scope
(`maxsim.clearScope`, `especificidad.clearScope`).

---

## Configuración (`.env`)

Todos los valores por defecto viven en `src/shared/constants/juicio/`;
`src/config.ts` solo mapea cada env var sobre su constante.

| Env var | Default | Componente |
|---|---|---|
| `JUICIO_ENABLED` | `1` | maestro |
| `JUICIO_NLI_MARGIN` | `0.03` | 1 |
| `JUICIO_NLI_MODEL_PATH` | `""` (fallback arquetipos) | 1 |
| `JUICIO_NOOP_MARGIN` | `0.10` | 2 |
| `JUICIO_ENERGY_T` | `0.05` | 2 |
| `JUICIO_ENERGY_TAU` | `0` (desactivada) | 2 |
| `JUICIO_GATE_BETA` | `8` | 3 |
| `JUICIO_GATE_GAMMA` | `5` | 3 |
| `JUICIO_GATE_TEXT_MIN` | `0.5` | 3 |
| `JUICIO_RERANKER_MODEL_PATH` | `""` (opcional) | 4 |
| `JUICIO_RERANKER_ALPHA` | `0.6` | 4 |
| `JUICIO_MAXSIM_WEIGHT` | `0.3` | 5 |
| `JUICIO_MAXSIM_MAX_TOKENS` | `48` | 5 |
| `JUICIO_SPECIFICITY_RHO` | `0.5` | 6 |
| `JUICIO_PROBLEM_MARGIN` | `0.08` | 6 |
| `JUICIO_POST_TOP_K` | `10` | post |

### Modelos ONNX opcionales

- **NLI cross-encoder** (componente 1): carpeta con `model.onnx` + `tokenizer.json`
  (p. ej. un `mDeBERTa/` `xlm-roberta` fine-tuned en NLI multilingüe).
- **Reranker** (componente 4): misma estructura, salida logit único (pareja).

Sin ellos, los componentes 1 (arquetipos) y 4 (omitido) degradan con elegancia y
el resto opera con e5-small ya cargado.

---

## Resultado del benchmark

`pnpm bench -- --remtk` (catálogo 17 tools, modo `small+graph`, CPU):

| | sin juicio | con juicio |
|---|---|---|
| Aciertos | 7/12 | **11/12** |
| Latencia media | ~70 ms | ~106 ms |

Corregidos por la capa: `rechazo-propuesta` (NLI), `small-talk` (abstención
NOOP), `cambio-tema` (puerta de coherencia), `intencion-implicita` (índice dual).
Pendiente: `multi-intencion` — requiere descomposición real de intenciones
(dividir el turno en sub-consultas y enrutar cada una), no solo alineación
token-level; `send_email` queda 4º, empatado con `save_draft` en la banda plana.

---

## Punto de quiebre medido (banco de esfuerzo)

`pnpm bench:stress` (`bench/juicio-stress.ts`) escala la complejidad del texto
por 7 ejes hasta encontrar dónde falla cada componente. Reporta por turno las
señales internas (`estado` NLI, `λ` de puerta, `noop`, señales de re-rank) y el
**rank** del tool esperado. Tras densificar las escaleras de los ejes débiles,
el resultado global es **18/36 niveles superados** y los umbrales quedan así:

| Eje | Aguanta | Quiebre EXACTO | Mecanismo |
|---|---|---|---|
| A · longitud (intención sepultada) | ramble 10 (~150 palabras, rank #2) | **ramble 13 (~200 palabras)** — cae a #5 y ya no remonta hasta #7 en 26 | El relleno diluye el embedding del turno: `read_file`/`web_search` (generalistas) suben por encima del núcleo en cuanto la petición pesa <10 % del vector. La puerta y el NLI no re-enfocan el centro de la oración. |
| B · multi-intención | 4 acciones (incluso con delimitadores «uno/dos/…», todas presentes) | **5 acciones** — la salida colapsa a 2 tools (`web_search`+`download_report`); `translate_text`/`save_draft`/`get_weather` nunca aparecen | El **umbral adaptativo** recorta el conjunto en la ruta topológica; con 5 verbos solo sobreviven 2. No es fallo del juicio sino del techo de post-recall. |
| C · negación a distancia | **sin quiebre** (4 niveles) | — | La indexación dual de `save_draft` («no quiero enviarlo todavía…») sostiene la negación incluso a 8 frases de distancia, sin cross-encoder. |
| D · cambio de tema | **solo contexto mínimo** (D1, 0 relleno extra, λ=0.76) | **el 1er nivel con relleno** (ramble 2) — λ salta a 0.79 y `get_weather` desaparece del top (rank 13-14) | El más frágil de todos y más de lo que parecía: basta AÑADIR 2 frases de relleno al contexto viejo para que la puerta lea continuidad. El pivote educado comparte vocabulario con el historial y `z_previo` contamina la consulta. |
| E · code-switching es/en/fr | **sin quiebre** (3 niveles) | — | e5 proyecta los tres idiomas al mismo espacio; spanglish y mezcla larga se enrutan bien. |
| F · límite de abstención | **quiebre total** (nivel 1) | — | La abstención `__NOOP__` solo dispara en saludos cortos (`noop=1.0`). ANY divagación con relleno —incluso 2 frases— queda en `noop≈0.85` dentro del margen de `top+0.10` y NO se abstiene. F2 (largo PERO con intención) tampoco la rescata (rank 5). |
| G · coreferencia en cadena | **sin quiebre** (2 saltos) | — | El estado latente de sesión + puerta sostienen «muéstrame la última…» y «el total del que hablamos al principio» entre relleno. |

**Lectura honesta del quiebre.** Los tres límites estructurales son (1) **longitud
con relleno** (eje A/F): el modelo colapsa el turno a un vector y pierde el
núcleo cuando la petición es <10 % del texto (umbral medido: ~200 palabras); (2)
**techo de acciones** en la ruta topológica (eje B), que es del umbral
adaptativo (4 acciones máx), no del juicio; y (3) **pivotes de tema tras el menor
contexto** (eje D), donde `λ` vira de continuidad con solo 2 frases de relleno.
Los componentes 1 (NLI), 5 (MaxSim) y el índice dual no rompen nunca en estos
ejes.

Mejoras sugeridas por estos datos (aún NO implementadas): ventana deslizante o
poda de relleno antes del `embedQuery` para A/F; un segundo gate por similitud
del turno con el ÚLTIMO mensaje del asistente (no con `z` mezclado) y umbral
`γ` más alto para D; y subir `JUICIO_MAX_OUTPUT` / relajar el umbral adaptativo
para B. La abstención (F) requiere un arquetipo `__NOOP__` por canal o un
detector de divagación, no solo el margen global.
