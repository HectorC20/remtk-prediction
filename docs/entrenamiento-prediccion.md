# Entrenamiento de la predicción — aprendizaje léxico por canal

Este documento **no** describe lo que hoy hace el servicio, sino lo que se propone
construir encima de él: una **capa de aprendizaje léxico incremental, por canal
(scope) y disparada al registrar herramientas**, capaz de mejorar la precisión de
predicción por encima de lo que hoy consigue el catálogo estático de MiTumbes.

- Estado: **propuesta** (nada de esto está implementado).
- Ámbito: `remtk-prediction` (servidor) + el punto de enganche en `key-remtk` (consumidor).
- Requisitos duros del pedido: ágil/rápido, por usuario conectado a su canal, sin
  afectar a otros usuarios, y **centrado en palabras clave**.

---

## 1. Resumen ejecutivo

La predicción actual es **estática**: el ranking se calcula siempre igual a partir
de los metadatos de la herramienta (`name`, `group`, `description`, `tags`,
`intentSummary`, `inputSchema`) y del prompt del turno. No hay ninguna señal que
se acumule con el uso: el sistema no aprende que *"constancia de residencia"* es
como ese usuario concreto llama a la herramienta que en el catálogo se describe
como *"certificado de domicilio"*.

La propuesta es añadir una **cuarta señal de scoring** —un score léxico aprendido
por canal— sin tocar los embeddings ni el grafo:

```
final_i = semántico_i  +  keywordBoost · BM25_i  +  learnWeight · aprendido_i
          └── grafo z_t ──┘  └─ Qdrant BM25 ─┘     └── NUEVO ──┘
```

El aprendizaje es un **refuerzo de pesos término→herramienta** (contadores
ponderados con decaimiento exponencial e IDF del propio canal). Es O(1) por
evento, no usa GPU, no usa batch y no reentrena ningún modelo. Cada canal
(`tenant::agentId`) tiene su propio perfil léxico, de modo que el aprendizaje de
un usuario es **físicamente incapaz** de alterar la predicción de otro.

```
POST /tools (alta de herramientas del canal)
        │
        ├─ indexTools ......... Qdrant sparse         (ya existe)
        └─ warmScope ......... keywords + grafo       (ya existe)
                    └─ seedScope ......... perfil léxico del canal   ← NUEVO

POST /predict
        └─ score = semántico + BM25 + learnWeight · aprendido(perfil del canal)  ← NUEVO

POST /predict/feedback (qué tools funcionaron de verdad)
        └─ observe ............. refuerza/penaliza término→tool del canal        ← NUEVO
```

---

## 2. Requisitos y restricciones

| # | Requisito | Cómo lo cubre este plan |
|---|---|---|
| R1 | Ágil y rápido en procesos | O(1) por evento, sin ONNX, sin batch, sin GPU. §8 |
| R2 | Entrenamiento al **agregar** herramientas | `POST /tools` → `seedScope` (semilla del perfil). §7 |
| R3 | Por **usuario conectado a su canal** | Perfil particionado por `scopeKey = tenant::agentId`. §9 |
| R4 | **No afectar a otros usuarios** | Mapas por scope + filtro obligatorio por `tenant` en Qdrant + test de aislamiento. §9 |
| R5 | Se entrena **por palabras clave** | La unidad de aprendizaje es el término normalizado, no el embedding. §6 |
| R6 | Mejor precisión que MiTumbes-Server | Golden set y comparación A/B/C. §12 |
| R7 | Sin regresión cuando no hay datos | `learnWeight` efectivo = 0 hasta `LEARN_MIN_EVENTS`. §6.5 |

---

## 3. Arquitectura actual (línea base)

### 3.1 Pipeline de `/predict`

Punto de entrada: `PredictionOrchestrator.predict()` en
[orchestrator.ts](../src/predict/orchestrator.ts).

1. **Clasificación de turno** (`TurnClassifier`): nueva consulta / confirmación
   vacía / confirmación con matiz.
2. **Early-exit**: meta-pregunta de capacidades → 0 tools; confirmación vacía con
   plan cacheado (`ConfirmationCache`) → devuelve el plan previo.
3. **`promptText`**: en consulta nueva se añaden hasta 4 mensajes de historial ×
   300 chars; en confirmación se usa el `priorPlan`.
4. **Estado latente de sesión `z_t`** (`SessionStateCacheService`): embedding
   `query:` del prompt, blend 70/30 con el estado previo si el coseno ≥ 0.86
   (`TOPIC_SHIFT_THRESHOLD`); si no, resetea.
5. **Dos rutas de ranking**:
   - **Ruta topológica** (existe grafo del scope): `RerankService.graphFilter()`
     `s_i = cos(z_t, E(T_i)) + keywordBoost·BM25_norm_i` → propagación de
     pre-requisitos `S = s + 0.2·(Aᵀ·s)` → piso de relevancia → umbral adaptativo
     → resolución de mutexes → orden topológico (Kahn).
   - **Ruta plana** (sin grafo): `KeywordService.reduce()` (top-K=20 por coseno
     cross-idioma de keywords) → `RerankService.filter()`
     `fused = coseno_keywords + keywordBoost · BM25_norm`.
6. **Umbral adaptativo** (`adaptiveThreshold`): mínimo garantizado 2, corte por
   gap natural (0.03) o cluster superior (3×gap) si el ranking es denso.
7. **Persistencia de traza** en el `Debugger`.

### 3.2 Señales de scoring hoy

| Señal | Origen | Peso | Naturaleza |
|---|---|---|---|
| Semántica | coseno `z_t` ↔ embedding de la firma de la tool (e5-small, 384d) | 1.0 | estática |
| Léxica BM25 | `QdrantService.retrieveTools` → colecciones `mcp_tools` + `tool_keywords`, blend 60/40 | `keywordBoost` = 0.15 | estática |
| Estructural | propagación por matriz de adyacencia del grafo | `α` = 0.2 | estática |
| Umbral | piso de relevancia `adaptiveMinScore` = 0.8 + gap adaptativo | — | estática |

Todas ellas se derivan **exclusivamente de los metadatos de la herramienta**. La
voz del usuario (con qué palabras pide las cosas) no influye en absoluto.

### 3.3 Partición por canal (ya existe)

`resolveScopeKey(tenant, agentId)` en [scope.ts](../src/shared/scope.ts) produce
`tenant::agentId` (o solo `tenant` si no hay agente). Todo lo pesado ya está
particionado por esa clave:

| Estructura | Fichero | ¿Aislada por scope? |
|---|---|---|
| `catalogs` | [qdrant-service.ts](../src/qdrant/qdrant-service.ts) | Sí |
| `perScope` (embeddings de keywords) | [keyword.service.ts](../src/predict/services/keyword.service.ts) | Sí |
| `graphs`, `edgesByScope` | [graph-cache.service.ts](../src/predict/services/graph-cache.service.ts) | Sí |
| `warmups` | orchestrator | Sí |
| `synonymCache` | [qdrant-service.ts](../src/qdrant/qdrant-service.ts) | **No** — ver §3.4 |

Las cachés de sesión (`SessionStateCacheService`, `ConfirmationCache`, `topics`)
se indexan por `sessionId`, no por scope: son efímeras y no constituyen
aprendizaje.

### 3.4 Diagnóstico del gap

1. **No existe aprendizaje.** Ninguna estructura acumula información entre
   turnos. `SessionStateCache` y `ConfirmationCache` son buffers de sesión, no
   memoria de canal: se pierden con la sesión y no sobreviven a un reinicio.
2. **El ranking es ciego al vocabulario del usuario.** El único puente entre
   las palabras del usuario y el catálogo es el embedding multilingüe, que
   generaliza pero no se especializa.
3. **El piso de relevancia bloquea cualquier rescate léxico.** En
   [rerank.service.ts](../src/predict/services/rerank.service.ts) el corte
   `topCos < adaptiveMinScore` (0.8) se aplica **sobre el coseno semántico
   puro** y devuelve `[]` antes de que ninguna otra señal pueda intervenir. Si el
   usuario usa un término que el embedding sitúa por debajo de 0.8, la
   predicción es 0 herramientas y ninguna capa posterior puede arreglarlo. Este
   es el modo de fallo más grave y el que más importa corregir.
4. **Contaminación cruzada real y ya presente.** `QdrantService.synonymCache` es
   un `Map<string, string[]>` **indexado solo por token**, y `lookupSynonyms`
   busca en `query_synonyms` **sin filtro de tenant**. Un sinónimo cacheado
   mientras atiende al canal A se sirve tal cual al canal B. Hoy es de bajo
   impacto (la colección se puebla por fuera), pero cualquier aprendizaje léxico
   que escriba sinónimos lo convertiría en un fallo de aislamiento. **Debe
   corregirse antes de añadir aprendizaje** (§9.2, F0).
5. **No hay señal de resultado.** `POST /predict` devuelve herramientas y nadie
   informa de cuáles funcionaron. El ejecutor de `key-remtk` ya sabe distinguir
   éxito real (`TaskExecutorV5Service.succeeded()` inspecciona el sobre del
   payload), así que esa señal existe pero no viaja.

---

## 4. Qué NO se va a hacer

| Descartado | Motivo |
|---|---|
| Fine-tuning / LoRA sobre `multilingual-e5-small` | Coste GPU, minutos-horas, un modelo por canal, y un modelo global afectaría a todos los usuarios. Viola R1 y R4. |
| Reentrenar el índice Qdrant en cada alta | El alta ya indexa; un reentrenamiento masivo rompe el warm-up no bloqueante. |
| Learning-to-rank con batch (LambdaMART, XGBoost) | Requiere corpus etiquetado y ciclo offline. Viola R1. |
| Un modelo compartido entre canales con features de usuario | Rompe R4 por construcción. Cualquier estado global es un riesgo de fuga. |
| Aprendizaje sobre el texto crudo del turno | Viola R5 y ensucia el perfil con muletillas. Se aprende **sobre términos normalizados**. |

---

## 5. Comparativa de algoritmos

Notación: `k` = términos de la consulta (≤ 20), `p` = herramientas del canal que
contienen esos términos, `f` = features activas.

| # | Algoritmo | Coste por evento | Coste por consulta | Estado a persistir | Aislamiento | Veredicto |
|---|---|---|---|---|---|---|
| **A** | **Refuerzo léxico incremental** — contadores término→tool con IDF del canal y decaimiento exponencial | O(k) | O(Σ posting) | pequeño (términos×tools) | trivial por `scopeKey` | **Recomendado — Fase 1** |
| B | Naive Bayes multinomial por canal — `P(tool) · Π P(t\|tool)` | O(k) | O(Σ posting) | pequeño (conteos) | trivial | Complemento natural de A (Fase 4) |
| C | Regresión logística online (SGD / FTRL-Proximal) sobre features léxicas | O(f) | O(f) | medio (vector de pesos) | trivial | Fase 4 si A no alcanza |
| D | Learning-to-rank supervisado (LambdaMART / XGBoost) | batch completo | O(f) | grande | reentrenar por canal | Descartado (R1) |
| E | Fine-tuning / LoRA del encoder | GPU, offline | igual | GB | global | Descartado (R1, R4) |

**Por qué A.** El pedido dice literalmente que *"son más por palabras claves que
se entrena"*. A.4 hace exactamente eso con la mínima maquinaria posible: una
tabla hash por canal. No hay gradientes que estabilicen, no hay convergencia que
esperar, no hay época. Un evento nuevo se contabiliza en el acto y el efecto es
visible en la predicción siguiente. B y C son **incrementales sobre A**: reusan
el mismo perfil y el mismo índice invertido, así que no hay trabajo perdido si
después se decide subir de nivel.

---

## 6. Algoritmo propuesto: perfil léxico aprendido por canal

### 6.1 Estructuras

```ts
/** Un perfil por canal. Nunca se comparte entre canales. */
interface LexicalProfile {
  scopeKey: string;                      // tenant::agentId
  version: number;                       // se incrementa en cada seed
  tools: Map<string, ToolLexicon>;       // toolName → léxico de esa tool
  df: Map<string, number>;               // document frequency por término
  postings: Map<string, Set<string>>;    // término → tools que lo contienen
  events: number;                        // nº de señales observadas (guarda de cold start)
  updatedAt: number;
}

interface ToolLexicon {
  terms: Map<string, number>;            // término → peso w(t→tool)
  sources: Map<string, "seed" | "learned">;
  seededHash: string;                    // sha1 de la semilla (evita re-embeder/re-sembrar)
  lastDecay: number;                     // marca para decaimiento perezoso
}
```

Índice invertido (`postings`) para que puntuar sea `O(Σ_t |postings[t]|)` y no
`O(N·k)` sobre todo el catálogo.

### 6.2 Puntuación (lectura)

Idf del propio canal — un término que aparece en todas las tools del canal no
discrimina:

```
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
aprendido_i = Σ_{t ∈ Q} idf(t) · w(t → tool_i)
```

con `Q` = términos normalizados de la consulta (los mismos que ya produce
`extractQueryKeywords` / `extractKeywordsRake`), y `w(...)` decaído
perezosamente en el momento de la lectura.

Normalización a `[0,1]` **dividiendo por la masa informativa de la consulta**
(`Σ_{t ∈ Q con postings} idf(t)`, topada en 1), y puntuando **solo los términos
con `sources[t] === "learned"`**.

> **Corrección medida (2026-09-28).** La versión original —recorrer todos los
> postings y normalizar por el máximo de la tanda— estaba refutada por el propio
> §6.5. Con la semilla del catálogo en el mismo índice, la lectura devolvía score
> aprendido para las 196 tools del canal (`learnTerms=196`) y el máximo de la
> tanda regalaba el peso entero de la capa (`+learnWeight`) a la tool que había
> coincidido en dos términos de quince. En el turno 2 de
> `nest-2026-09-28 (2).log`, tres eventos de una subtarea *hermana* (otra
> subtarea del mismo plan, que repite el nombre de la entidad) bastaron para
> desalojar `mitumbes_item_crear` de la tanda de cinco. Con la lectura acotada a
> evidencia de uso y a su cobertura de la consulta, `learnTerms` cae a las tools
> realmente reforzadas y el refuerzo ya no puede desbancar a la herramienta que
> la consulta pide; cuando la evidencia cubre la consulta (§2-B de
> `tests/mitumbes-barrier.test.ts`) la capa sigue rescatando el flujo completo.

### 6.3 Actualización (escritura) — el "entrenamiento"

```
observar(tool_ganadora, Q, +1):
    para cada t ∈ Q:
        w(t→tool_ganadora) += η · idf(t)
        sources[t] = "learned"

observar(tool_perdedora, Q, −γ):
    para cada t ∈ Q:
        w(t→tool_perdedora) = max(0, w(t→tool_perdedora) − γ · idf(t))
```

Es **una pasada sobre los términos de la consulta** (≤ 20). No hay
retropropagación, no hay matriz, no hay época. El coste real de un evento es
indistinguible del ruido.

### 6.4 Decaimiento (evita que el perfil se fosilice)

Perezoso, O(1), aplicado por tool en la primera lectura/escritura tras `Δt`:

```
w ← w · exp(−λ · Δt)      // λ = LEARN_DECAY_LAMBDA por día
```

Los términos que caen por debajo de `LEARN_TERM_MIN_WEIGHT` se podan junto con
su entrada en `postings`. Así el perfil sigue el vocabulario vivo del canal en
lugar de arrastrar historia.

### 6.5 Cold start y no regresión (R7)

`learnWeightEfectivo = eventos >= LEARN_MIN_EVENTS ? LEARN_WEIGHT : 0`

Con el canal recién creado o sin señal, la capa aporta **exactamente cero** y la
predicción es bit a bit la de hoy. Es la garantía de que el aprendizaje no puede
empeorar nada mientras no tenga datos.

### 6.6 Integración en el ranking

En `RerankService.filter` y `RerankService.graphFilter` se añade el término
aprendido y se corrige el piso:

```
fused_i = base_i + keywordBoost · BM25_norm_i + learnWeightEfectivo · aprendido_norm_i
```

**Corrección del piso de relevancia (§3.4.3).** Hoy el piso se evalúa solo sobre
el coseno. Pasa a evaluarse sobre la señal fusionada:

```
relevancia = max_i ( semántico_i + learnWeightEfectivo · aprendido_norm_i )
si relevancia < adaptiveMinScore  →  0 tools
```

De este modo un término aprendido del usuario puede **rescatar** una
herramienta que el coseno dejaba en 0.79 y que hoy devolvía lista vacía. Sin
aprendizaje (`learnWeightEfectivo == 0`) la expresión es idéntica a la actual.

### 6.7 Presupuesto de latencia

| Operación | Coste | Objetivo |
|---|---|---|
| `seedScope` (alta de M tools) | `O(Σ términos)` ≈ 30 tokens/tool, sin ONNX | < 1 ms/tool |
| `observe` (un evento de feedback) | `O(k)` ≤ 20 ops de mapa | < 0.1 ms |
| `score` (una consulta) | `O(Σ_t |postings[t]|)`, acotado por `LEARN_MAX_POSTINGS` | < 1 ms p95 |
| Memoria | `O(nº términos × tools)` con poda | acotada por `LEARN_MAX_TERMS_PER_TOOL` |

Ninguna de las tres toca ONNX Runtime, así que no compiten con la cola de
inferencia (`high`/`low`) descrita en [embeddings.md](embeddings.md).

---

## 7. Disparador: cuando se agregan herramientas

`PredictionOrchestrator.registerTools()` ya hace `indexTools` (síncrono) y
`warmScope` (background, encadenado por `warmups`, prioridad baja). Se añade un
tercer paso **dentro de `warmScope`**, después del grafo:

```ts
const { recomputed, cached } = await this.keywords.upsertTools(scopeKey, tools);
await this.graphCache.buildTenantGraph(scopeKey, tools);
await this.lexical.seedScope(scopeKey, tools);        // ← NUEVO
```

Reglas de `seedScope`:

1. **Semilla de términos** por tool, reusando lo que ya existe:
   `toolKeywords(tool)` (tags + intentSummary) ∪ `extractToolKeywords(tool)` ∪
   `normalizeToken(name)` ∪ `normalizeToken(group)` ∪ `normalizeToken(category)`
   ∪ claves de `inputSchema` (normalizadas).
2. **Peso de semilla bajo** (`LEARN_SEED_WEIGHT` = 0.3) para que el uso real lo
   domine en pocos eventos.
3. **No pisar lo aprendido**: si `sources[t] === "learned"`, la semilla no
   reescribe el peso. Un re-registro del catálogo no borra el entrenamiento.
4. **Idempotencia**: `seededHash` (sha1 de la semilla + `version`) evita trabajo
   repetido cuando el catálogo se re-registra sin cambios — el mismo patrón que
   `KeywordService.upsertTools` y `computeVersionHash` del grafo.
5. **Podado de bajas**: las tools que estaban en el perfil y ya no están en el
   nuevo catálogo se eliminan de `tools`, de `postings` y de `df`. Sin esto
   quedarían "fantasmas" puntuando por términos huérfanos.
6. **Cero bloqueo**: corre en el warm-up de fondo con prioridad baja. El alta de
   herramientas nunca espera al aprendizaje.

---

## 8. Señal de refuerzo

Aprender requiere saber qué predicción salió bien. Hoy esa información existe en
`key-remtk` pero no viaja.

### 8.1 Señal explícita (principal)

Nuevo endpoint en el puerto 6776:

```
POST /predict/feedback
{ tenant, agentId?, sessionId, text, used: string[], rejected?: string[] }
→ { ok: true, events: number }
```

`used` = herramientas cuya ejecución **confirma éxito**; `rejected` = las que se
predijeron y fallaron o no se usaron.

En `key-remtk` el emisor natural es `TaskExecutorV5Service`, que ya distingue
éxito real de falso éxito con `succeeded()`. El feedback debe emitirse **solo con
herramientas que ejecutaron y devolvieron éxito**, nunca con las meramente
predichas: usar "fue predicha" como señal positiva cierra el ciclo de
realimentación y refuerza los errores del propio modelo.

Contrato del cliente a extender:
[prediction-client.service.ts](../../key-remtk/src/core/ai-core/v4/services/prediction-client.service.ts)
y [prediction.types.ts](../../key-remtk/src/core/ai-core/v4/types/prediction.types.ts).

### 8.2 Señal implícita (complementaria)

Cuando en la misma sesión llega un `NewQuery` inmediatamente después de un
`NewQuery` y las herramientas predichas no reaparecen en el `priorPlan` del
turno siguiente, se emite una **penalización débil** (`−γ_weak`) sobre los
términos que llevaron a esas herramientas. No requiere cambios en el
consumidor y cubre el caso "el usuario reformuló porque no era eso".

El disparador es **`topicSim < TOPIC_SHIFT_THRESHOLD`**, donde `topicSim` es el
coseno medido del turno contra `z_{t-1}` — **no** `topicShift`. La razón es que
`topicShift` también lo alza la puerta de coherencia (λ) cuando la sesión viene
de un tema denso, y tratar eso como "el usuario cambió de tema" penaliza las
herramientas del objetivo que el propio plan está ejecutando.

Evidencia medida (reproducción del log de Nest,
`tests/e2e-nest-log-reproduction.test.ts`): en el turno del planificador,
`sim=0.866` frente a `umbral=0.86` con `lambda=0.05` → `topicShift = true` por λ,
no por deriva temática. La penalización disparada ahí degradó
`mitumbes_item_crear` (la herramienta del objetivo) en el canal, y la 2ª pasada
del planificador perdía la mutación objetivo. Con `topicSim` como disparador la
penalización ya no fire y el pre-plan DAG sobrevive al juicio.

### 8.3 Desactivación

`LEARN_ENABLED=false` desactiva `seedScope`, `observe` y el término de scoring.
El sistema vuelve al comportamiento actual sin tocar el perfil almacenado.

---

## 9. Aislamiento multi-usuario

### 9.1 Garantía por construcción

Toda la capa vive dentro de `Map<scopeKey, LexicalProfile>` y se resuelve con
`resolveScopeKey(tenant, agentId)` — el mismo mecanismo que ya particiona
catálogos, grafos y embeddings de keywords. Un evento del canal A escribe
**únicamente** en el perfil de A. No hay ningún contador, caché ni peso
compartido entre canales.

### 9.2 Deuda a saldar antes de aprender (F0)

| Riesgo actual | Corrección |
|---|---|
| `synonymCache: Map<token, string[]>` global | Pasar a `Map<scopeKey, Map<token, string[]>>`; `expandSynonyms` recibe el `scopeKey` |
| `lookupSynonyms` busca en `query_synonyms` sin filtro | Añadir `must: [{ key: "synonyms_tenant", match: { value: scopeKey } }]` o colección por scope |
| `expandSynonyms(query)` público en el puerto 6775 no recibe canal | Añadir `tenant`/`agentId` al contrato de `POST /tools/synonyms` |

Sin esta corrección, el aislamiento sería aparente: el perfil léxico estaría
bien particionado, pero los sinónimos que escriba seguirían cruzando canales.

### 9.3 Si se persiste (Fase 4)

Colección Qdrant nueva `tool_lexicon`, sparse, sobre el mismo patrón de
`mcp_tools`:

- ID: `uuidv5('lexicon:{scopeKey}:{toolName}')` — determinista, como
  `mcp-tool:{group}:{name}`.
- Payload: `{ tenant: scopeKey, toolName, terms: [{term, weight}], updatedAt }`.
  La clave se llama `tenant` a propósito: el `scopeKey` completo viaja ahí.
- **Todo** acceso lleva filtro `must: [{ key: "tenant", match: { value: scopeKey } }]`.
  Nunca una búsqueda sin filtro, siguiendo el patrón que `searchMemories` ya
  aplica con `userId` + `agentId`.
- Carga perezosa en el primer `predict` o `registerTools` del canal; escritura
  diferida con debounce para no golpear Qdrant en cada evento.

### 9.4 Test de aislamiento (obligatorio)

1. Canal A: registrar tools, observar 10 eventos con término `X` → tool `Y`.
2. Canal B: registrar el mismo catálogo, **sin** observaciones.
3. Predicar el mismo texto con `X` en ambos canales.
4. Assert: la predicción de B es **idéntica** a la de B antes del entrenamiento de
   A, y B no tiene ninguna entrada en su perfil. Igual para sinónimos.

---

## 10. Configuración

Nuevas variables en [config.ts](../src/config.ts) con defaults en
`shared/constants/predict/general.predict.ts`, siguiendo el patrón existente
(`keywordBoostDefault`, `gapThresholdDefault`, …).

| Variable | Default | Descripción |
|---|---|---|
| `LEARN_ENABLED` | `true` | Interruptor maestro de la capa. |
| `LEARN_WEIGHT` | `0.25` | Peso del score aprendido en la fusión (comparable a `KEYWORD_BOOST` = 0.15). |
| `LEARN_ETA` | `0.5` | Tasa de refuerzo positivo. |
| `LEARN_NEGATIVE_GAMMA` | `0.15` | Tasa de penalización negativa. |
| `LEARN_DECAY_LAMBDA` | `0.02` | Decaimiento exponencial por día. |
| `LEARN_MIN_EVENTS` | `3` | Eventos mínimos del canal antes de que la capa puntúe. |
| `LEARN_SEED_WEIGHT` | `0.3` | Peso inicial de los términos de la semilla. |
| `LEARN_MAX_TERMS_PER_TOOL` | `64` | Tope de términos por herramienta (poda por peso). |
| `LEARN_TERM_MIN_WEIGHT` | `0.05` | Umbral de poda de términos. |
| `LEARN_MAX_POSTINGS` | `200` | Tope de acumulaciones por consulta (cota dura de latencia). |
| `LEARN_PERSIST` | `false` | Persistencia en `tool_lexicon` (Fase 4). |

---

## 11. Plan de implementación por fases

### F0 — Aislamiento y contrato (bloqueante)
- Corregir `synonymCache` y `lookupSynonyms` a por-scope (§9.2).
- Añadir `tenant`/`agentId` al contrato de `POST /tools/synonyms`.
- Test de aislamiento de sinónimos entre dos canales.

**Ficheros:** `src/qdrant/qdrant-service.ts`, `src/qdrant/qdrant.controller.ts`, `tests/`.

### F1 — Perfil léxico en memoria + semilla
- Nuevo `src/predict/services/lexical-profile.service.ts`: `seedScope`, `observe`, `score`, `prune`.
- Enganche en `warmScope` de `PredictionOrchestrator`.
- Normalización de términos reusando `normalizeToken` / `extractQueryKeywords` de `src/predict/keywords.ts`.
- Sin efecto en la predicción todavía (`learnWeightEfectivo = 0`): se puede medir el perfil en `/debug` antes de activarlo.

**Ficheros:** `src/predict/services/lexical-profile.service.ts` (nuevo), `src/predict/orchestrator.ts`, `src/shared/interfaces/graph.interface.ts` (o nuevo `lexical.interface.ts`), `src/shared/factory/orchestrator.factory.ts`.

### F2 — Señal de refuerzo
- `POST /predict/feedback` en el controlador + `observe` en el orquestador.
- Cliente `feedback()` en `prediction-client.service.ts` de `key-remtk`.
- Emisión desde `TaskExecutorV5Service` **solo** con herramientas que pasaron `succeeded()`.
- Señal implícita de reformulación.

**Ficheros:** `src/predict/predict.controller.ts`, `src/predict/orchestrator.ts`; en `key-remtk`: `src/core/ai-core/v4/services/prediction-client.service.ts`, `src/core/ai-core/v4/types/prediction.types.ts`, `src/core/aimcp/v5/services/task-executor-v5.service.ts`.

### F3 — Integración en el ranking
- Término `learnWeightEfectivo · aprendido_norm` en `RerankService.filter` y `graphFilter`.
- Piso de relevancia sobre la señal fusionada (§6.6).
- Observabilidad: score base vs aprendido y contribución por término en `/debug`.

**Ficheros:** `src/predict/services/rerank.service.ts`, `src/predict/orchestrator.ts`, `src/predict/helper/debugger.helper.ts`, `src/config.ts`, `src/shared/constants/predict/general.predict.ts`.

### F4 — Persistencia por canal (opcional)
- Colección `tool_lexicon`, `indexLexicon` / `searchLexicon` con filtro por `tenant`.
- Carga perezosa + escritura diferida con debounce.
- Si el perfil no alcanza el objetivo de precisión, evaluar B (Naive Bayes) o C (logística online) **sobre el mismo perfil**.

**Ficheros:** `src/qdrant/qdrant-service.ts`, `src/qdrant/qdrant-client.ts`, `src/shared/constants/qdrant/general.constant.ts`.

### F5 — Validación
- Golden set y métricas (§12).

---

## 12. Validación y métricas

### 12.1 Golden set

Conjunto de pares `prompt → herramienta(s) esperada(s)` construido con los casos
reales observados en MiTumbes (`key-remtk/tests/docs/`). Debe cubrir, como
mínimo, las tres familias donde el catálogo estático falla:

1. **Vocabulario del usuario ≠ vocabulario del catálogo** — el usuario nombra el
   trámite con su palabra local, el catálogo con la formal.
2. **Prompts parafraseados** — misma intención, sin ningún término literal de la
   descripción.
3. **Prompts densos multi-paso** — varias tools correctas; evalúa el umbral
   adaptativo, no solo el top-1.

### 12.2 Métricas

| Métrica | Qué mide |
|---|---|
| **Tasa de 0 tools** | El fallo más grave: el piso de relevancia vacía la respuesta. Es la métrica que la §6.6 debe mover. |
| P@1 / P@3 | Precisión del top de la lista. |
| Recall@5 | Cobertura en prompts multi-paso. |
| MRR | Calidad global del orden. |
| Latencia p50/p95 de `/predict` | Debe permanecer dentro del presupuesto (§6.7). |

### 12.3 Comparación

| Escenario | Descripción |
|---|---|
| **A — baseline** | `LEARN_ENABLED=false`, comportamiento actual. |
| **B — tras N eventos** | Mismo golden set, canal entrenado con N observaciones. Curva de mejora vs N. |
| **C — MiTumbes** | Resultado de MiTumbes-Server sobre el mismo golden set. Es la referencia a superar (R6). |

### 12.4 Criterio de aceptación

1. Reduce la **tasa de 0 tools** frente a A en los tres subconjuntos.
2. **MRR ≥ A** en cada subconjunto y **estrictamente mejor** en el agregado.
3. Ninguna métrica cae más de 2% respecto a A en ningún subconjunto (no regresión).
4. **Cero contaminación** en el test de aislamiento (§9.4).
5. Latencia p95 de `/predict` dentro del presupuesto.

---

## 13. Riesgos

| Riesgo | Mitigación |
|---|---|
| Contaminación cruzada entre canales | Partición por `scopeKey` + filtro obligatorio por `tenant` en Qdrant + test §9.4 + F0 |
| Ciclo de realimentación (el sistema refuerza sus propios errores) | La señal positiva exige **éxito real** (`succeeded()`), no "fue predicha" |
| Sobreajuste a un término ruidoso | Decaimiento exponencial + γ negativo + `LEARN_TERM_MIN_WEIGHT` + cap de términos |
| Regresión en canales sin datos | `learnWeightEfectivo = 0` hasta `LEARN_MIN_EVENTS` (§6.5) |
| Crecimiento de memoria | Poda por peso y `LEARN_MAX_TERMS_PER_TOOL` |
| Latencia por encima del presupuesto | Índice invertido + `LEARN_MAX_POSTINGS`; la capa nunca toca ONNX |
| Herramientas eliminadas que siguen puntuando | Podado de bajas en `seedScope` (§7.5) |
| Deriva del catálogo que borra el aprendizaje | La semilla no pisa `sources[t] === "learned"` (§7.3) |

---

## 14. Anexos

### 14.1 Contratos HTTP nuevos

```
POST /predict/feedback                                  (puerto 6776)
  { tenant, agentId?, sessionId, text,
    used: string[], rejected?: string[] }
  → { ok: true, events: number }
```

### 14.2 Puntos de enganche exactos

| Punto | Fichero | Cambio |
|---|---|---|
| Alta de herramientas | `src/predict/orchestrator.ts` → `warmScope` | `lexical.seedScope(scopeKey, tools)` |
| Ranking plano | `src/predict/services/rerank.service.ts` → `filter` | sumar `aprendido_norm` |
| Ranking por grafo | `src/predict/services/rerank.service.ts` → `graphFilter` | sumar `aprendido_norm` |
| Piso de relevancia | ambos métodos | evaluar sobre la señal fusionada |
| Canal | `src/shared/scope.ts` → `resolveScopeKey` | sin cambios, ya es la clave de partición |

### 14.3 Referencias

- [pipeline-prediccion.md](pipeline-prediccion.md) — flujo detallado de `/predict`.
- [arquitectura.md](arquitectura.md) — módulos y creación del sistema.
- [qdrant.md](qdrant.md) — colecciones, BM25 sparse e IDs uuidv5.
- [configuracion.md](configuracion.md) — variables de entorno y defaults.
