# Plan 04: Migración en `src/juicio/`, Validación de Regresión y Puesta en Producción

## 1. Refactorización Limpia dentro de `src/juicio/`

El reemplazo en el código TypeScript se realiza manteniendo intacta la interfaz pública que consume `Orchestrator` (`src/predict/orchestrator.ts`).

### Mapa de Transición de Archivos

| Archivo Actual en `src/juicio/` | Estado tras la Migración | Rol en la Nueva Arquitectura |
| :--- | :--- | :--- |
| `juicio.service.ts` | **Se mantiene (Simplificado)** | Orquesta la preparación de tensores desde `EmbeddingService` y llama a `MoEDecisionService`. |
| `services/moe-decision.service.ts` | **NUEVO** | Carga `models/remtk-juicio-moe-int8/model_quantized.onnx` vía `onnxruntime-node` y ejecuta las 3 etapas en un solo paso. |
| `services/estado-nli.service.ts` | **Deprecado / Eliminado** | Absorbido por la **Etapa 1** interna del grafo `.onnx`. |
| `services/cross-encoder.service.ts` | **Deprecado / Eliminado** | Absorbido por la **Etapa 2 (Expertos MoE)** interna del grafo `.onnx`. |
| `services/maxsim.service.ts` | **Integrado en el Grafo** | El operador `MaxSim` corre como una operación matricial nativa dentro de la Etapa 2 del `.onnx`. |
| `services/abstencion.service.ts` | **Integrado en el Grafo** | La entropía de Shannon, el margen y el umbral corren en la **Etapa 3** del `.onnx`. |

---

## 2. Flujo de Ejecución en Caliente en `POST /predict` (Puerto 6776)

1. **Indexación Previa (`POST /tools`):**
   * Al registrar las herramientas de un `tenant`, `EmbeddingService` calcula el tensor de tokens $(L_{\text{tool}}, 384)$ de cada herramienta (`MaxLen = 96`) y lo guarda en un `Float32Array` contiguo en memoria RAM por tenant.
   * Para $200$ herramientas, esta caché ocupa únicamente:
     $$200 \times 96 \times 384 \times 4\text{ bytes} \approx 29.4\text{ MB de RAM por tenant}$$
2. **Llegada de Consulta (`POST /predict`):**
   * Si es confirmación rápida (`ConfirmCacheService`), retorna en $<1\text{ ms}$.
   * Qdrant (Puerto `6775`) + `intentContext` + Etapa A (`MAX_CATEGORIES = 60`) filtran las $K$ herramientas candidatas (ej. $K = 80$).
   * `EmbeddingService` (`e5-small`) extrae los tensores de tokens de `text` + `keywords` y del último turno de `history`.
   * Se recorta el sub-tensor de las $K$ herramientas candidatas desde la caché en RAM ($0\text{ ms}$).
   * `MoEDecisionService` ejecuta `session.run()` pasando `(query_tokens, context_tokens, tools_tokens)`.
   * El modelo devuelve en $\sim 4\text{ ms}$:
     * `ranked_scores`: Array de longitud $K$ con scores calibrados en $[0, 1]$.
     * `estado_probs`: Probabilidades de `[TRANSICION, CONTINUACION, CORRECCION, ABANDONO]`.
     * `abstencion_prob`: Escalar en $[0, 1]$.
     * `topic_shift_score`: Escalar en $[0, 1]$ (reutilizable también en `/memory/predict`).

---

## 3. Criterios de Aceptación en la Suite de Tests (`pnpm test:all`)

Antes de eliminar los modelos antiguos de la carpeta `models/`, el nuevo `remtk-juicio-moe-int8` debe superar la batería completa de regresión:

1. **Cero Falsos Positivos en Escenarios Nulos:**
   * `tests/retorno-nulo-escenarios.test.ts` y `tests/webpage-prediction-null.test.ts` deben lograr $100\%$ de precisión (`tools: []` cuando `abstencion_prob > 0.5`).
2. **Resistencia al Estrés Adversarial y Párrafos Largos:**
   * `tests/casos-estres-adversariales.test.ts` y `tests/parrafos-contexto-estres.test.ts` deben mantener un margen $\Delta \ge 0.15$ entre la herramienta correcta y los distractores léxicos.
3. **Continuidad Multi-Turno:**
   * `tests/continuous-flow-intent.test.ts` y `tests/sequential-conversation-loop.test.ts` deben clasificar correctamente los giros de tema (`TRANSICION`) frente a los refinamientos (`CONTINUACION` / `CORRECCION`).
4. **Latencia P95:**
   * En `bench/`, el P95 de `/predict` con $150$ herramientas en catálogo debe mantenerse por debajo de **$18\text{ ms}$ en CPU**.