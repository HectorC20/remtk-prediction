# Plan 01: Hoja de Ruta de la Migración a Rust Nativo

## 1. Visión General y Cronograma Estratégico

La migración de `remtk-prediction` a Rust nativo se ejecuta en **4 fases secuenciales y desacopladas**. Cada fase cuenta con criterios de aceptación cuantitativos y puertas de control de calidad (*quality gates*). Esto asegura que no se introduzca código al servidor HTTP hasta que el núcleo de inferencia matemática garantice paridad numérica total con el modelo actual.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CRONOGRAMA DE MIGRACIÓN: 4 FASES HACIA PRODUCCIÓN                                      │
├────────────────────┬────────────────────┬─────────────────────┬────────────────────────┤
│ FASE 1             │ FASE 2             │ FASE 3              │ FASE 4                 │
│ Core Inferencia    │ Caché Zero-Copy    │ Servidor Axum       │ Paridad, Shadowing     │
│ ort + tokenizers   │ Multi-Tenant Tools │ 3 Puertos (6776/7/5)│ y Corte de Tráfico     │
├────────────────────┼────────────────────┼─────────────────────┼────────────────────────┤
│ • Sesiones ort v2  │ • ToolRegistry FP16│ • Listener 6776     │ • 17 suites tests TS   │
│ • Tokenizer BPE    │ • DashMap concurren│ • Listener 6777     │ • Dark Shadowing 10%   │
│ • Prefijos e5      │ • Slicing ArrayView│ • Listener 6775     │ • Verificación RAM     │
│ • 2 llamadas MoE   │ • POST /tools      │ • spawn_blocking    │ • Corte definitivo     │
└────────────────────┴────────────────────┴─────────────────────┴────────────────────────┘
```

---

## 2. Desglose Detallado por Fase

### Fase 1: Core de Inferencia ort + tokenizers
**Objetivo:** Construir una biblioteca interna en Rust (`remtk-inference-core`) que cargue los pesos ONNX y ejecute la tokenización y la inferencia sin ninguna dependencia de red ni servidor HTTP.

#### Actividades Clave:
1. **Configuración de Dependencias:** Integrar `ort = "2.0.0-rc.9"`, `tokenizers = "0.21"`, `ndarray = "0.16"`.
2. **Carga Segura de Modelos con `mmap`:**
   - Carga de `models/multilingual-e5-small-onnx/model.onnx` con memoria mapeada compartida.
   - Carga de `models/remtk-juicio-moe-int8/model_quantized.onnx` con optimizaciones de grafo AVX2/FMA.
3. **Carga y Normalización del Tokenizer:**
   - Cargar `tokenizer.json` de `e5-small`.
   - Implementar el formateador de prefijos:
     - `query: ` para texto de consulta principal e `intentContext`.
     - `passage: ` para definiciones de herramientas e ítems de catálogo.
4. **Pipeline de Inferencia de 2 Llamadas:**
   - **Llamada 1 (`e5-small`):** Obtener tensor de secuencia `last_hidden_state` $(1, L_q, 384)$ y vector pooled $(1, 384)$.
   - **Llamada 2 (`remtk-juicio-moe-int8`):** Cruzar `query_tokens`, `context_tokens` y `tools_tokens` en el grafo MoE.
5. **Tests Unitarios de Inferencia:**
   - Validar que un texto de prueba genere exactamente los mismos shapes y dimensiones tensoriales.

**Entregables:**
* Módulo `src/inference/engine.rs` y `src/inference/tokenizer.rs`.
* Suite de tests unitarios: `cargo test --lib test_inference_pipeline`.

---

### Fase 2: Caché de Herramientas y Zero-Copy Multi-Tenant
**Objetivo:** Implementar la estructura en memoria de alta densidad para almacenar las representaciones de herramientas por inquilino (`tenant`), reduciendo el consumo de RAM a la mitad mediante FP16 y permitiendo rebanado (*slicing*) de tensores en $< 0.05\text{ ms}$.

#### Actividades Clave:
1. **Diseño de `ToolRegistry` en Memoria:**
   - Almacenar definiciones de herramientas (`ToolDefinition`) junto a sus tensores pre-calculados de tokens $(L_t, 384)$ en `half::f16`.
   - Máscara booleana compacta `tool_mask` de $96\text{ bits}$ para omitir tokens de padding en la operación `MaxSim`.
2. **Estructura Concurrente Global:**
   - `Arc<DashMap<TenantId, Arc<ToolRegistry>>>` con acceso sin cerrojos pesados (*lock-free concurrent reads*).
3. **Pipeline de Indexación (`POST /tools` y `/tools/upsert`):**
   - Recepción de lote de herramientas en texto crudo.
   - Formateo canónico: `passage: {name} | {group} / {category} | {intentSummary} | {description} | tags: {tags}`.
   - Ejecución de `e5-small` en batch y conversión de `f32` a `f16`.
4. **Ensamblado Zero-Copy para `/predict`:**
   - Cuando el recall selecciona las $K$ candidatas, construir un `Array3<f32>` temporal mediante `ArrayView` directamente sobre la memoria contigua de las herramientas seleccionadas sin clonar búferes redundantes.

**Entregables:**
* Módulo `src/registry/mod.rs` y `src/registry/cache.rs`.
* Benchmarks de micro-latencia: `cargo bench --bench tool_slicing`.

---

### Fase 3: Servidor HTTP Axum en Puertos 6776, 6777 y 6775
**Objetivo:** Exponer la funcionalidad del motor a través de una API HTTP ultra-concurrente, escuchando en los 3 puertos requeridos y despachando las tareas intensivas a hilos de cómputo para no bloquear el reactor Tokio.

#### Actividades Clave:
1. **Topología Multi-Puerto:**
   - Inicializar 3 instancias de `axum::Router` unidas en un solo binario vía `tokio::try_join!`.
   - **Puerto 6776:** `/predict`, `/spatial/predict`, `/predict/route`, `/memory/predict`, `/tools`, `/tools/count`, `/health`, `/debug`.
   - **Puerto 6777:** `/embed`, `/models`, `/health`.
   - **Puerto 6775:** `/tools/upsert`, `/tools/count`, `/tools/search`, `/memories/search`, `/tools/synonyms`, `/health`, `/status`.
2. **Despacho No-Bloqueante:**
   - Las peticiones HTTP son aceptadas de inmediato por los hilos de I/O de Tokio.
   - La inferencia ONNX se envía a `tokio::task::spawn_blocking` con un canal acotado para evitar la degradación del reactor bajo ráfagas intensas.
3. **Middleware y Observabilidad con Tower:**
   - Límite de concurrencia adaptativo (`tower::limit::ConcurrencyLimitLayer`).
   - Trazabilidad estructurada con `tracing` y métricas Prometheus de latencia y uso de memoria.
   - Manejo de señales `SIGINT` y `SIGTERM` para apagado elegante (*graceful shutdown*).

**Entregables:**
* Módulos `src/server/mod.rs`, `src/server/predict.rs`, `src/server/embed.rs`, `src/server/qdrant.rs`.
* Binario compilable: `cargo build --release`.

---

### Fase 4: Paridad Numérica, Dark Shadowing y Corte de Tráfico
**Objetivo:** Validar que el servidor Rust cumpla con el 100% de los tests del ecosistema actual y desplegarlo en producción con riesgo cero.

#### Actividades Clave:
1. **Ejecución de los 17 Suites de Tests de Node.js:**
   - Levantar el binario Rust en local.
   - Ejecutar el comando de tests de Node.js:
     ```bash
     PORT_PREDICT=6776 PORT_EMBED=6777 PORT_QDRANT=6775 pnpm test
     ```
   - Validar que los 17 archivos de prueba (`tests/*.test.ts`) pasen en verde al 100%.
2. **Validación de Tolerancia Numérica:**
   - Embeddings: error cuadrático medio $MSE < 10^{-6}$ y diferencia máxima absoluta $\le 10^{-4}$.
   - Juicio MoE: diferencia en `rankedScores` $\le 10^{-3}$ y concordancia categórica del 100% en `estadoDominante` y `debeAbstenerse`.
3. **Tráfico en Modo Sombra (Dark Shadowing):**
   - Configurar un proxy inverso (Envoy o Nginx) para clonar el 10% del tráfico real hacia el contenedor de Rust en modo pasivo.
   - Comparar discrepancias en logs y verificar estabilidad de RAM en $< 88\text{ MB}$.
4. **Corte Definitivo (Cutover):**
   - Cambiar el balanceador de carga al 100% hacia Rust.
   - Mantener el contenedor Node.js en modo standby durante 48 horas como plan de contingencia.

---

## 3. Matriz de Definición de Terminado (Definition of Done)

| Fase | Hito Principal | Criterio de Aceptación Cuantitativo | Estado de Verificación |
| :--- | :--- | :--- | :--- |
| **Fase 1** | Inferencia MoE funcional | Latencia de inferencia pura $< 5\text{ ms}$ en CPU (4 cores). Error flotante $\le 10^{-4}$. | Automatizado en `cargo test` |
| **Fase 2** | Caché de herramientas FP16 | RAM para 150 tools $< 11\text{ MB}$. Slicing de $75$ herramientas candidatas en $< 0.1\text{ ms}$. | Automatizado en `cargo bench` |
| **Fase 3** | Servidor Axum 3 puertos | Throughput $> 1,200\text{ req/s}$. P99 $< 10\text{ ms}$. Peticiones `/health` responden en $< 0.5\text{ ms}$ bajo carga. | Benchmark con `k6` / `wrk` |
| **Fase 4** | Paridad total y corte | 17 suites de tests de TypeScript pasando en verde. RAM RSS $< 90\text{ MB}$ sostenida tras 100,000 peticiones. | CI/CD y Dark Shadowing |

---

## 4. Plan de Rollback (Contingencia Operativa)

Si durante la Fase 4 o el corte definitivo se detecta alguna de las siguientes anomalías:
1. Desviación en la selección de herramientas superior al $0.1\%$ frente a Node.js.
2. Fuga o incremento de RAM por encima de $110\text{ MB}$ RSS.
3. Latencia P99 superior a $25\text{ ms}$ sostenida por más de 60 segundos.

**Protocolo de Rollback:**
1. El balanceador de carga conmuta el $100\%$ del tráfico de regreso al servicio Node.js (que permanece caliente en standby) en menos de 5 segundos vía actualización de DNS o upstream HTTP.
2. Se extrae un volcado de memoria del proceso Rust mediante `gcore` o `dhat` para análisis post-mortem.
3. Se restaura el plan de revisión en la Fase correspondiente.
