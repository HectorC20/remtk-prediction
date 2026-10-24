---
name: remtk-rust-migration-architect
description: Skill operativo de ingeniería de sistemas para ejecutar, verificar y desplegar la migración de remtk-prediction desde Node.js/NestJS hacia Rust nativo (Tokio, Axum, ort v2), garantizando un consumo de RAM inferior a 90 MB y paridad total de contratos.
---

# Skill Operativo: Arquitectura de Migración de `remtk-prediction` a Rust

## 1. Mapeo de Documentación de la Suite

Antes de ejecutar cualquier comando o cambio de código, consulta los documentos de referencia técnica según el área de trabajo:

| Documento | Ubicación | Propósito Clave |
| :--- | :--- | :--- |
| **Resumen Ejecutivo** | `migration/README.md` | Visión general, comparativa Node.js vs Rust y principios. |
| **Diagnóstico de RAM** | `migration/00_diagnostico_y_presupuesto_ram.md` | Desglose forense de V8 vs Presupuesto milimétrico (< 88 MB). |
| **Hoja de Ruta** | `migration/plan/plan_01_hoja_de_ruta_migracion.md` | Fases 1 a 4, hitos y plan de contingencia (rollback). |
| **Benchmarking y SLAs** | `migration/plan/plan_02_presupuesto_y_benchmark_ram.md` | jemalloc, profiling con dhat-rs y SLAs (< 5ms P50). |
| **Paridad y Tests** | `migration/plan/plan_03_estrategia_paridad_y_tests.md` | Protocolo para los 17 suites de tests de TypeScript. |
| **Topología Servidor** | `migration/architecture/arquitectura_01_topologia_servidor_rust.md` | Axum, 3 puertos (6776, 6777, 6775) y spawn_blocking. |
| **Gestión Zero-Copy** | `migration/architecture/arquitectura_02_gestion_memoria_y_zero_copy.md` | ArrayView, DashMap multi-tenant y compresión FP16. |
| **Pipeline MoE** | `migration/architecture/arquitectura_03_pipeline_moe_y_e5.md` | Pipeline 2 llamadas: e5-small tokens + MoE de 3 etapas. |
| **Contratos HTTP** | `migration/architecture/arquitectura_04_contratos_http_y_serializacion.md` | DTOs Serde, manejo de errores thiserror y Json handlers. |
| **Cargo y rustc** | `migration/guide/guia_01_cargo_setup_y_optimizaciones.md` | Cargo.toml, fat LTO, codegen-units=1 y flags AVX2. |
| **Docker Ultraligero** | `migration/guide/guia_02_despliegue_docker_ultraligero.md` | Imagen distroless (< 45 MB), no-root y docker-compose. |

---

## 2. Reglas de Oro de la Migración

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAS 5 REGLAS DE ORO DE LA MIGRACIÓN A RUST                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. PRESUPUESTO DE RAM NO NEGOCIABLE: El RSS nunca debe superar los 90 MB    │
│    bajo ninguna circunstancia operativa ni ráfaga de tráfico.               │
│ 2. CERO CLONACIÓN EN CALIENTE: Los tensores entran a ORT mediante           │
│    `ndarray::ArrayView` y `ort::Value::from_array` sin `to_vec()` ni clone. │
│ 3. NO BLOQUEAR EL REACTOR TOKIO: Toda llamada de inferencia ONNX debe       │
│    despacharse a través de `tokio::task::spawn_blocking`.                   │
│ 4. DISCIPLINA ESTRICTA DE PREFIJOS: Toda consulta debe llevar "query: " y   │
│    toda herramienta debe indexarse obligatoriamente con "passage: ".        │
│ 5. PARIDAD NUMÉRICA RIGUROSA: Embeddings con error absoluto <= 1e-4 y       │
│    scores de herramientas con tolerancia <= 1e-3.                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Checklist Secuencial Paso a Paso

### Paso 1: Configuración del Entorno de Compilación
- [ ] Instalar Rust Toolchain estable: `rustup update stable`.
- [ ] Descargar o enlazar la biblioteca compartida `libonnxruntime.so` (o `.dll` en Windows) versión 1.20+.
- [ ] Configurar `.cargo/config.toml` con las banderas de optimización `-C target-cpu=x86-64-v3`.
- [ ] Verificar que `models/multilingual-e5-small-onnx/model.onnx` y `models/remtk-juicio-moe-int8/model_quantized.onnx` estén disponibles en la ruta esperada.

### Paso 2: Validación del Core de Inferencia (`cargo test`)
- [ ] Implementar `ZeroCopyTokenizer` y probar la tokenización de oraciones simples y compuestas.
- [ ] Cargar las sesiones ONNX con `ort` v2.0 utilizando memoria mapeada (`mmap`).
- [ ] Ejecutar prueba de inferencia aislada validando que la Llamada 1 (`e5-small`) devuelva `(1, Lq, 384)`.
- [ ] Validar que la Llamada 2 (`remtk-juicio-moe-int8`) reciba `(1, Lq, 384)`, `(1, Lc, 384)` y `(K, 96, 384)` y produzca los 5 outputs del contrato.

### Paso 3: Implementación del Caché de Herramientas FP16
- [ ] Crear el struct `CachedTool` con `tokens_fp16: Vec<half::f16>`.
- [ ] Implementar la función de ensamblado de tensores de candidatas (`assemble_candidate_tensor`) y medir en benchmark que tome menos de $0.05\text{ ms}$.
- [ ] Inicializar el `Arc<DashMap<TenantId, Arc<ToolRegistry>>>` global en `AppState`.

### Paso 4: Construcción del Servidor Axum Multi-Puerto
- [ ] Configurar el router del Puerto 6776 (`/predict`, `/spatial/predict`, `/predict/route`, `/tools`, `/health`).
- [ ] Configurar el router del Puerto 6777 (`/embed`, `/models`, `/health`).
- [ ] Configurar el router del Puerto 6775 (`/tools/upsert`, `/tools/search`, `/health`, `/status`).
- [ ] Probar el arranque concurrente con `tokio::select!` y verificar que los tres puertos escuchen peticiones en paralelo.

### Paso 5: Certificación de Paridad contra los 17 Suites de Tests
- [ ] Arrancar el binario Rust en local: `cargo run --release`.
- [ ] Ejecutar la suite completa de Node.js:
  ```bash
  PORT_PREDICT=6776 PORT_EMBED=6777 PORT_QDRANT=6775 pnpm test
  ```
- [ ] Verificar que las 17 suites pasen al 100% en verde.
- [ ] Inspeccionar el log de memoria del proceso Rust: constatar que el RSS se mantenga en $< 75\text{ MB}$.

### Paso 6: Construcción de la Imagen Docker de Producción
- [ ] Construir la imagen multi-stage distroless:
  ```bash
  docker build -f Dockerfile.rust -t remtk-prediction:rust-v1 .
  ```
- [ ] Verificar el tamaño final de la imagen: `docker images remtk-prediction:rust-v1` (debe ser $< 45\text{ MB}$).
- [ ] Levantar el contenedor con límites estrictos de recursos (`docker compose -f docker-compose.rust.yml up -d`).
- [ ] Ejecutar el script de estrés `k6` durante 10 minutos y registrar que el RSS no sobrepase los $88\text{ MB}$.

### Paso 7: Despliegue en Modo Sombra y Conmutación de Tráfico
- [ ] Configurar el proxy inverso para duplicar el 10% del tráfico de producción hacia el contenedor Rust.
- [ ] Monitorear discrepancias de scores durante 24 horas.
- [ ] Conmutar el 100% del tráfico al puerto 6776 del servidor Rust.
- [ ] Mantener Node.js en standby durante 48 horas antes del desmantelamiento final.

---

## 4. Matriz de Diagnóstico y Resolución Rápida de Errores (Troubleshooting)

| Síntoma o Mensaje de Error | Causa Raíz Probable | Acción Correctiva Inmediata |
| :--- | :--- | :--- |
| **`ORT Error: Invalid dimension for input 'tools_tokens'`** | El tensor tridimensional no tiene forma `(K, 96, 384)`. Ocurre si la longitud de secuencia de una herramienta no se fijó a 96 con padding. | Verificar que `MAX_TOOL_TOKENS = 96` y que las secuencias más cortas se rellenen con ceros (`f16::ZERO`) hasta completar 96 tokens. |
| **`HTTP 500: Tokio reactor thread starvation`** | Inferencia de ONNX Runtime ejecutada directamente en un manejador `async fn` sin `spawn_blocking`. | Mover la llamada `engine.predict(...)` dentro de `tokio::task::spawn_blocking(move || { ... }).await??`. |
| **`RSS creep: Memoria supera los 95 MB tras miles de peticiones`** | El asignador de memoria retiene páginas sucias en pools por hilo o no purga de forma agresiva. | Confirmar que `tikv-jemallocator` esté activo y que `MALLOC_CONF` contenga `background_thread:true,dirty_decay_ms:1000`. |
| **`Divergencia en ranked_scores: scores entre 0.3 y 0.6 en vez de 0.8+`** | Se omitió la capa de re-escalado afín en la Etapa 3 del MoE o falta el prefijo `query: ` en la entrada. | Verificar que el texto de consulta incluya `"query: "` antes de tokenizar y que la salida del Set-Transformer proyecte al rango calibrado `[0.80, 0.98]`. |
| **`Error de carga: error while loading shared libraries: libonnxruntime.so`** | La biblioteca compartida de ONNX Runtime no está en la ruta del linker dinámico de Linux. | Configurar la variable de entorno `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` o exportar `ORT_DYLIB_PATH`. |
| **`Healthcheck timeout (P99 > 50 ms bajo carga pesada)`** | El pool de hilos de Tokio está saturado o el límite de concurrencia de Tower no está activo. | Incorporar `ConcurrencyLimitLayer::new(64)` en el router de Axum para encolar peticiones excedentes sin degradar el reactor. |
