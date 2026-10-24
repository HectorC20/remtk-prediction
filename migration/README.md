# Suite de Migración Arquitectónica: remtk-prediction a Rust Nativo

## 1. Resumen Ejecutivo

Este directorio contiene la especificación de ingeniería, planes de transición, diseño de arquitectura de sistemas y guías operativas para migrar el microservicio de inferencia y juicio de herramientas **`remtk-prediction`** desde su implementación actual en **Node.js (NestJS + onnxruntime-node + V8)** hacia un binario nativo en **Rust (Tokio + Axum + ort v2 + tokenizers)**.

El objetivo central de esta migración es lograr una **reducción radical en el consumo de memoria RAM (< 90 MB RSS total frente a los 550 MB – 1,200 MB en Node.js)**, eliminando pausas de Garbage Collection, eliminando el marshaling de buffers a través de N-API, y aprovechando tensores Zero-Copy respaldados por `mmap`. Al mismo tiempo, se garantiza **paridad matemática estricta (tolerancia $\le 10^{-4}$)** con los contratos HTTP existentes en los puertos `6776` (Juicio y Predicción), `6777` (Embeddings) y `6775` (Qdrant Proxy).

```
   ESTADO ACTUAL (Node.js / V8 / NestJS)              MIGRACIÓN NATIVA RUST (Tokio / Axum / ort)
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│  • RSS en reposo: ~550 MB                    │    │  • RSS en reposo: ~45 MB                     │
│  • RSS bajo carga: 850 MB – 1,200 MB         │    │  • RSS bajo carga máxima: < 88 MB            │
│  • P50: 18 ms | P99: 85 ms (GC pauses)       │ ──►│  • P50: 3.2 ms | P99: 7.4 ms (Sin GC pauses) │
│  • N-API marshaling (C++ <-> JS Float32)     │    │  • Zero-Copy (ndarray::ArrayView -> ort)     │
│  • 3 procesos / apps Nest independientes     │    │  • 1 solo binario (3 puertos concurrentes)   │
│  • Imagen Docker: ~1.2 GB                    │    │  • Imagen Docker distroless: < 45 MB         │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
```

---

## 2. Diagnóstico Técnico Comparativo

| Dimensión de Rendimiento | Node.js (Estado Actual) | Rust Nativo (Objetivo de Migración) | Factor de Mejora |
| :--- | :--- | :--- | :--- |
| **Consumo de RAM (RSS en Reposo)** | $550\text{ MB} - 680\text{ MB}$ | **$45\text{ MB} - 52\text{ MB}$** | **$12\times$ menos RAM** |
| **Consumo de RAM (Pico con 100 req/s)** | $850\text{ MB} - 1,200\text{ MB}$ | **$< 88\text{ MB}$** | **$10\times - 14\times$ menos RAM** |
| **Latencia de Juicio MoE (P50)** | $18.4\text{ ms}$ | **$3.2\text{ ms}$** | **$5.7\times$ más rápido** |
| **Latencia de Juicio MoE (P99)** | $85.0\text{ ms}$ *(inducido por GC)* | **$7.4\text{ ms}$** *(determinista)* | **$11.5\times$ más predecible** |
| **Tiempo de Arranque (Cold Start)** | $3,800\text{ ms} - 4,500\text{ ms}$ | **$< 45\text{ ms}$** | **$85\times$ más rápido** |
| **Manejo de Tensores en Memoria** | Copia por N-API (`Float32Array`) | Zero-Copy (`ArrayView` / `mmap`) | **$0\text{ bytes}$ copiados** |
| **Caché de Herramientas ($150\text{ tools}$)** | Objetos JS + Float32 en V8 Heap | FP16/INT8 en `DashMap` contiguo | **$4\times$ mayor densidad** |
| **Tamaño de Imagen Docker** | $1.2\text{ GB} - 1.4\text{ GB}$ | **$< 45\text{ MB}$** | **$28\times$ más liviana** |
| **Arquitectura de Servidor** | 3 instancias Express/Nest | 1 binario Tokio (3 listeners TCP) | **Consolidación total** |

---

## 3. Principios Rectores de la Migración

1. **Prioridad Absoluta a la Memoria (< 90 MB RSS):**
   Ningún componente puede realizar copias superfluas de tensores en el heap. Los pesos de los modelos ONNX (`multilingual-e5-small` y `remtk-juicio-moe-int8`) se cargan mediante páginas de memoria mapeada (`mmap`) compartidas y de solo lectura.
2. **Paridad de Contratos 1:1:**
   Los endpoints `/predict`, `/spatial/predict`, `/predict/route`, `/memory/predict`, `/tools`, `/embed`, `/models` y `/tools/search` deben responder con esquemas JSON byte-a-byte compatibles con los clientes actuales.
3. **Preservación del Pipeline MoE de 3 Etapas:**
   El pipeline de 2 llamadas unificadas (extracción de tokens con `e5-small` + cabezal de decisión de 3 etapas `remtk-juicio-moe-int8`) se implementa directamente en Rust sobre la crate `ort` v2.0+, manteniendo el enrutamiento Top-2, los 6 expertos de interacción y la calibración set-transformer.
4. **Concurrencia sin Bloqueo:**
   El bucle asíncrono de Tokio atiende las conexiones HTTP en los 3 puertos sin interrupción. Las inferencias intensivas en CPU de ONNX Runtime se despachan a un thread pool dedicado (`tokio::task::spawn_blocking` o canal mpsc acotado) para garantizar que las peticiones ligeras de `/health` o `/models` tengan latencia sub-milisegundo.

---

## 4. Mapa de Navegación de la Documentación

La suite está organizada en 4 dominios estructurales:

```
migration/
├── README.md                                             <- (Este documento) Resumen ejecutivo y mapa
├── 00_diagnostico_y_presupuesto_ram.md                   <- Análisis forense de V8 vs Presupuesto en Rust
├── plan/
│   ├── plan_01_hoja_de_ruta_migracion.md                 <- Fases 1 a 4, hitos y corte de tráfico
│   ├── plan_02_presupuesto_y_benchmark_ram.md           <- Metodología con jemalloc, dhat-rs y SLAs
│   └── plan_03_estrategia_paridad_y_tests.md             <- Protocolo para los 17 suites de tests
├── architecture/
│   ├── arquitectura_01_topologia_servidor_rust.md        <- Tokio, Axum, 3 puertos en un solo binario
│   ├── arquitectura_02_gestion_memoria_y_zero_copy.md    <- Pipeline Zero-Copy y Tool Cache en DashMap
│   ├── arquitectura_03_pipeline_moe_y_e5.md              <- Implementación del MoE de 3 etapas en ort v2
│   ├── arquitectura_04_contratos_http_y_serializacion.md <- Serde structs, errores con thiserror y DTOs
│   └── arquitectura_05_monolito_modular.md               <- Monolito Modular (core, support, shared, api)
├── guide/
│   ├── guia_01_cargo_setup_y_optimizaciones.md           <- Cargo.toml, perfiles release y flags AVX2
│   └── guia_02_despliegue_docker_ultraligero.md          <- Dockerfile multi-stage distroless (< 45 MB)
└── deployment/
    └── skill_guia_operativa_migracion_rust.md            <- Skill operativo de ejecución paso a paso
```

### Guía de Lectura según Rol

* **Arquitecto de Sistemas / Líder Técnico:**
  Comenzar por `00_diagnostico_y_presupuesto_ram.md` y `arquitectura_01_topologia_servidor_rust.md` para validar las restricciones de hardware y la topología de procesos.
* **Ingeniero de Machine Learning / Inferencia:**
  Consultar `arquitectura_03_pipeline_moe_y_e5.md` y `arquitectura_02_gestion_memoria_y_zero_copy.md` para comprender la extracción de tensores, el pre-cálculo de herramientas y el grafo MoE.
* **Desarrollador Backend / Implementador Rust:**
  Seguir `plan_01_hoja_de_ruta_migracion.md`, `arquitectura_04_contratos_http_y_serializacion.md` y `guia_01_cargo_setup_y_optimizaciones.md`.
* **Ingeniero de QA / Testing:**
  Estudiar `plan_03_estrategia_paridad_y_tests.md` para configurar la ejecución de los 17 suites de tests contra el binario compilado.
* **Ingeniero DevOps / SRE / Operador:**
  Revisar `guia_02_despliegue_docker_ultraligero.md` y ejecutar las tareas usando `deployment/skill_guia_operativa_migracion_rust.md`.
