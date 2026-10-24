# Plan 02: Metodología de Perfilado de Memoria, Benchmarking y SLAs

## 1. Estrategia del Asignador de Memoria Global (Global Allocator)

El asignador estándar de la biblioteca estándar de C (`glibc malloc` en Linux o `MSVCRT` en Windows) no está optimizado para microservicios de alta concurrencia con tensores y concurrencia multihilo. `glibc malloc` tiende a fragmentar el espacio de memoria virtual y retiene páginas libres en pools por hilo sin devolverlas al sistema operativo, lo que en Node.js contribuye al crecimiento de la RAM.

En Rust, reemplazamos el asignador global por **`tikv-jemallocator`** (para Linux/Docker en producción) o **`mimalloc`** (para entornos multiplataforma Windows/Linux).

```rust
// src/main.rs
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(target_env = "msvc")]
use mimalloc::MiMalloc;

#[cfg(target_env = "msvc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

### 1.1. Ajuste Fino de `jemalloc` para Cero Fragmentación
Configuramos `jemalloc` mediante la variable de entorno `MALLOC_CONF` para forzar la devolución activa de memoria al kernel de Linux:

```bash
# Variables de entorno en producción (Dockerfile / Kubernetes)
MALLOC_CONF="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:2000,abort_conf:true"
```

* **`background_thread:true`:** Habilita un hilo en segundo plano que purga páginas sucias de forma asíncrona sin penalizar los hilos de inferencia.
* **`dirty_decay_ms:1000`:** Las páginas de memoria marcadas como sucias se devuelven al sistema operativo en un tiempo medio de decaimiento de 1 segundo.
* **`muzzy_decay_ms:2000`:** Purga agresiva de páginas en estado intermedio (*muzzy*), evitando la acumulación de memoria virtual no mapeada.

---

## 2. Herramientas de Perfilado y Detección de Fugas

Para garantizar el cumplimiento estricto de la meta **$\text{RSS} < 90\text{ MB}$**, implementamos un protocolo de perfilado de 3 capas:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ PIRÁMIDE DE PERFILADO DE MEMORIA EN RUST                                         │
├────────────────────────────────┬─────────────────────────────────────────────────┤
│ [1] Análisis de Asignaciones   │ • dhat-rs (Heap Profiler integrado en tiempo de │
│     a Nivel de Código          │   compilación para detectar bytes acumulados)   │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ [2] Perfilado de Proceso       │ • heaptrack / Valgrind Massif (Seguimiento de   │
│     del Sistema Operativo      │   picos de memoria y procedencia de allocs)     │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ [3] Monitoreo en Producción    │ • /proc/self/smaps_rollup + Métricas Prometheus │
│     en Tiempo Real             │   (RSS, PSS, Private Dirty Pages, VM_Peak)      │
└────────────────────────────────┴─────────────────────────────────────────────────┘
```

### 2.1. Perfilado de Heap con `dhat-rs`
Durante la fase de desarrollo e integración continua, habilitamos el feature `dhat-heap`:

```rust
// En el binario de profiling: src/bin/profile_memory.rs
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // Ejecutar 50,000 peticiones simuladas de /predict con 150 herramientas...
}
```

Al finalizar la prueba, `dhat` genera un archivo `dhat-heap.json` que se visualiza en el navegador (`dh_view.html`), mostrando:
1. Puntos exactos del código donde se realizaron asignaciones de memoria que no fueron liberadas.
2. Número total de asignaciones (`total bytes`, `total blocks`).
3. El *high-water mark* (marca máxima de memoria alcanzada en el heap).

### 2.2. Seguimiento de Páginas Sucias del Sistema Operativo
En Linux, leemos `/proc/self/smaps_rollup` periódicamente a través de una tarea de telemetría:
* **`RssAnon` (Memoria Anónima Residente):** Refleja la memoria real asignada en el heap por Rust y ONNX Runtime. Debe permanecer por debajo de **$45\text{ MB}$**.
* **`RssFile` (Memoria de Archivos Mapeados):** Refleja las páginas de los modelos `.onnx` cargadas desde disco. Debe ser inferior a **$45\text{ MB}$** y es susceptible de ser descartada por el kernel si hay presión de memoria.
* **$\text{RSS Total} = \text{RssAnon} + \text{RssFile} \le 88.0\text{ MB}$.**

---

## 3. Acuerdos de Nivel de Servicio (SLAs) y Objetivos Cuantitativos

| Métrica de Rendimiento | Umbral Máximo Tolerado | Objetivo Óptimo (Target) | Método de Medición |
| :--- | :--- | :--- | :--- |
| **Latencia P50 (`/predict`)** | $\le 5.0\text{ ms}$ | **$3.0\text{ ms} - 3.5\text{ ms}$** | `k6` con 50 usuarios concurrentes |
| **Latencia P95 (`/predict`)** | $\le 8.5\text{ ms}$ | **$5.5\text{ ms} - 6.5\text{ ms}$** | `k6` con 50 usuarios concurrentes |
| **Latencia P99 (`/predict`)** | $\le 10.0\text{ ms}$ | **$7.0\text{ ms} - 7.5\text{ ms}$** | `k6` con 50 usuarios concurrentes |
| **Latencia P50 (`/embed`)** | $\le 4.0\text{ ms}$ | **$2.0\text{ ms} - 2.5\text{ ms}$** | `k6` con 100 usuarios concurrentes |
| **Memoria RSS en Reposo** | $\le 55\text{ MB}$ | **$45\text{ MB}$** | Lectura `/proc/self/status` |
| **Memoria RSS Pico (Estrés)** | $\le 90\text{ MB}$ | **$\le 85\text{ MB}$** | Prueba sostenida de 10 minutos |
| **Throughput por Instancia (4 Cores)** | $\ge 800\text{ req/s}$ | **$> 1,200\text{ req/s}$** | Carga máxima con `wrk` |
| **Tasa de Errores HTTP (5xx)** | $0.00\%$ | **$0.00\%$** | Cero tolerancia a panics |

---

## 4. Harness y Scripts de Benchmark

### 4.1. Prueba de Estrés y Medición de Memoria con `k6`
Guardar en `bench/k6_stress_rust.js`:

```javascript
import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "1m", target: 50 },   // Rampa de subida a 50 VU
    { duration: "8m", target: 100 },  // 8 minutos a 100 peticiones simultáneas
    { duration: "1m", target: 0 },    // Rampa de bajada
  ],
  thresholds: {
    http_req_duration: ["p(50)<5", "p(95)<8.5", "p(99)<10"],
    http_req_failed: ["rate<0.001"],
  },
};

const PAYLOAD_PREDICT = JSON.stringify({
  tenant: "bench-tenant",
  sessionId: "bench-sess-001",
  text: "Necesito generar el reporte mensual de inventario y exportarlo a PDF",
  source: "human",
  keywords: ["reporte", "inventario", "pdf", "exportar"],
  intentContext: "El usuario está en el módulo de contabilidad cerrando el mes",
});

export default function () {
  const params = {
    headers: {
      "Content-Type": "application/json",
      "key-remtk": "bench-secret-key",
    },
  };

  const res = http.post("http://localhost:6776/predict", PAYLOAD_PREDICT, params);
  check(res, {
    "status 200": (r) => r.status === 200,
    "has rankedScores": (r) => JSON.parse(r.body).rankedScores !== undefined,
  });

  sleep(0.01);
}
```

### 4.2. Microbenchmarks en Rust con `criterion`
Guardar en `benches/inference_bench.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use remtk_inference_core::{InferenceEngine, PredictInput};

fn bench_predict_pipeline(c: &mut Criterion) {
    let engine = InferenceEngine::new("models").expect("Fallo al inicializar modelos");
    let input = PredictInput {
        query_text: "Necesito generar el reporte mensual de inventario".to_string(),
        context_text: "El usuario está en el módulo contable".to_string(),
        tenant_id: "bench-tenant".to_string(),
    };

    c.bench_function("moe_inference_complete", |b| {
        b.iter(|| {
            let output = engine.evaluate(black_box(&input)).unwrap();
            black_box(output);
        });
    });
}

criterion_group!(benches, bench_predict_pipeline);
criterion_main!(benches);
```
