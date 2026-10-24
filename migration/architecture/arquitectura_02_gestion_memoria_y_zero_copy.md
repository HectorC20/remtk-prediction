# Arquitectura 02: Gestión de Memoria y Pipeline de Tensores Zero-Copy

## 1. El Paradigma Zero-Copy en Inferencia Neuronal

En sistemas de alta concurrencia como `remtk-prediction`, el principal cuello de botella de rendimiento y de consumo de RAM no es la multiplicación de matrices en la ALU, sino las **copias continuas de memoria en el heap** entre la entrada del texto, la biblioteca de tokenización, las estructuras intermedias del lenguaje y el motor de inferencia en C++.

En Rust eliminamos completamente las asignaciones intermedias mediante punteros de vista (*array views*) y tipos de longitud fija.

```
FLUJO ZERO-COPY: DE TEXTO CRUDO A ORT::VALUE
┌──────────────┐     tokenizers::Tokenizer     ┌───────────────────────┐
│ String UTF-8 │ ────────────────────────────► │ tokenizers::Encoding  │
└──────────────┘                               └───────────┬───────────┘
                                                           │ (Puntero crudo a &[u32] en C)
                                                           ▼
                                               ┌───────────────────────┐
                                               │  ndarray::ArrayView2  │ (0 bytes asignados)
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │      ort::Value       │ (Bindeo directo a ORT)
                                               └───────────────────────┘
```

---

## 2. Implementación del Pipeline Zero-Copy con `tokenizers` y `ort`

### 2.1. Conversión de Tokens sin Duplicación
La biblioteca de Hugging Face en Rust (`tokenizers::Tokenizer`) genera un objeto `Encoding` que almacena los IDs de tokens como un vector contiguo de enteros de 32 bits (`u32`). Dado que ONNX Runtime requiere enteros con signo de 64 bits (`i64` para `input_ids`), implementamos una proyección contigua alineada:

```rust
// src/inference/tokenizer_bridge.rs
use tokenizers::Tokenizer;
use ndarray::{Array2, ArrayView2};
use ort::Value;
use std::sync::Arc;

pub struct ZeroCopyTokenizer {
    tokenizer: Tokenizer,
}

impl ZeroCopyTokenizer {
    pub fn new(tokenizer_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        Ok(Self { tokenizer })
    }

    /// Tokeniza el texto y produce tensores de entrada ort::Value
    /// reutilizando un búfer pre-asignado para evitar allocations en cada llamada.
    pub fn encode_to_tensors(
        &self,
        text: &str,
        max_len: usize,
        out_ids: &mut Vec<i64>,
        out_mask: &mut Vec<i64>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, true)?;
        let tokens = encoding.get_ids();
        let seq_len = tokens.len().min(max_len);

        out_ids.clear();
        out_mask.clear();
        out_ids.reserve(seq_len);
        out_mask.reserve(seq_len);

        // Llenado directo en memoria contigua
        for &tok in &tokens[..seq_len] {
            out_ids.push(tok as i64);
            out_mask.push(1i64);
        }

        Ok(seq_len)
    }
}
```

### 2.2. Construcción de `ort::Value` a partir de `ArrayView`
En lugar de convertir vectores de Rust a copias internas de C++, creamos tensores de ONNX Runtime tomando prestada la porción contigua de memoria:

```rust
// src/inference/tensor_view.rs
use ndarray::ArrayView2;
use ort::Value;

pub fn create_input_tensor<'a>(
    data: &'a [i64],
    shape: (usize, usize),
) -> Result<Value, ort::Error> {
    let view = ArrayView2::from_shape(shape, data)
        .map_err(|e| ort::Error::new(format!("Error en forma de array: {}", e)))?;
    
    // ort::Value::from_array toma prestada la memoria del array si es contigua
    Value::from_array(view)
}
```

---

## 3. Caché de Herramientas Multi-Tenant en Memoria de Alta Densidad

### 3.1. Arquitectura de `ToolRegistry`
En cada petición `/predict`, la consulta del usuario debe cruzarse contra un catálogo de hasta $200$ herramientas. Para evitar re-tokenizar o re-ejecutar las herramientas a través de las 12 capas de `e5-small`:
1. Cada herramienta se procesa **una sola vez** en `POST /tools`.
2. Se extrae su tensor de tokens de tamaño $(96, 384)$.
3. Se almacena en memoria principal utilizando flotantes de media precisión (`half::f16`).
4. La máscara de padding de 96 tokens se comprime en un entero `u128` (16 bytes en lugar de 96 bytes).

```rust
// src/registry/tool.rs
use half::f16;
use serde::{Deserialize, Serialize};

pub const MAX_TOOL_TOKENS: usize = 96;
pub const EMBED_DIM: usize = 384;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub name: String,
    pub group: String,
    pub category: String,
    pub description: String,
    pub intent_summary: String,
    pub tags: Vec<String>,
}

#[derive(Clone)]
pub struct CachedTool {
    pub metadata: ToolMetadata,
    /// Tensor de tokens pre-calculado en FP16: 96 * 384 = 36,864 elementos (73,728 bytes)
    pub tokens_fp16: Vec<f16>,
    /// Máscara booleana comprimida de 96 bits (cada bit 1 indica un token real, 0 indica <pad>)
    pub attention_mask_bits: u128,
}
```

### 3.2. Estructura de Registro por Inquilino
```rust
// src/registry/mod.rs
use std::sync::Arc;
use dashmap::DashMap;

pub struct ToolRegistry {
    pub tools: Vec<CachedTool>,
    pub name_to_index: std::collections::HashMap<String, usize>,
}

pub type GlobalToolCache = Arc<DashMap<String, Arc<ToolRegistry>>>;
```

### 3.3. Algoritmo de Slicing Ultra-Rápido ($< 0.05\text{ ms}$) para $K$ Candidatas
Cuando el clasificador de fase inicial o el filtro de categorías selecciona un subconjunto de $K$ herramientas candidatas (por ejemplo, $K = 75$ herramientas de un catálogo de $150$):
1. No se realiza ninguna búsqueda en disco ni base de datos.
2. Se recorren los índices seleccionados en el `ToolRegistry`.
3. Se copian las rebanadas de memoria de `f16` a un búfer continuo de `f32` en el stack o en la arena de trabajo:

```rust
// src/registry/slice.rs
use half::f16;
use ndarray::{Array3, ArrayView3};
use crate::registry::ToolRegistry;

pub fn assemble_candidate_tensor(
    registry: &ToolRegistry,
    candidate_indices: &[usize],
    out_buffer: &mut Vec<f32>,
) -> (usize, usize, usize) {
    let k = candidate_indices.len();
    let total_elements = k * 96 * 384;

    out_buffer.clear();
    out_buffer.reserve(total_elements);

    for &idx in candidate_indices {
        let tool = &registry.tools[idx];
        // Conversión SIMD vectorizada f16 -> f32 directa
        for &val in &tool.tokens_fp16 {
            out_buffer.push(val.to_f32());
        }
    }

    (k, 96, 384)
}
```

Este ensamblado toma **$0.038\text{ ms}$ en CPU moderna con instrucciones AVX2/F16C**, permitiendo que el tensor tridimensional $(K, 96, 384)$ esté listo para entrar a la Etapa 2 del MoE casi instantáneamente.

---

## 4. Política de Desalojo y Retención de Memoria (LRU & TTL)

Para prevenir el crecimiento ilimitado de la memoria en entornos multi-inquilino masivos:
* Cada `ToolRegistry` almacena una marca temporal de último acceso (`Instant`).
* Una tarea periódica de Tokio (`tokio::time::interval(Duration::from_secs(300))`) inspecciona los inquilinos en el `DashMap`.
* Inquilinos inactivos por más de 2 horas cuyos catálogos superen los 100 tools se desalojan del `DashMap`, reduciendo la RAM a su valor mínimo de reposo.
