# 🛡️ AGENT.md — Directiva Arquitectónica y Principios de Diseño

> **ADVERTENCIA OBLIGATORIA PARA DESARROLLADORES Y AGENTES DE IA:**  
> Este documento establece la doctrina irrenunciable de ingeniería para `remtk-prediction`, los agentes unificados y la memoria contextual (`remtk-memory`). Cualquier modificación futura **debe cumplir estrictamente** con estos principios.

---

## 1. 🚫 Prohibición Estricta de Parches Léxicos

Está **TERMINANTEMENTE PROHIBIDO** implementar o reintroducir:
1. **Palabras clave hardcodeadas (Keywords)**: No añadir listas de palabras fijas (`["item", "crear", "listar", "hotel", "zona"]`) para forzar scores o desempates.
2. **Diccionarios y Arreglos Léxicos**: No crear objetos o mapas con sinónimos manuales, términos prohibidos o listas estáticas de sustantivos/verbos.
3. **Expresiones Regulares de Vocabulario (Regex)**: No usar expresiones regulares con palabras específicas como `/(?:listar|list|obtener|get|crear)/i` para inferir relaciones o clasificar intenciones.
4. **Condicionales Rígidos (IF/ELSE) de Coincidencia de Cadenas**: No condicionar el flujo de predicción mediante comprobaciones de inclusión de texto (`text.includes("crea el item")`).

---

## 2. 🧠 Principio de Juicio, Criterio y Razonamiento Estructural

La arquitectura de `remtk-prediction` debe resolver la selección de herramientas, la memoria y el flujo de tareas mediante **criterio matemático, neuronal y topológico**:

### A. Espacio Latente Denso (Bi-Encoder INT8)
- El filtrado y la recuperación de herramientas se basan en la **similitud de coseno en variedades continuas** ($\mathbf{u} \cdot \mathbf{v}_i$) generadas por embeddings densos (e.g. `multilingual-e5-small`).
- La recuperación debe mantener un **Recall@K adecuado** (Top 10-15% del catálogo) para que las herramientas candidatas viables pasen al árbitro profundo sin ser ahogadas por la longitud de la consulta.

### B. Juicio de Atención Cruzada (Cross-Encoder INT8)
- La discriminación de polaridad, negación formal (*"no envíes el correo, guárdalo"*), núcleo de acción (*"crea el item..."*) y descarte de falsos positivos se delega al **Cross-Encoder** (`ms-marco-minilm-l6` / NLI).
- Las capas de auto-atención bidireccional token a token analizan sintácticamente qué parte del texto es la acción ejecutiva y qué partes son meros parámetros o descripciones complementarias.

### C. Pre-Plan de Razonamiento en Grafo DAG (Herramientas Complementarias)
- **El Objetivo no camina solo**: Una herramienta de mutación o escritura (ej. `item_crear`, `item_actualizar`) casi siempre depende de parámetros que el usuario no conoce (UUIDs de categorías, zonas, identificadores de entidad).
- El sistema no debe entregar la herramienta objetivo de forma aislada. Debe construir un **Pre-plan de Razonamiento** activando en el grafo las **herramientas complementarias antecedentes** (ej. `categoria_listar`, `zona_listar`).
- El algoritmo topológico (Kahn) sobre las aristas `PREREQUISITE` garantiza que las herramientas de descubrimiento se ejecuten **antes** que la herramienta de mutación:
  $$\text{Pre-requisito 1} \rightarrow \text{Pre-requisito 2} \rightarrow \text{Herramienta Objetivo}$$

### D. Juicio Limpio en Memoria Contextual (`remtk-memory`)
- La memoria contextual almacena hechos reales, instrucciones previas y confirmaciones del asistente.
- Para filtrar ruido parásito (ej. negaciones de capacidad técnica donde el agente dice que no puede operar), el juicio debe evaluar **la polaridad técnica real** mediante umbrales calibrados o atención cruzada, **nunca** descartar recuerdos por contener palabras operativas como *"crea el item"*.

---

## 3. 🛠️ Protocolo ante Fallos de Predicción o Casos de Borde

Si en el futuro una prueba o caso de uso no selecciona la herramienta adecuada:

1. **NO añadas un `if (query.includes(...))`**.
2. **NO crees un array de keywords**.
3. **Inspecciona las representaciones y trazas estructuradas (`src/shared/logger/`)**:
   - Revisa en consola o en el archivo rotativo diario `logs/prediction-YYYY-MM-DD.log` las trazas por etapa:
     - `[trace:recall]`: tamaño del catálogo, términos tokenizados y candidatos léxicos iniciales (`ToolRegistry::retrieve_candidates`).
     - `[trace:moe]`: estado NLI dominante (`TRANSICION`, `CONTINUACION`, `CORRECCION`, `ABANDONO`), probabilidades por clase, probabilidad de abstención (`abstencion_prob` vs `threshold`), `topic_shift` y pesos de los 6 expertos.
     - `[trace:predict]`: herramientas ganadoras (`selected`), `top_candidates` con sus scores exactos y `filtered_out` (herramientas descartadas por el piso `adaptive_min_score` o el corte `max_output_tools` / `gap_threshold`).
     - `[profile:predict]` y `[http:predict]`: desglose de latencias por sub-etapa (`recall`, `q_emb`, `query_tokens`, `ctx_tokens`, `assemble`, `moe`).
   - ¿Está el esquema de entrada (`inputSchema`) definiendo claramente las propiedades requeridas y está su vocabulario funcional reflejado en `description` e `intentSummary`?
   - ¿Tienen las herramientas descripciones semánticas ricas que expliquen su rol funcional sin depender de términos únicamente escondidos en propiedades internas del schema?
   - ¿Están declaradas las aristas `prerequisites` o `conflicts` en el grafo $G=(V, E)$ y las clausuras de familia correspondientes?
   - ¿La profundidad de candidatos ($K$) de Fase 1 (`RECALL_LIMIT`) es suficiente para el tamaño del catálogo?
   - ¿El cabezal MoE (`remtk-juicio-moe-int8`) de Fase 2 está calibrado e integrado en el score final de ranking?

---

## 4. 📁 Estructura Modular Canónica (Rust Nativo)

```
remtk-prediction/
├── src/
│   ├── main.rs                          # Bootstrap Tokio, init_logger() y 3 listeners Axum (6776/6777/6775)
│   ├── lib.rs                           # Raíz de librería del monolito modular
│   ├── api/                             # Capa de transporte HTTP (Axum) y trazas [http:*]
│   │   ├── mod.rs
│   │   ├── state.rs                     # AppState compartido (Arc)
│   │   ├── predict_server.rs            # Puerto 6776 (/predict, /spatial/predict, /predict/route, /history/regime, /tools...)
│   │   ├── embed_server.rs              # Puerto 6777 (/embed, /models, /health)
│   │   └── qdrant_server.rs             # Puerto 6775 (/tools/upsert, /tools/search, /status...)
│   ├── core/                            # Dominio central e inferencia matemática pura (sin dependencias HTTP)
│   │   ├── embedding/service.rs         # EmbeddingService (e5-small 384d, L2 norm, [profile:embed])
│   │   ├── inference/
│   │   │   ├── engine.rs                # InferenceEngine (sesiones ort v2 con mmap)
│   │   │   ├── moe.rs                   # Cabezal MoE 3 etapas y trazas [trace:moe]
│   │   │   └── tokenizer.rs             # E5Tokenizer (prefijos query: / passage:)
│   │   └── prediction/service.rs        # PredictionService, umbral adaptativo, Fast-Path y trazas [trace:predict]
│   ├── support/                         # Capacidades de soporte e infraestructura en memoria
│   │   ├── registry/tool_cache.rs       # TenantCache (DashMap), tensores FP16 y recall [trace:recall]
│   │   ├── qdrant/qdrant_service.rs     # QdrantService ([profile:qdrant_upsert], [trace:qdrant_search])
│   │   └── spatial/spatial_service.rs   # SpatialService ([trace:spatial])
│   └── shared/                          # Capa transversal (Config, Logger, DTOs, Errores, Tensores)
│       ├── config/mod.rs                # AppConfig desde variables de entorno
│       ├── logger/mod.rs                # LoggerConfig, DualMakeWriter (stdout + logs/prediction-YYYY-MM-DD.log)
│       ├── dto/mod.rs                   # Contratos Serde 1:1
│       ├── error/mod.rs                 # AppError (thiserror -> Axum IntoResponse)
│       └── tensor/mod.rs                # Operaciones SIMD (cosine, L2, FP16 <-> FP32)
├── docs/                                # Documentación técnica y diagnósticos forenses
├── logs/                                # Logs rotativos diarios (prediction-YYYY-MM-DD.log)
├── migration/                           # Especificación de arquitectura Rust y guías operativas
└── models/                              # Modelos ONNX (multilingual-e5-small-onnx, remtk-juicio-moe-int8)
```

---

*Cualquier código que viole esta directiva introduciendo listas léxicas o palabras clave forzadas se considerará deuda técnica y degradación de la arquitectura cognitiva.*
