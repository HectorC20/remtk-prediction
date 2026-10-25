# Plan 04: Migración del Modelo 300M en `src/juicio/`, Validación y Puesta en Producción

## 1. Mapa de Transición en `src/juicio/`

El archivo cuantizado `models/remtk-juicio-moe-int8/model_quantized.onnx` (**~295 MB**) reemplaza de forma transparente tanto a los modelos antiguos (`mDeBERTa`, `ms-marco-MiniLM`) como al prototipo inicial de 18.5M.

| Archivo en `src/juicio/` | Rol con `remtk-juicio-moe-300m` |
| :--- | :--- |
| `juicio.service.ts` | Prepara los `Float32Array` de `query_tokens` y `context_tokens` (asegurando `0.0` en posiciones de padding) y recorta el sub-tensor de `tools_tokens` desde la caché en RAM. |
| `services/moe-decision.service.ts` | Carga el modelo de **300M INT8** con `enableCpuMemArena: true`, ejecuta `warmup()` al iniciar el servidor y resuelve las 3 etapas en una sola llamada `session.run()`. |

---

## 2. Garantía de Cero Padding Corrupto en la Caché de Herramientas (`POST /tools`)

Para que las máscaras automáticas `t_mask = (norm(tools_tokens, dim=-1) > 1e-5)` del modelo 300M funcionen con exactitud matemática:
1. Al indexar una herramienta en `POST /tools` con longitud real $L_{\text{real}} \le 96$, los primeros $L_{\text{real}} \times 384$ floats contienen los embeddings L2-normalizados de `e5-small`.
2. Los restantes $(96 - L_{\text{real}}) \times 384$ floats del `Float32Array` quedan inicializados en **`0.0` exacto** (comportamiento nativo de `new Float32Array(...)` en V8/Node.js).
3. **Nunca** se debe rellenar el final del tensor repitiendo el último token ni el token `[PAD]` de `e5-small`, ya que eso impediría que el modelo detecte dónde termina realmente la descripción de cada herramienta.

---

## 3. Criterios de Aceptación para el Modelo 300M (`pnpm test:all`)

1. **Precisión en Casos de Negación y Corrección (`continuous-flow-intent.test.ts`):** $100\%$ de acierto cuando el usuario cancela o cambia de opinión a mitad de frase (`estadoDominante === "CORRECCION"` o `"TRANSICION"`).
2. **Frontera de Abstención (`retorno-nulo-escenarios.test.ts`, `webpage-prediction-null.test.ts`):** `debeAbstenerse === true` (`tools: []`) en el $100\%$ de textos informativos pegados, saludos y preguntas fuera de catálogo.
3. **Presupuesto de Latencia y RAM:**
   * RAM total del proceso Node.js en reposo con 1 tenant de 200 tools: **$< 600\text{ MB}$**.
   * Tiempo de inferencia de `MoEDecisionService.evaluate()` para $K = 80$ tools en CPU DDR5: **$< 22\text{ ms}$**.