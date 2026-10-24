# Plan 03: Estrategia de Paridad Matemática, Tolerancia Flotante y Validación de Tests

## 1. Protocolo de Ejecución de los 17 Suites de Tests de TypeScript

La mayor fortaleza de la arquitectura de pruebas existente en `remtk-prediction` es que sus suites en `tests/*.test.ts` interactúan primordialmente con el servicio a través de llamadas de red HTTP o mediante la fachada de servicio en los puertos estándar (`6776`, `6777`, `6775`).

Para certificar la paridad funcional y numérica sin reescribir las pruebas en Rust, utilizamos un **protocolo de caja negra de validación cruzada**: levantamos el binario compilado en Rust y redirigimos el arnés de pruebas de Node.js contra él.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ARQUETIPO DE VALIDACIÓN CRUZADA: TESTS TYPESCRIPT -> SERVIDOR RUST                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   ┌────────────────────────┐                   HTTP JSON                    ┌─────────┐│
│   │ 17 Suites de Tests TS  │  ───────────────────────────────────────────►  │ Rust    ││
│   │ (tests/*.test.ts)      │  ◄───────────────────────────────────────────  │ Server  ││
│   │ pnpm test              │            Puertos: 6776 / 6777 / 6775         │ (Axum)  ││
│   └────────────────────────┘                                                └─────────┘│
│               │                                                                  │     │
│               ▼                                                                  ▼     │
│     Validación de Aserciones:                                           Validación     │
│     • Herramientas Top-1 idénticas                                      Interna:       │
│     • Scores con tolerancia <= 1e-4                                     • Zero-Copy    │
│     • Estados NLI categóricos idénticos                                 • RAM < 90 MB  │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1. Los 17 Suites Críticos a Certificar

| # | Archivo de Prueba | Dominio Evaluado | Criterio de Aceptación Crítico |
| :-: | :--- | :--- | :--- |
| **1** | `tests/rerank.test.ts` | Re-ranking y ordenamiento semántico | Orden topológico de herramientas preservado |
| **2** | `tests/e2e.test.ts` | Flujo E2E de predicción de herramientas | Contrato de respuesta `/predict` exacto |
| **3** | `tests/graph.test.ts` | Grafo de dependencias y co-activación | Preservación de cadenas DAG de tools |
| **4** | `tests/learn-barrier.test.ts` | Barreras de aprendizaje y abstención | Respeto de umbrales mínimos de corte |
| **5** | `tests/mitumbes-barrier.test.ts` | Catálogo de negocio específico (mitumbes) | Discriminación fina de herramientas de catálogo |
| **6** | `tests/calibration-context.test.ts` | Calibración de scores con contexto previo | Modulación correcta por `intentContext` |
| **7** | `tests/spatial-predict.test.ts` | `/spatial/predict` y coordenadas de cursor | Intersección correcta de cursor $(x, y)$ |
| **8** | `tests/intent-recognition.test.ts` | Clasificación de intenciones y verbos | Activación precisa de expertos de acción |
| **9** | `tests/complex-system-plan.test.ts` | Flujos de sistemas y orquestación multi-herramienta | Detección de planes compuestos |
| **10** | `tests/continuous-flow-intent.test.ts` | Conversación continua y continuidad de turnos | Estado NLI = `CONTINUACION` |
| **11** | `tests/web-file-generation.test.ts` | Manejo de texto con código y generación web | Robustez ante sintaxis especial y markdown |
| **12** | `tests/catalog-inquiry.test.ts` | Consultas sobre catálogos densos | Rendimiento con $> 100$ herramientas |
| **13** | `tests/predict-dag-cold-catalog.test.ts` | Catálogos fríos y arranque sin herramientas | Retorno vacío `tools: []` sin error 500 |
| **14** | `tests/predict-intent-context.test.ts` | Segmentación de `intentContext` | Inyección de pre-pensamiento del planificador |
| **15** | `tests/e2e-nest-log-reproduction.test.ts` | Reproducción de logs y trazas de auditoría | Formato estándar de logs y métricas |
| **16** | `tests/skill-memory-architecture.test.ts` | `/memory/predict` y recuperación de skills | Coincidencia de skills aprendidos |
| **17** | `tests/skill-learning-sequential.test.ts` | Aprendizaje secuencial en memoria de sesión | Persistencia y actualización en memoria |

---

## 2. Metodología de Comparación Numérica y Tolerancia Flotante

Debido a que Node.js utiliza `onnxruntime-node` compilado con ciertas banderas del compilador y Rust utiliza la biblioteca dinámica oficial `libonnxruntime` a través de `ort`, pueden presentarse variaciones infinitesimales de precisión causadas por la reordenación asociativa de instrucciones vectoriales SIMD (FMA / AVX2).

Fijamos los siguientes **márgenes matemáticos de tolerancia:**

### 2.1. Embeddings Vectoriales (Puerto 6777 - `/embed`)
Para cualquier vector de salida $\mathbf{v}_{\text{rust}}, \mathbf{v}_{\text{node}} \in \mathbb{R}^{384}$:
* **Diferencia Absoluta Máxima:**
  $$\max_i |v_{\text{rust}, i} - v_{\text{node}, i}| \le 1.0 \times 10^{-4}$$
* **Similitud Coseno entre Ambos Vectores:**
  $$\cos(\mathbf{v}_{\text{rust}}, \mathbf{v}_{\text{node}}) = \frac{\mathbf{v}_{\text{rust}} \cdot \mathbf{v}_{\text{node}}}{\|\mathbf{v}_{\text{rust}}\|_2 \|\mathbf{v}_{\text{node}}\|_2} \ge 0.99999$$

### 2.2. Scores del Cabezal MoE (Puerto 6776 - `/predict`)
* **Scores Calibrados de Herramientas (`rankedScores`):**
  $$|s_{\text{rust}, k} - s_{\text{node}, k}| \le 1.0 \times 10^{-3}$$
* **Probabilidad de Abstención (`abstencionProb`):**
  $$|p_{\text{abs, rust}} - p_{\text{abs, node}}| \le 2.0 \times 10^{-3}$$
* **Decisiones Categóricas Booleanas:**
  $$\text{debeAbstenerse}_{\text{rust}} \equiv \text{debeAbstenerse}_{\text{node}} \quad (100\% \text{ concordancia})$$
  $$\text{estadoDominante}_{\text{rust}} \equiv \text{estadoDominante}_{\text{node}} \quad (100\% \text{ concordancia})$$

---

## 3. Automatización del Arquetipo de Pruebas de Paridad

Se crea un script de automatización en `scripts/verify-rust-parity.sh`:

```bash
#!/usr/bin/env bash
set -e

echo "=== [1/4] Compilando binario de Rust en modo Release ==="
cargo build --release --manifest-path rust-server/Cargo.toml

echo "=== [2/4] Arrancando Servidor Rust en segundo plano ==="
./rust-server/target/release/remtk-prediction-server &
RUST_PID=$!

# Esperar a que los 3 puertos respondan
while ! nc -z localhost 6776; do sleep 0.1; done
while ! nc -z localhost 6777; do sleep 0.1; done
while ! nc -z localhost 6775; do sleep 0.1; done
echo "-> Servidor Rust listo en puertos 6776, 6777 y 6775 (PID: $RUST_PID)"

echo "=== [3/4] Ejecutando las 17 Suites de Tests en TypeScript ==="
export PORT_PREDICT=6776
export PORT_EMBED=6777
export PORT_QDRANT=6775
export RUN_AGAINST_REMOTE_SERVER=true

pnpm test

TEST_STATUS=$?

echo "=== [4/4] Limpieza y Apagado ==="
kill -SIGTERM $RUST_PID
wait $RUST_PID 2>/dev/null || true

if [ $TEST_STATUS -eq 0 ]; then
  echo "✅ ¡ÉXITO TOTAL! Las 17 suites pasaron en verde contra el servidor Rust."
  exit 0
else
  echo "❌ Fallo en las pruebas de paridad. Revisa los logs anteriores."
  exit 1
fi
```

---

## 4. Validación de Casos Extremos y Escenarios Límite

1. **Textos Extensos (> 512 tokens):**
   Verificar que la ventana deslizante (*sliding window*) en `e5-small` maneje bloques de código o contratos legales extensos sin causar *panic*, desbordamiento de pila (*stack overflow*) ni errores de dimensión en ONNX Runtime.
2. **Catálogos Vacíos o con 1 sola Herramienta:**
   Probar peticiones con $K = 0$ y $K = 1$. Validar que el Set-Transformer en la Etapa 3 del MoE compute correctamente la matriz de auto-atención $\mathbf{H}$ sin divisiones por cero en el cálculo de entropía normalizada $\frac{1}{\log K}$ (aplicando el límite matemático $\lim_{K \to 1} \mathcal{H} = 0$).
3. **Caracteres Multilingües y Emojis:**
   Asegurar que el tokenizer de Hugging Face en Rust procese secuencias UTF-8 complejas (japonés, árabe, caracteres cirílicos y emojis) con exactamente los mismos IDs de tokens que `@huggingface/tokenizers` en Node.js.
