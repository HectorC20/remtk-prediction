# Plan 02: Estrategia de Datos y Destilación de Conocimiento ("Teacher-Student")

## 1. Filosofía de Entrenamiento sin Etiquetado Manual Masivo

Para entrenar los $\sim 18.5\text{M}$ de parámetros del modelo `remtk-juicio-moe` sin partir de cero ni etiquetar decenas de miles de casos a mano, implementamos un pipeline de **Destilación de Conocimiento Multi-Profesor (*Multi-Teacher Knowledge Distillation*)** apoyado en tu suite actual de pruebas (`tests/`).

Tus modelos y servicios actuales (`JuicioService`, `EstadoNliService`, `CrossEncoderService`, `MaxSimService`, `AbstencionService`) actúan fuera de línea como el **Profesor Ensamblado**, generando tensores de supervisión ricos (*soft targets* y *hard targets*).

---

## 2. Fuentes de Datos Semilla en el Repositorio

El dataset de entrenamiento se extrae y expande a partir de los siguientes activos ya presentes en `remtk-prediction`:

1. **Catálogos Reales y de Producción:**
   * `tests/fixtures/mitumbes-real-catalog.ts` y `mitumbes-production-tools.ts`
   * `tests/example-tools.ts`
2. **Flujos de Programación y Planes Complejos:**
   * `tests/coding-programmer-workflows.test.ts`
   * `tests/complex-system-plan.test.ts`
3. **Continuidad Conversacional y Estado NLI:**
   * `tests/continuous-flow-intent.test.ts`
   * `tests/sequential-conversation-loop.test.ts`
   * `tests/predict-intent-context.test.ts`
4. **Fronteras de Abstención y Estrés Adversarial:**
   * `tests/retorno-nulo-escenarios.test.ts`
   * `tests/webpage-prediction-null.test.ts`
   * `tests/casos-estres-adversariales.test.ts`
   * `tests/parrafos-contexto-estres.test.ts`
   * `tests/vague-implicit-juicio.test.ts`

---

## 3. Estructura del Registro de Destilación (`distilled_samples.jsonl`)

Cada muestra generada por el recolector contiene la tupla completa de decisión:

```json
{
  "sample_id": "mitumbes_turn_0492",
  "query_text": "no quiero hoteles caros, mejor dime dónde comer ceviche cerca de la plaza",
  "keywords": ["ceviche", "comer", "plaza"],
  "intent_context": "El usuario descartó alojamiento y cambió hacia gastronomía local.",
  "history_text": "assistant: Aquí tienes 3 hoteles en Tumbes. | user: están muy caros",
  "candidate_tools": [
    { "id": "t_rest", "text": "search_restaurants | gastronomy | Busca restaurantes y cevicherías..." },
    { "id": "t_hotel", "text": "search_hotels | lodging | Busca hoteles y hospedajes..." }
  ],
  "teacher_targets": {
    "estado_probs": [0.82, 0.05, 0.11, 0.02],
    "estado_label": 0,
    "topic_shift": 1.0,
    "tool_scores": [0.94, 0.08],
    "abstencion": 0.0,
    "expert_hint": 5
  }
}
```

---

## 4. Pipeline de Generación de Datos en 3 Pasos

### Paso 1: Extracción de Casos Exactos de `tests/` ($2,500$ muestras base)
Un script en TypeScript (`scripts/extract-test-cases.ts`) importa todos los escenarios de `tests/*.test.ts`, ejecuta el pipeline actual con las correcciones de los `expect()` de cada test (garantizando $100\%$ de fidelidad en los casos donde `mDeBERTa` o `MiniLM` fallaban) y vuelca las decisiones esperadas.

### Paso 2: Perturbación Sintética y Hard Negatives ($25,000$ muestras expandidas)
Por cada herramienta de los catálogos (`mitumbes`, `coding`, `general`):
* **Muestras Positivas Directas e Implícitas:** Consultas coloquiales en español (incluyendo modismos peruanos/latinoamericanos y términos técnicos en spanglish).
* **Hard Negatives (Crucial para los Expertos 4 y 6):** Consultas que comparten el $80\%$ de las palabras con una herramienta pero piden lo opuesto o son informativas (ej. *"cómo funciona internamente un servidor SMTP"* frente a la tool `send_email`).
* **Párrafos Extensos con Ruido:** Párrafos de $150\text{ a }400$ tokens donde la intención real está al final o en medio del texto.

### Paso 3: Pre-cálculo de Tensores `e5-small` en Disco (`tensor_cache.safetensors`)
Para que el entrenamiento en CPU tarde minutos en lugar de horas:
1. Pasamos todos los textos únicos de consultas, contextos y definiciones de herramientas por `multilingual-e5-small` **una única vez**.
2. Guardamos los tensores ocultos de la última capa `(SeqLen, 384)` cuantizados en `float16` en disco.
3. Durante las épocas de entrenamiento, la CPU lee directamente los tensores $(SeqLen, 384)$ sin volver a ejecutar el Transformer `e5-small`.