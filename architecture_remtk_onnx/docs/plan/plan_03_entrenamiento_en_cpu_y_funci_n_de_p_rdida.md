# Plan 03: Entrenamiento en CPU Local, Función de Pérdida Multi-Tarea y Calibración

## 1. Estrategia de Entrenamiento Viable en CPU

Entrenar un modelo desde cero en CPU es inviable si se retropropagan gradientes a través de 12 capas de auto-atención y una matriz de 250,000 tokens. Nuestro plan elimina ese cuello de botella mediante **Desacoplamiento del Extractor**:

* **Extractor Congelado (`multilingual-e5-small`):** No se entrena. Sus salidas a nivel de token $\mathbf{H} \in \mathbb{R}^{L \times 384}$ se pre-calculan una sola vez.
* **Red Entrenable (`remtk-juicio-moe`):** Solo se entrenan las **3 etapas de decisión** ($\sim 18.5\text{M}$ de parámetros en total, con solo $\sim 6.5\text{M}$ activos por muestra).
* **Complejidad Computacional por Época:** Con $25,000$ muestras y *batch size* de $32$, una CPU de $8$ hilos con soporte AVX2 completa **1 época en $\sim 3.5\text{ minutos}$** ($6$ épocas $\approx 21\text{ minutos}$).

---

## 2. Función de Pérdida Conjunta (*Multi-Task Loss Function*)

Para que las 3 etapas dentro del mismo archivo `.onnx` colaboren entre sí, optimizamos una única función de pérdida compuesta:

$$\mathcal{L}_{\text{Total}} = \lambda_1 \mathcal{L}_{\text{ListNet}} + \lambda_2 \mathcal{L}_{\text{Margin}} + \lambda_3 \mathcal{L}_{\text{Estado}} + \lambda_4 \mathcal{L}_{\text{Abstención}} + \lambda_5 \mathcal{L}_{\text{BalanceMoE}}$$

Donde los pesos de equilibrio son $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0.4$, $\lambda_4 = 0.8$, $\lambda_5 = 0.02$.

### 2.1. Pérdida de Ranking Listwise ($\mathcal{L}_{\text{ListNet}}$) y de Margen ($\mathcal{L}_{\text{Margin}}$)
En lugar de evaluar cada herramienta de forma aislada, obligamos al modelo a respetar el `ONNX_ADAPTIVE_GAP_THRESHOLD` ($0.03$) y `ONNX_ADAPTIVE_MIN_SCORE` ($0.80$):

$$\mathcal{L}_{\text{Margin}} = \frac{1}{|P||N|} \sum_{i \in P} \sum_{j \in N} \max\left(0, \gamma - (s_i - s_j)\right)$$

Donde $P$ es el conjunto de herramientas correctas para el turno, $N$ son las herramientas incorrectas del mismo catálogo, y $\gamma = 0.25$ fuerza una separación clara entre la última herramienta válida y la primera inválida.

### 2.2. Pérdida de Estado Conversacional NLI ($\mathcal{L}_{\text{Estado}}$)
Combina *Cross-Entropy* con *KL-Divergence* sobre los 4 estados (`TRANSICION`, `CONTINUACION`, `CORRECCION`, `ABANDONO`) a temperatura $\tau = 2.0$:

$$\mathcal{L}_{\text{Estado}} = (1 - \alpha)\,\text{CE}(\mathbf{y}_{\text{est}}, \hat{\mathbf{y}}_{\text{est}}) + \alpha \tau^2\,\text{KL}\left(\sigma(\mathbf{z}_{\text{teacher}}/\tau) \parallel \sigma(\mathbf{z}_{\text{student}}/\tau)\right)$$

### 2.3. Pérdida de Abstención Selectiva ($\mathcal{L}_{\text{Abstención}}$)
Usa *Focal Binary Cross-Entropy* para penalizar severamente los falsos positivos en consultas nulas (como `webpage-prediction-null.test.ts` y `retorno-nulo-escenarios.test.ts`):

$$\mathcal{L}_{\text{Abstención}} = -(1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0$$

### 2.4. Pérdida de Balanceo de Carga de Expertos ($\mathcal{L}_{\text{BalanceMoE}}$)
Evita el colapso del Router (donde solo 1 o 2 expertos reciben todas las consultas):

$$\mathcal{L}_{\text{BalanceMoE}} = E \sum_{e=1}^{E} f_e \cdot P_e$$

Donde $E = 6$ es el número de expertos, $f_e$ es la fracción de consultas despachadas al experto $e$ y $P_e$ es la probabilidad media asignada por la compuerta a dicho experto.

---

## 3. Hiperparámetros de Entrenamiento en CPU

* **Optimizador:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.98$, `weight_decay = 0.01`).
* **Learning Rate:** $4 \times 10^{-4}$ con *Cosine Annealing Schedule* y $10\%$ de *warmup*.
* **Batch Size:** $16$ muestras por paso con `gradient_accumulation_steps = 2` (Batch efectivo de $32$).
* **Dropout:** $0.15$ en las capas de proyección de los expertos para evitar sobreajuste a palabras específicas del catálogo de entrenamiento.