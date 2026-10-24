# Arquitectura 01: Topología Matemática del Modelo `remtk-juicio-moe`

## 1. Visión General del Grafo Unificado

El modelo `remtk-juicio-moe` es una red neuronal diferenciable de **3 etapas explícitas** que opera sobre secuencias de embeddings de tokens de dimensión $d = 384$ (provenientes de `multilingual-e5-small`).

En lugar de concatenar todo el texto en una única secuencia larga (lo que tendría complejidad cuadrática $\mathcal{O}((L_q + L_c + K \cdot L_t)^2)$), la arquitectura mantiene **tres ejes tensoriales independientes** y aplica atención cruzada dirigida con complejidad lineal respecto al número de herramientas $\mathcal{O}(K \cdot L_q \cdot L_t)$.

---

## 2. Etapa 1: Compuerta Episódica y Estado Conversacional NLI ($\sim 3.5\text{M}$ Parámetros)

Recibe dos tensores de entrada:
* **Consulta + Keywords ($\mathbf{Q}$):** Tensor de forma $(1, L_q, 384)$.
* **Historial + IntentContext ($\mathbf{C}$):** Tensor de forma $(1, L_c, 384)$.

### 2.1. Codificación Secuencial Local (BiGRU Ligera + RoPE)
Para capturar el orden sintáctico fino, negaciones (*"no envíes correo, solo guarda el borrador"*) y referencias pronominales (*"elimínalo"*), ambas secuencias pasan por una capa **Gated Convolution 1D (`kernel_size=3`) + Proyección Residual**:

$$\tilde{\mathbf{Q}} = \text{LayerNorm}\left(\mathbf{Q} + \text{GLU}(\text{Conv1D}(\mathbf{Q}))\right) \in \mathbb{R}^{1 \times L_q \times 384}$$

$$\tilde{\mathbf{C}} = \text{LayerNorm}\left(\mathbf{C} + \text{GLU}(\text{Conv1D}(\mathbf{C}))\right) \in \mathbb{R}^{1 \times L_c \times 384}$$

### 2.2. Atención Cruzada Query-a-Contexto y Clasificación NLI
Calculamos la matriz de afinidad entre los tokens de la consulta actual y los tokens del contexto previo:

$$\mathbf{A}_{qc} = \text{Softmax}\left(\frac{(\tilde{\mathbf{Q}}\mathbf{W}_q)(\tilde{\mathbf{C}}\mathbf{W}_k)^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times L_q \times L_c}$$

$$\mathbf{M}_{qc} = \mathbf{A}_{qc}(\tilde{\mathbf{C}}\mathbf{W}_v) \in \mathbb{R}^{1 \times L_q \times 384}$$

A partir de los vectores resumidos por *Attention Pooling* $\mathbf{q}_{\text{pool}}, \mathbf{c}_{\text{pool}}, \mathbf{m}_{\text{pool}} \in \mathbb{R}^{384}$, construimos el vector de interacción de estado:

$$\mathbf{u}_{\text{nli}} = \left[\mathbf{q}_{\text{pool}} \;\|\; \mathbf{c}_{\text{pool}} \;\|\; |\mathbf{q}_{\text{pool}} - \mathbf{c}_{\text{pool}}| \;\|\; (\mathbf{q}_{\text{pool}} \odot \mathbf{m}_{\text{pool}})\right] \in \mathbb{R}^{1536}$$

Un bloque denso proyecta $\mathbf{u}_{\text{nli}}$ hacia los **4 estados conversacionales**:

$$\mathbf{p}_{\text{estado}} = \text{Softmax}(\mathbf{W}_{\text{nli}}\,\text{GELU}(\mathbf{W}_1 \mathbf{u}_{\text{nli}})) \in \mathbb{R}^4$$

Correspondientes a:
* $p_0$: `TRANSICION` (Cambio de objetivo o tema nuevo)
* $p_1$: `CONTINUACION` (Aporte de parámetros o seguimiento del mismo flujo)
* $p_2$: `CORRECCION` (Enmienda o negación de lo anterior)
* $p_3$: `ABANDONO` (Cierre, saludo o texto sin intención operativa)

### 2.3. Compuerta Diferenciable de Fusión Contextual ($\alpha$)
La probabilidad de continuidad y corrección gobierna cuánto del contexto histórico e `intentContext` se inyecta en los tokens de la consulta antes de evaluar las herramientas:

$$\alpha = \sigma\left(\mathbf{w}_g^T \mathbf{p}_{\text{estado}} + b_g\right) \in [0, 1]$$

$$\mathbf{Q}^* = \text{LayerNorm}\left(\tilde{\mathbf{Q}} + \alpha \cdot \mathbf{M}_{qc}\right) \in \mathbb{R}^{1 \times L_q \times 384}$$

Si el modelo detecta `TRANSICION` o `ABANDONO`, $\alpha \to 0$ automáticamente, limpiando el ruido del historial previo sin necesidad de `if/else` externos.

---

## 3. Etapa 2: Enrutador y 6 Expertos de Interacción ($\sim 13.5\text{M}$ Parámetros Totales)

Recibe la consulta contextualizada $\mathbf{Q}^* \in \mathbb{R}^{1 \times L_q \times 384}$ y el tensor 3D de las $K$ herramientas candidatas $\mathbf{T} \in \mathbb{R}^{K \times L_t \times 384}$.

### 3.1. Enrutador Top-2 (*Noisy Top-K Gating*)
El Router analiza $\mathbf{q}^*$ y $\mathbf{p}_{\text{estado}}$ para elegir los **2 expertos más aptos** entre $E = 6$:

$$\mathbf{g} = \text{TopKSoftmax}\left(\mathbf{W}_{\text{router}} [\mathbf{q}^* \;\|\; \mathbf{p}_{\text{estado}}], \; k=2\right) \in \mathbb{R}^6$$

### 3.2. Especialización de los 6 Expertos
Cada experto $e \in \{1 \dots 6\}$ tiene sus propias matrices de proyección $\mathbf{U}_e, \mathbf{V}_e \in \mathbb{R}^{384 \times 256}$ y su propio bloque de evaluación bilineal:
* **Experto 1 (Acción e Imperativos):** Optimizado para verbos de ejecución, creación, modificación y llamadas directas a APIs.
* **Experto 2 (Consulta de Catálogo y Búsqueda):** Optimizado para recuperación de información, filtros de negocio y directorios (`mitumbes`).
* **Experto 3 (Resolución Implícita y Vaga):** Optimizado para conectar descripciones de síntomas o problemas con herramientas cuya descripción es técnica (`vague-implicit-juicio`).
* **Experto 4 (Verificación de Especificidad y Schema):** Aprende el rol de `EspecificidadService` de forma densa, penalizando herramientas genéricas cuando hay coincidencias de dominio exacto.
* **Experto 5 (Orquestación Multi-Herramienta y Código):** Especializado en flujos donde múltiples herramientas deben co-activarse con scores altos (`coding-programmer-workflows` y `complex-system-plan`).
* **Experto 6 (Filtro Adversarial y Chitchat):** Especializado en detectar negaciones, preguntas retóricas, contenido de páginas web pegado y falsos amigos semánticos.

### 3.3. Operador MaxSim Aprendido + Cruce Bilineal dentro de cada Experto Activo
Para cada experto activo $e \in \text{Top2}$, proyectamos los tokens de la consulta y de las $K$ herramientas:

$$\mathbf{Q}_e = \mathbf{Q}^* \mathbf{U}_e \in \mathbb{R}^{1 \times L_q \times 256}, \qquad \mathbf{T}_e = \mathbf{T} \mathbf{V}_e \in \mathbb{R}^{K \times L_t \times 256}$$

Calculamos el tensor de similitud token-a-token para las $K$ herramientas simultáneamente mediante contracción de Einstein (`einsum("bld,ktd->bkl t")`):

$$\mathbf{S}_e(k, i, j) = \frac{\langle \mathbf{Q}_e(1, i), \, \mathbf{T}_e(k, j) \rangle}{\|\mathbf{Q}_e(1, i)\|_2 \, \|\mathbf{T}_e(k, j)\|_2}$$

De esta matriz $\mathbf{S}_e \in \mathbb{R}^{K \times L_q \times L_t}$ extraemos un vector de características ricas por herramienta $k$:
1. **MaxSim Forward (Query $\to$ Tool):** $\mu_{\text{fwd}, k} = \frac{1}{L_q} \sum_{i=1}^{L_q} \max_{j} \mathbf{S}_e(k, i, j)$
2. **MaxSim Backward (Tool $\to$ Query, Especificidad):** $\mu_{\text{bwd}, k} = \frac{1}{L_t} \sum_{j=1}^{L_t} \max_{i} \mathbf{S}_e(k, i, j)$
3. **Atención Cruzada Ponderada:** $\mathbf{z}_{e, k} = \text{MLP}_e\left([\mathbf{q}_e^{\text{pool}} \;\|\; \mathbf{t}_{e, k}^{\text{pool}} \;\|\; (\mathbf{q}_e^{\text{pool}} \odot \mathbf{t}_{e, k}^{\text{pool}}) \;\|\; \mu_{\text{fwd}, k} \;\|\; \mu_{\text{bwd}, k}]\right) \in \mathbb{R}^{128}$

La representación combinada de la herramienta $k$ tras el MoE es la suma ponderada de los 2 expertos activos:

$$\mathbf{h}_k = \sum_{e \in \text{Top2}} g_e \cdot \mathbf{z}_{e, k} \in \mathbb{R}^{128}$$

---

## 4. Etapa 3: Set-Attention y Cabeza de Abstención ($\sim 1.5\text{M}$ Parámetros)

A diferencia de un Cross-Encoder tradicional que puntúa cada herramienta sin saber qué otras herramientas hay en la lista, la **Etapa 3** aplica una capa de **Self-Attention entre las $K$ herramientas candidatas** sobre la matriz $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_K] \in \mathbb{R}^{K \times 128}$:

$$\tilde{\mathbf{H}} = \text{MultiHeadSelfAttention}(\mathbf{H}, \mathbf{H}, \mathbf{H}) \in \mathbb{R}^{K \times 128}$$

1. **Scores Calibrados de Salida (`ranked_scores`):**
   $$s_k = \sigma(\mathbf{w}_{\text{score}}^T \tilde{\mathbf{h}}_k + b_{\text{score}}) \in (0, 1)$$
2. **Cálculo Diferenciable de Incertidumbre y Abstención (`abstencion_prob`):**
   El grafo calcula internamente:
   * $s_{(1)} = \max_k s_k$ (Score Top-1)
   * $\Delta_{1,2} = s_{(1)} - s_{(2)}$ (Margen entre Top-1 y Top-2)
   * $\mathcal{H}(\mathbf{s}) = -\frac{1}{\log K} \sum_{k=1}^{K} \hat{p}_k \log(\hat{p}_k)$ (Entropía de Shannon normalizada)
   
   Y produce la probabilidad final de abstención combinando la distribución de scores y el estado NLI:
   $$p_{\text{abstencion}} = \sigma\left(\text{MLP}_{\text{abs}}\left([s_{(1)}, \; \Delta_{1,2}, \; \mathcal{H}(\mathbf{s}), \; \mathbf{p}_{\text{estado}}, \; \max_k \tilde{\mathbf{h}}_k]\right)\right)$$