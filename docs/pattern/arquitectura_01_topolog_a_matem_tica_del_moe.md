# Arquitectura 01: Topología Matemática del Modelo `remtk-juicio-moe-300m`

## 1. Visión General del Grafo Unificado (Escala ~300M Parámetros)

El modelo `remtk-juicio-moe` (versión **300M**) es una red neuronal diferenciable de **3 etapas explícitas** que opera sobre secuencias de embeddings de tokens de dimensión $d = 384$ (provenientes de `multilingual-e5-small` congelado) y sus respectivas máscaras de atención reales.

Para superar los fallos de decisión en escenarios de negación distante, ambigüedad semántica, catálogos densos ($K \le 200$) y párrafos extensos, esta arquitectura introduce:
1. **Convoluciones Compuerta Dilatadas Multiescala (`dilation = 1, 2, 4`)** con campo receptivo efectivo de 15 tokens.
2. **Multi-Head Attentive Pooling** (sustituyendo el promedio simple `mean(dim=1)` que destruía modificadores lógicos como *"no"*, *"sin"*, *"en vez de"*).
3. **6 Macro-Expertos Profundos SwiGLU** ($\sim 44\text{M}$ de parámetros por experto, $\sim 264\text{M}$ en la Etapa 2) con **MaxSim Bidireccional Enmascarado**.
4. **Set-Attention Profundo de 3 Capas Transformer ($d_{\text{out}} = 512$)** con conciencia explícita de cardinalidad $\log(K)$ en la Cabeza de Abstención.

---

## 2. Etapa 1: Compuerta Episódica Multiescala y Estado NLI ($\sim 22.4\text{M}$ Parámetros)

Recibe dos tensores de secuencias y sus máscaras booleanas de validez (generadas automáticamente por norma L2 $> 10^{-5}$ dentro del grafo ONNX para mantener intacta la firma de entrada):
* **Consulta + Keywords ($\mathbf{Q}$):** Tensor $(1, L_q, 384)$ con máscara $\mathbf{m}_q \in \{0, 1\}^{1 \times L_q}$.
* **Historial + IntentContext ($\mathbf{C}$):** Tensor $(1, L_c, 384)$ con máscara $\mathbf{m}_c \in \{0, 1\}^{1 \times L_c}$.

### 2.1. Pila Secuencial Multiescala (3 Capas Dilated GatedConv1D: $d = 384 \to 768 \to 384$)
Para capturar negaciones a distancia (*"no quiero que borres la base de datos de producción, solo respalda los logs"*), $\mathbf{Q}$ y $\mathbf{C}$ atraviesan 3 bloques residuales con dilataciones crecientes $r \in \{1, 2, 4\}$:

$$\mathbf{H}^{(l)} = \text{LayerNorm}\left(\mathbf{H}^{(l-1)} + \text{SwiGLU}\left(\text{Conv1D}_{k=3, \, \text{dilation}=2^{l-1}}(\mathbf{H}^{(l-1)})\right)\right), \quad l \in \{1, 2, 3\}$$

### 2.2. Atención Cruzada Multi-Cabeza (8 Cabezas, $d_{\text{attn}} = 512$) y Attentive Pooling
Calculamos la afinidad enmascarando los tokens de padding de $\mathbf{C}$ con $-\infty$:

$$\mathbf{A}_{qc} = \text{Softmax}\left(\frac{(\tilde{\mathbf{Q}}\mathbf{W}_q)(\tilde{\mathbf{C}}\mathbf{W}_k)^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}(C)\right) \in \mathbb{R}^{1 \times 8 \times L_q \times L_c}$$

En lugar de promediar todos los tokens por igual, un cabezal de **Attentive Pooling** pondera los tokens de acción y negación:

$$\boldsymbol{\beta}_q = \text{Softmax}\left(\mathbf{w}_{\text{pool}}^T \tanh(\mathbf{W}_p \tilde{\mathbf{Q}}) + \mathbf{m}_q^{\text{log}}\right), \qquad \mathbf{q}_{\text{att}} = \sum_{i=1}^{L_q} \beta_{q, i} \tilde{\mathbf{Q}}_i \in \mathbb{R}^{384}$$

El vector de interacción NLI $\mathbf{u}_{\text{nli}} \in \mathbb{R}^{1536}$ atraviesa un clasificador residual profundo ($1536 \to 1024 \to 512 \to 4$) hacia los **4 estados conversacionales**:
* $p_0$: `TRANSICION` | $p_1$: `CONTINUACION` | $p_2$: `CORRECCION` | $p_3$: `ABANDONO`

### 2.3. Compuerta Diferenciable Por-Canal ($\boldsymbol{\alpha} \in [0, 1]^{384}$) y Escalado Escalar
La fusión contextual modula cuánto del contexto $\mathbf{M}_{qc}$ se inyecta en la consulta:

$$\alpha_{\text{scalar}} = \sigma\left(\mathbf{w}_g^T [\mathbf{p}_{\text{estado}} \;\Vert{}\; \mathbf{q}_{\text{att}} \;\Vert{}\; \mathbf{m}_{\text{att}}] + b_g\right) \in [0, 1]$$

$$\mathbf{Q}^* = \text{LayerNorm}\left(\tilde{\mathbf{Q}} + \alpha_{\text{scalar}} \cdot \mathbf{M}_{qc}\right) \in \mathbb{R}^{1 \times L_q \times 384}$$

---

## 3. Etapa 2: Enrutador y 6 Macro-Expertos SwiGLU ($\sim 264.0\text{M}$ Parámetros Totales)

Recibe $\mathbf{Q}^* \in \mathbb{R}^{1 \times L_q \times 384}$ y el tensor 3D de las $K$ herramientas $\mathbf{T} \in \mathbb{R}^{K \times L_t \times 384}$.

### 3.1. Enrutador Top-2 con Balanceo de Carga
El Router evalúa $[\mathbf{q}_{\text{att}}^* \;\Vert{}\; \mathbf{p}_{\text{estado}}] \in \mathbb{R}^{388}$ mediante una red $388 \to 512 \to 6$ y selecciona los **2 expertos principales** ($\sim 88\text{M}$ de parámetros activos en esta etapa por consulta):
* **Experto 1:** Acción e Imperativos.
* **Experto 2:** Consulta de Catálogo y Búsqueda (`mitumbes`, filtros de negocio).
* **Experto 3:** Resolución Implícita y Vaga (`vague-implicit-juicio`).
* **Experto 4:** Verificación de Especificidad y Schema (discriminación fina entre tools hermanas).
* **Experto 5:** Orquestación Multi-Herramienta y Flujos de Código (`coding-programmer-workflows`).
* **Experto 6:** Filtro Adversarial, Texto Web y Chitchat (`webpage-prediction-null`).

### 3.2. Proyección de Alta Dimensión ($d_{\text{proj}} = 1024$) y MaxSim Enmascarado
Cada experto $e \in \{1 \dots 6\}$ proyecta tokens de consulta y herramientas hacia un subespacio rico de dimensión **$d_{\text{proj}} = 1024$** mediante bloques residuales SwiGLU ($384 \to 1024 \to 1024$):

$$\mathbf{S}_e(k, i, j) = \langle \hat{\mathbf{Q}}_e(1, i), \, \hat{\mathbf{T}}_e(k, j) \rangle \in [-1, 1]^{K \times L_q \times L_t}$$

Aplicando las máscaras exactas de tokens válidos $\mathbf{m}_q \in \{0, 1\}^{L_q}$ y $\mathbf{m}_{t, k} \in \{0, 1\}^{L_t}$ (con longitud real $N_q = \sum \mathbf{m}_q$ y $N_{t, k} = \sum \mathbf{m}_{t, k}$):
1. **MaxSim Forward Enmascarado (Cobertura de la Query):**
   $$\mu_{\text{fwd}, k} = \frac{1}{N_q} \sum_{i=1}^{L_q} m_{q, i} \left(\max_{j \,:\, m_{t,k,j}=1} \mathbf{S}_e(k, i, j)\right)$$
2. **MaxSim Backward Enmascarado (Especificidad Real sin sesgo de longitud):**
   $$\mu_{\text{bwd}, k} = \frac{1}{N_{t, k}} \sum_{j=1}^{L_t} m_{t, k, j} \left(\max_{i \,:\, m_{q,i}=1} \mathbf{S}_e(k, i, j)\right)$$
3. **Scorer Bilineal Profundo ($4104 \to 1536 \to 512$):**
   Genera el vector latente de cada herramienta $\mathbf{z}_{e, k} \in \mathbb{R}^{512}$, combinándose vía Top-2 en $\mathbf{h}_k = \sum_{e \in \text{Top2}} g_e \mathbf{z}_{e, k} \in \mathbb{R}^{512}$.

---

## 4. Etapa 3: Transformer Set-Attention de 3 Capas y Cabeza de Abstención ($\sim 16.0\text{M}$ Parámetros)

Sobre la matriz de herramientas candidatas $\mathbf{H} \in \mathbb{R}^{1 \times K \times 512}$:
1. **Pila de 3 Capas Pre-LN Transformer Encoder (8 cabezas, FFN $512 \to 2048 \to 512$):** Permite que las $K$ herramientas compitan entre sí, suprimiendo herramientas redundantes y resaltando combinaciones complementarias (Multi-Tool).
2. **Cabeza de Abstención Consciente de Cardinalidad ($K$):**
   Integra `[top1, top2, top3, margen_12, margen_23, entropia_shannon, mean_s, std_s, log(K)/log(200), estado_probs(4), gate_alpha(1), max_pool(512), mean_pool(512)]` ($\text{dim} = 1038 \to 512 \to 128 \to 1$) para emitir `abstencion_prob` $\in (0, 1)$ inmune a cambios en el tamaño del catálogo.