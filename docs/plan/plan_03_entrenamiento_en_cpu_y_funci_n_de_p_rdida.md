---

### 7. `docs/plan/plan_03_entrenamiento_en_cpu_y_funci_n_de_p_rdida.md`

```markdown
# Plan 03: Entrenamiento del Modelo 300M (Colab Pro GPU L4 + Respaldo CPU DDR5) y Función de Pérdida

## 1. Estrategia Dual de Entrenamiento para 302.4M de Parámetros

Con el salto de $18.5\text{M}$ a **$302.4\text{M}$ de parámetros**, el proyecto soporta dos entornos de ejecución:

| Entorno de Entrenamiento | Hardware Seleccionado | Tiempo por Época (100k muestras) | Tiempo Total (6 Épocas) | Uso de Memoria |
| :--- | :--- | :--- | :--- | :--- |
| **Entorno Primario: Google Colab Pro** | **GPU NVIDIA L4 (24 GB VRAM)** + **Alta Capacidad de RAM (32–51 GB)** | **~3.5 minutos** | **~21 minutos** (~1.8 CU de Colab Pro) | ~11.4 GB VRAM (`BF16 Mixed Precision`) |
| **Entorno Secundario: PC Local (Fine-Tuning rápido)** | **CPU con 16 GB RAM DDR5** (usando `numpy.memmap` + `torch.bfloat16`) | ~42 min (sobre subconjunto de 25k muestras) | ~4.2 horas (o **~15 min** para 1 época incremental de parches) | ~9.2 GB RAM (Pesos + AdamW + Batch 16) |

---

## 2. Función de Pérdida Multi-Tarea Reforzada ($\mathcal{L}_{\text{Total-300M}}$)

Para garantizar que los **6 Macro-Expertos** se especialicen sin colapsar y que la Etapa 3 separe tajantemente la última herramienta válida de la primera inválida:

$$\mathcal{L}_{\text{Total}} = \lambda_1 \mathcal{L}_{\text{ListNet}} + \lambda_2 \mathcal{L}_{\text{Margin}} + \lambda_3 \mathcal{L}_{\text{Estado}} + \lambda_4 \mathcal{L}_{\text{Abstención}} + \lambda_5 \mathcal{L}_{\text{BalanceMoE}} + \lambda_6 \mathcal{L}_{\text{ContrastiveNLI}}$$

Con hiperparámetros calibrados para 300M:
* $\lambda_1 = 1.0$ (**ListNet + KL Teacher Distillation** sobre las $K$ herramientas).
* $\lambda_2 = 0.75$ (**Margen Estricto $\gamma = 0.35$** entre la peor herramienta positiva $i \in P$ y la mejor negativa $j \in N$):
  $$\mathcal{L}_{\text{Margin}} = \frac{1}{\vert{}P\vert{}\vert{}N\vert{}} \sum_{i \in P} \sum_{j \in N} \max\left(0, \; 0.35 - (s_i - s_j)\right)^2$$
* $\lambda_3 = 0.50$ (**Estado Conversacional NLI** con suavizado de etiquetas $\epsilon = 0.05$).
* $\lambda_4 = 1.10$ (**Focal Loss de Abstención** con $\gamma = 2.0$, penalizando falsos positivos en consultas nulas).
* $\lambda_5 = 0.05$ (**Balanceo de Carga de los 6 Expertos + Supervisión de `expert_hint`**).
* $\lambda_6 = 0.30$ (**Penalización de Compuerta $\alpha$:** fuerza $\alpha < 0.1$ cuando el estado es `TRANSICION` o `ABANDONO`, evitando que el historial contamine una consulta nueva).

---

## 3. Regulación contra Sobreajuste en 300M de Parámetros

Para entrenar $302.4\text{M}$ de parámetros con estabilidad:
1. **Optimizador `AdamW` con `weight_decay = 0.02`** y recorte de gradiente (`clip_grad_norm_ = 1.0`).
2. **OneCycleLR Scheduler:** `max_lr = 2e-4` con $10\%$ de *warmup* lineal y decaimiento coseno.
3. **Token Dropout (10%):** Durante el entrenamiento, el $10\%$ de los tokens de las descripciones de las herramientas se enmascaran al azar para obligar a los expertos a no memorizar palabras aisladas sino comprender la semántica global.