---

## 📁 Carpeta 2: `C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\docs\plan\`

### 5. `docs/plan/plan_01_unificaci_n_de_modelos_y_optimizaci_n_de_recursos.md`

```markdown
# Plan 01: Resolución de Necesidades — Escalado a 300M de Parámetros y Corrección de Decisiones

## 1. Diagnóstico Raíz: ¿Por qué fallaban tanto el sistema anterior como el prototipo MoE inicial?

A lo largo de la investigación y pruebas de estrés se identificaron **5 causas técnicas exactas** por las que el sistema tomaba decisiones erróneas:

1. **Techo de Cristal del Profesor Antiguo (`ms-marco-MiniLM-L-6-v2`):** El Cross-Encoder anterior tenía solo $22.7\text{M}$ de parámetros entrenados en inglés. Destilar desde él transmitía su ceguera ante matices e instrucciones compuestas en español.
2. **Destrucción de Negaciones por Promedio Simple (`mean(dim=1)`):** Promediar todos los tokens de una consulta diluía partículas críticas como *"no"*, *"excepto"*, *"sin"* o *"en lugar de"*, provocando que el modelo activara herramientas que el usuario había prohibido explícitamente.
3. **Campo Receptivo Corto (`kernel_size=3` sin dilatación):** Una ventana de 3 tokens no podía relacionar una negación al inicio de la oración con el verbo de la herramienta ubicado 10 palabras después.
4. **Sesgo de Longitud en `MaxSim` por Falta de Máscaras de Padding:** Al rellenar herramientas hasta $96$ tokens con ceros sin enmascarar, `mean_max_bwd` castigaba a las herramientas con descripciones concisas y premiaba falsos positivos largos.
5. **Inestabilidad de la Entropía de Abstención ante Catálogos Variables ($K$):** La cabeza de abstención no recibía la cardinalidad $\log(K)$, dudando cuando el filtro previo entregaba pocas candidatas ($K=12$) frente a catálogos grandes ($K=150$).

---

## 2. Solución Arquitectónica Definitiva: `remtk-juicio-moe-300m-int8`

Escalamos el cabezal de decisión a **$\sim 302.4\text{M}$ de parámetros totales** ($\sim 108\text{M}$ activos por consulta gracias al enrutamiento Top-2) entrenado en **Google Colab Pro (GPU L4 / A100)** mediante un nuevo Profesor Multilingüe de Estado del Arte (**`BAAI/bge-reranker-v2-m3` de 568M params** + supervisión exacta de tests y datasets abiertos):

* **Qué se mantiene intacto:** `multilingual-e5-small` congelado para el puerto `6777`, Qdrant (`6775`) y la caché de tensores de herramientas en RAM.
* **Qué entra en `src/juicio/`:** `models/remtk-juicio-moe-int8/model_quantized.onnx` (**~295 MB** en `INT8`), con Convoluciones Dilatadas (`dilation=1, 2, 4`), *Attentive Pooling*, *MaxSim Enmascarado* y *Set-Attention Transformer de 3 capas*.

---

## 3. Presupuesto Comparativo de Rendimiento y Precisión

| Métrica del Sistema | Stack Antiguo (3 Modelos separados) | Prototipo MoE Ligero (18.5M) | **Nuevo `remtk-juicio-moe-300m` (INT8)** |
| :--- | :--- | :--- | :--- |
| **Parámetros del Decisor** | ~300M (repartidos en modelos inconexos) | 18.5M (6.5M activos) | **302.4M (108M activos por turno)** |
| **Memoria RAM en Node.js** | 850 MB – 1,200 MB | ~210 MB | **~535 MB (cabe holgado en 16 GB DDR5)** |
| **Latencia en CPU (`POST /predict`)** | 65 ms – 180 ms | 4 ms – 8 ms | **14 ms – 22 ms (1 sola llamada C++)** |
| **Comprensión de Negaciones Distancias** | Baja (requería regex compensatorios) | Media (ventana de 3 tokens) | **Alta (Dilated Conv 15 tokens + Attentive Pool)** |
| **Precisión en Especificidad y Retorno Nulo** | Degradada en español | Sensible al padding de tools | **Inmune al padding (MaxSim Enmascarado + $\log(K)$)** |