# Plan 02: Estrategia de Datos Masivos Multilingües y Destilación con Profesor de 568M

## 1. Por qué un Modelo de 300M Necesita un Dataset Mixto de ~100,000 Muestras

Un modelo de **302.4M de parámetros** sobreajustaría (*overfitting*) si se entrena únicamente con los $2,500$ casos base del repositorio. Además, el profesor antiguo (`ms-marco-MiniLM-L-6-v2`) queda descartado como generador de etiquetas porque degradaba el español.

En su lugar, ejecutamos en **Google Colab Pro (GPU L4)** un pipeline de **Destilación Híbrida** que combina:
1. **Verdad Terreno Exacta (*Ground Truth*):** Tus suites de `tests/*.test.ts` y variaciones sintéticas.
2. **Nuevo Profesor Multilingüe Pesado (`BAAI/bge-reranker-v2-m3` — 568M parámetros):** Evalúa en GPU los pares `(Query + Contexto, Tool)` para producir *soft targets* precisos en español, inglés y código.
3. **4 Librerías Abiertas de Hugging Face (`datasets`):** Inyectan diversidad masiva de herramientas, negaciones y casos fuera de dominio.

---

## 2. Composición del Dataset de 100,000 Muestras (`Data Mix`)

| Proporción | Fuente de Datos / Librería Hugging Face | Muestras | Etapa del MoE que Refuerza |
| :--- | :--- | :--- | :--- |
| **40%** | **Suites propias (`tests/*.test.ts`) + Expansión Sintética** (`mitumbes`, `coding-programmer-workflows`, `complex-system-plan`) | $40,000$ | **Todas las etapas** (Garantiza $100\%$ de fidelidad en herramientas de tus tenants). |
| **25%** | **`Salesforce/xlam-function-calling-60k` + `Team-ACE/ToolACE`** | $25,000$ | **Etapa 2 (Expertos 1, 2, 4, 5):** Discriminación fina de schemas y llamadas multi-herramienta. |
| **20%** | **`MadeAgents/xlam-irrelevance-7.5k` + `clinc_oos` + `webpage-prediction-null`** | $20,000$ | **Etapa 3 (`abstencion_head`) y Experto 6:** Retorno nulo `[]` ante preguntas trampa o texto web pegado. |
| **15%** | **`facebook/xnli` (es/en) + `AmazonScience/massive` (`es-ES`, `es-MX`)** | $15,000$ | **Etapa 1 (`Etapa1CompuertaEpisodicaNLI`):** Negaciones, correcciones (`CORRECCION`), seguimiento y `ABANDONO`. |

---

## 3. Script de Descarga y Unificación en Google Colab (`training/01_build_300m_dataset.py`)

```python
# training/01_build_300m_dataset.py
import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# Nuevo Profesor Multilingüe de 568M parámetros (reemplaza a MiniLM inglés)
TEACHER_RERANKER = "BAAI/bge-reranker-v2-m3"
teacher_tok = AutoTokenizer.from_pretrained(TEACHER_RERANKER)
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_RERANKER, torch_dtype=torch.float16
).to(device).eval()

@torch.no_grad()
def score_with_bge_teacher(query_ctx: str, tool_texts: list[str]) -> list[float]:
    if not tool_texts:
        return []
    pairs = [[query_ctx, t] for t in tool_texts]
    inputs = teacher_tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = teacher_model(**inputs).logits.view(-1).float()
    return torch.sigmoid(logits).cpu().tolist()