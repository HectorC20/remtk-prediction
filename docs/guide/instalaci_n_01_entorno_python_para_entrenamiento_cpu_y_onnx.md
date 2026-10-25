# Guía de Instalación 01: Entorno de Entrenamiento en Google Colab Pro (GPU L4) y PC Local (DDR5) para el Modelo 300M

## 1. Configuración en Google Colab Pro (Recomendado para Entrenamiento Completo de 300M)

Para entrenar los **$302.4\text{M}$ de parámetros** en **~20 minutos**:

1. Abre [Google Colab](https://colab.research.google.com) y ve a **Entorno de ejecución $\rightarrow$ Cambiar tipo de entorno de ejecución**.
2. Configura los siguientes parámetros:
   * **Tipo de entorno de ejecución:** `Python 3`
   * **Acelerador por hardware:** **`GPU L4`** (24 GB VRAM) — *Usa `GPU T4` mientras depuras scripts y `GPU L4` al lanzar las 6 épocas completas*.
   * **Alta capacidad de RAM:** **Activado (ON)** (otorga ~32 GB a 51 GB de RAM del sistema).
3. En la primera celda del Notebook, monta tu Google Drive e instala el stack de exportación ONNX:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers datasets accelerate onnx onnxruntime optimum safetensors