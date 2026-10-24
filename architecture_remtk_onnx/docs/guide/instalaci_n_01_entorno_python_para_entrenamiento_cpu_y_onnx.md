# Instalación 01: Entorno de Entrenamiento en CPU y Exportación a ONNX (`training/`)

Esta guía detalla la instalación aislada del entorno Python necesario para entrenar `remtk-juicio-moe`, destilar los modelos actuales y exportar el binario cuantizado `model_quantized.onnx` sin ensuciar las dependencias de Node.js ni requerir tarjeta gráfica (GPU).

---

## 1. Requisitos del Sistema para Entrenamiento Local en CPU

* **Procesador:** x86_64 con soporte de instrucciones vectoriales **AVX2** o **AVX-512 VNNI** (Intel Core / Xeon o AMD Ryzen / EPYC).
* **Memoria RAM Libre:** Mínimo $4\text{ GB}$ disponibles durante el entrenamiento ($8\text{ GB}$ recomendados).
* **Python:** Versión `3.10`, `3.11` o `3.12`.

---

## 2. Creación del Entorno Virtual y Paquetes CPU-Only

Desde la raíz de `remtk-prediction/`, ejecuta los siguientes comandos para crear la carpeta `training/` y su entorno virtual ligero (descargando la versión de PyTorch específica para CPU, que pesa $180\text{ MB}$ en lugar de los $2.5\text{ GB}$ de la versión CUDA):

```bash
# 1. Crear directorio de entrenamiento y entorno virtual
mkdir -p training
cd training
python3 -m venv .venv

# 2. Activar el entorno virtual
# En Linux / macOS:
source .venv/bin/activate
# En Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 3. Actualizar pip e instalar PyTorch optimizado para CPU (OneDNN / AVX2)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar librerías de exportación ONNX, cuantización INT8 y manejo de tensores
pip install onnx==1.16.2 onnxruntime==1.19.2 optimum[onnxruntime]==1.22.0 safetensors==0.4.5 transformers==4.45.2 datasets==3.0.1 numpy==1.26.4 tqdm
```

---

## 3. Variables deOptimización para Velocidad Máxima en CPU

Antes de lanzar los scripts de pre-cálculo de tensores y entrenamiento, exporta estas variables de entorno para que PyTorch y OpenMP utilicen todos los núcleos físicos de tu CPU sin sufrir contención de hilos:

```bash
# Ajustar al número de núcleos físicos de tu CPU (ej. 8)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8
export ONEDNN_MAX_CPU_ISA=AVX2
```

---

## 4. Script de Exportación y Cuantización a `INT8` (`training/04_export_quantize_onnx.py`)

Guarda este script en `training/04_export_quantize_onnx.py` para compilar los pesos entrenados de PyTorch al archivo `.onnx` final que consumirá TypeScript:

```python
# training/04_export_quantize_onnx.py
import os
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from architecture_moe import RemtkJuicioMoE

OUTPUT_DIR = "../models/remtk-juicio-moe-int8"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FP32_ONNX_PATH = os.path.join(OUTPUT_DIR, "model_fp32.onnx")
INT8_ONNX_PATH = os.path.join(OUTPUT_DIR, "model_quantized.onnx")

def export_and_quantize(checkpoint_path: str = "./checkpoints/best_moe.pt"):
    model = RemtkJuicioMoE()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    # Tensores dummy representativos para trazar el grafo dinámico
    dummy_query = torch.randn(1, 32, 384, dtype=torch.float32)
    dummy_context = torch.randn(1, 64, 384, dtype=torch.float32)
    dummy_tools = torch.randn(20, 96, 384, dtype=torch.float32)

    print("[1/3] Exportando grafo PyTorch de 3 etapas a ONNX FP32...")
    torch.onnx.export(
        model,
        (dummy_query, dummy_context, dummy_tools),
        FP32_ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["query_tokens", "context_tokens", "tools_tokens"],
        output_names=[
            "ranked_scores",
            "estado_probs",
            "abstencion_prob",
            "topic_shift_score",
            "sparse_gates"
        ],
        dynamic_axes={
            "query_tokens": {1: "q_len"},
            "context_tokens": {1: "ctx_len"},
            "tools_tokens": {0: "num_tools", 1: "tool_len"},
            "ranked_scores": {0: "num_tools"}
        }
    )

    print("[2/3] Verificando integridad del grafo ONNX...")
    onnx_model = onnx.load(FP32_ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    print("[3/3] Aplicando cuantización dinámica INT8 (AVX2/VNNI)...")
    quantize_dynamic(
        model_input=FP32_ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True
    )

    size_mb = os.path.getsize(INT8_ONNX_PATH) / (1024 * 1024)
    print(f"Listo: {INT8_ONNX_PATH} generado correctamente ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_and_quantize()
```