cd "C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction"
mkdir training -Force
cd training

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install transformers datasets onnx onnxruntime optimum safetensors numpy tqdm

# training/03_train_and_export_300m.py
import os
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from architecture_moe import RemtkJuicioMoE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[300M Trainer] Dispositivo activo: {device}")

    # Instanciar arquitectura de ~302.4M parámetros
    model = RemtkJuicioMoE(
        num_experts=6,   # Mantiene compatibilidad 1:1 con activeExperts[6] en TypeScript
        d_model=384,     # Fijo por contrato con multilingual-e5-small
        d_proj=1024,     # Proyección de alta resolución por experto
        d_out=512        # Dimensión de Set-Attention en Etapa 3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[300M Trainer] Parámetros entrenables totales: {total_params / 1e6:.2f} M")
    assert 280_000_000 <= total_params <= 330_000_000, "El modelo debe estar en la escala de ~300M params"

    # --- AQUÍ CORRE EL LOOP DE ENTRENAMIENTO CON L_TOTAL ---
    # (Ver Plan 03 para los pesos lambda_1..lambda_6)

    # Exportación a ONNX (opset 17) y Cuantización Dinámica INT8
    model_cpu = model.to("cpu").eval()
    os.makedirs("../models/remtk-juicio-moe-int8", exist_ok=True)
    fp32_path = "./model_300m_fp32.onnx"
    int8_path = "../models/remtk-juicio-moe-int8/model_quantized.onnx"

    dummy_q = torch.randn(1, 32, 384)
    dummy_c = torch.randn(1, 64, 384)
    dummy_t = torch.randn(20, 64, 384)
    # Simular padding con ceros en los últimos tokens para validar la máscara automática
    dummy_q[:, 24:, :] = 0.0
    dummy_t[:, 48:, :] = 0.0

    torch.onnx.export(
        model_cpu,
        (dummy_q, dummy_c, dummy_t),
        fp32_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["query_tokens", "context_tokens", "tools_tokens"],
        output_names=["ranked_scores", "estado_probs", "abstencion_prob", "topic_shift_score", "sparse_gates"],
        dynamic_axes={
            "query_tokens": {1: "q_len"},
            "context_tokens": {1: "ctx_len"},
            "tools_tokens": {0: "num_tools", 1: "tool_len"},
            "ranked_scores": {0: "num_tools"}
        }
    )

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8
    )
    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"[300M Trainer] Exportación exitosa: {int8_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()


    ---

## 📁 Carpeta 4: `C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\docs\deployment\`

### 11. `docs/deployment/skill_gu_a_operativa_de_los_10_archivos_markdown.md`

```markdown
---
name: remtk-moe-300m-onnx-architect
description: Skill operativa para implementar, entrenar en Google Colab Pro (GPU L4) o CPU DDR5, auditar y mantener el motor Mixture of Experts (MoE) de 300M de parámetros y 3 etapas en ONNX dentro de remtk-prediction.
---

# Skill: Guía Operativa de los 10 Archivos Markdown (`remtk-juicio-moe-300m`)

Este documento define el protocolo obligatorio de ingeniería para operar el modelo de toma de decisiones **`remtk-juicio-moe-300m`** (**~302.4M de parámetros**, **~295 MB en INT8**) dentro de `remtk-prediction`.

---

## 1. Mapa de Navegación de los 10 Documentos Técnicos

Según la tarea que vayas a ejecutar, consulta el documento correspondiente:

| Fase Operativa | Archivos de Referencia | Objetivo Técnico |
| :--- | :--- | :--- |
| **A. Diagnóstico y Capacidad (300M)** | `docs/plan/plan_01_unificaci_n_de_modelos_y_optimizaci_n_de_recursos.md`<br>`docs/pattern/arquitectura_02_capacidad_de_tokens_y_ejes_din_micos.md` | Auditar las 5 correcciones estructurales, el enmascaramiento automático de padding por norma L2 y el presupuesto de RAM (~535 MB). |
| **B. Construcción del Dataset (100k muestras)** | `docs/plan/plan_02_destilaci_n_de_conocimiento_y_dataset.md` | Combinar tus tests (`40%`) con `xlam-function-calling` (`25%`), `xlam-irrelevance`/`clinc_oos` (`20%`) y `xnli`/`massive` (`15%`) usando `bge-reranker-v2-m3` como Profesor. |
| **C. Entrenamiento 300M (Colab Pro L4 / CPU)** | `docs/guide/instalaci_n_01_entorno_python_para_entrenamiento_cpu_y_onnx.md`<br>`docs/pattern/arquitectura_01_topolog_a_matem_tica_del_moe.md`<br>`docs/pattern/arquitectura_03_implementaci_n_pytorch_del_modelo_moe.md`<br>`docs/plan/plan_03_entrenamiento_en_cpu_y_funci_n_de_p_rdida.md` | Entrenar los $302.4\text{M}$ de parámetros en Google Colab Pro (`GPU L4` + `Alta capacidad de RAM`) y exportar a `model_quantized.onnx` (`INT8`). |
| **D. Integración TS y Despliegue** | `docs/pattern/arquitectura_04_integraci_n_typescript_en_juicioservice.md`<br>`docs/plan/plan_04_integraci_n_progresiva_y_validaci_n.md`<br>`docs/guide/instalaci_n_02_despliegue_en_node_js_docker_y_producci_n.md` | Integrar `MoEDecisionService` con `enableCpuMemArena: true`, validar `pnpm test:all` y desplegar en Node.js/Docker. |

---

## 2. Las 4 Reglas de Oro de la Arquitectura 300M

1. **Prohibición de Parches Léxicos (`if/else` o regex en TypeScript):**
   * Toda corrección de comportamiento se realiza agregando el escenario al dataset de `plan_02` y ejecutando 1 época de ajuste fino (~3.5 minutos en Colab Pro GPU L4).
2. **Inmutabilidad de `multilingual-e5-small`:**
   * El extractor base ($d = 384$) jamás se modifica para no invalidar el puerto `6777` ni los índices de Qdrant.
3. **Cero Padding Corrupto en Tensores (`0.0` estricto):**
   * Los buffers `Float32Array` de `query_tokens`, `context_tokens` y `tools_tokens` deben dejar en `0.0` exacto las posiciones no utilizadas para que el grafo ONNX calcule las máscaras `q_mask`, `c_mask` y `t_mask` sin sesgar `MaxSim`.
4. **Prohibido Volver a `mean(dim=1)` o `ms-marco-MiniLM`:**
   * Mantener siempre `AttentivePooling` en las Etapas 1 y 2, y usar exclusivamente `BAAI/bge-reranker-v2-m3` (o superior) como Profesor offline.

---

## 3. Diagnóstico Rápido en `/debug`

* **Si el modelo activa herramientas cuando el usuario dijo *"no hagas X, mejor haz Y"*:**
  * Verifica que los 3 bloques `DilatedGatedConvBlock` (`dilation = 1, 2, 4`) y `AttentivePooling` estén activos en `Etapa1CompuertaEpisodicaNLI`, y sube $\lambda_6$ a $0.45$.
* **Si una herramienta con descripción muy corta pierde contra una herramienta larga genérica:**
  * Verifica que `toolsTokens` tenga `0.0` en las posiciones de padding para que `t_len_valid` en `ExpertoInteraccionTool300M` divida únicamente entre el número real de tokens de la herramienta.