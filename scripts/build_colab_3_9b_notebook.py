import re
import json
from pathlib import Path

source_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_3_8b_standalone.py")
with open(source_path, "r", encoding="utf-8") as f:
    raw_content = f.read()

# Dividir por encabezados de celda
cell_splits = re.split(r'# =+\s*\n# (CELL \d+:.*?)\n# =+', raw_content)
markdown_header = cell_splits[0].strip()

# Modificar encabezado general
markdown_header = markdown_header.replace("3.8B", "3.9B").replace("48,000", "235,000")
markdown_header = markdown_header.replace("~3.82B", "~3.92B")

cells_data = []

# Celda inicial de Markdown
cells_data.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" for line in markdown_header.split("\n")]
})

for i in range(1, len(cell_splits), 2):
    cell_title = cell_splits[i].strip()
    cell_code = cell_splits[i+1].strip()

    # Ajustes específicos por celda
    cell_title = cell_title.replace("3.8B", "3.9B").replace("48,000", "235,000").replace("48000", "235000").replace("175,000", "235,000").replace("100,000", "235,000")
    cell_code = cell_code.replace("3.8B", "3.9B").replace("48,000", "235,000").replace("175,000", "235,000").replace("100,000", "235,000")

    if "CELL 2:" in cell_title:
        # Actualizar parámetros a 3.9B y 235k episodios
        cell_code = cell_code.replace('"d_exp_ffn": 4736', '"d_exp_ffn": 4928')
        cell_code = cell_code.replace('N_SYNTH_EPISODES = 48000', 'N_SYNTH_EPISODES = 235000')
        cell_code = cell_code.replace('checkpoints_jev_3_8b', 'checkpoints_jev_3_9b')
        cell_code = cell_code.replace('remtk-juicio-jev-moe-3.8b', 'remtk-juicio-jev-moe-3.9b')
        cell_code = cell_code.replace('qwen_100pct_generative_corpus_3_8b_v2.json', 'qwen_100pct_generative_corpus_3_9b_235k.json')

    elif "CELL 6:" in cell_title:
        # Expandir longitud de contexto para capturar diálogos multi-turno largos
        cell_code = cell_code.replace(
            'MAX_QUERY_LEN, MAX_CTX_LEN, MAX_TOOL_LEN = 40, 56, 56',
            'MAX_QUERY_LEN, MAX_CTX_LEN, MAX_TOOL_LEN = 48, 96, 64'
        )

    elif "CELL 7:" in cell_title:
        # Inyección obligatoria de las 3 hermanas gemelas latentes más cercanas
        old_pattern = 'cands_set = set(latent_negs)'
        new_pattern = (
            "cands_set = set(latent_negs)\n"
            "    # Minería obligatoria de las 3 hermanas gemelas latentes más cercanas para contraste fino\n"
            "    if active_target_tool in LATENT_NEAREST_NEIGHBORS:\n"
            "        cands_set.update(LATENT_NEAREST_NEIGHBORS[active_target_tool][:3])"
        )
        if old_pattern in cell_code:
            cell_code = cell_code.replace(old_pattern, new_pattern, 1)

    elif "CELL 8:" in cell_title:
        # Desbloqueo del MoE + Set-Transformer: el anchor léxico es solo un sesgo residual (0.20),
        # permitiendo que las 8 capas de razonamiento atencional dominen la decisión fina (>87-92% Top-1)
        old_fusion = (
            "combined_choice = (1.0 - 0.55 * fusion_alpha) * anchor_logit + (0.65 + 0.85 * fusion_alpha) * mlp_tool_logit\n"
            "        combined_aff = 0.70 * anchor_logit + 0.85 * mlp_aff_logit"
        )
        new_fusion = (
            "combined_choice = 0.20 * anchor_logit + (1.20 + 0.60 * fusion_alpha) * mlp_tool_logit\n"
            "        combined_aff = 0.25 * anchor_logit + 1.25 * mlp_aff_logit"
        )
        if old_fusion in cell_code:
            cell_code = cell_code.replace(old_fusion, new_fusion, 1)

    elif "CELL 11:" in cell_title:
        cell_code = cell_code.replace('EPOCHS = 16', 'EPOCHS = 12')

    elif "CELL 14:" in cell_title:
        cell_code = cell_code.replace('REMTK-JUICIO-JEV-MOE-3.8B-INT8-400TOOLS', 'REMTK-JUICIO-JEV-MOE-3.9B-INT8-400TOOLS')
        cell_code = cell_code.replace('remtk-juicio-jev-moe-3.8b', 'remtk-juicio-jev-moe-3.9b')
        cell_code = cell_code.replace('remtk_juicio_jev_moe_3_8b_hybrid_int8', 'remtk_juicio_jev_moe_3_9b_hybrid_int8')

    cell_content = f"# ==============================================================================\n# {cell_title}\n# ==============================================================================\n{cell_code}"
    cells_data.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell_content.split("\n")]
    })

notebook_json = {
    "cells": cells_data,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Guardar notebook formateado correctamente
out_ipynb = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_colab.ipynb")
with open(out_ipynb, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2, ensure_ascii=False)

print(f"Jupyter Notebook generado exitosamente con {len(cells_data)} celdas en: {out_ipynb}")

# Guardar también el script standalone python
out_py = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_3_9b_standalone.py")
full_script_lines = ['"""\n' + markdown_header + '\n"""\n\n']
for c in cells_data:
    if c["cell_type"] == "code":
        full_script_lines.extend(c["source"])
        full_script_lines.append("\n\n")

with open(out_py, "w", encoding="utf-8") as f:
    f.writelines(full_script_lines)

print(f"Script Standalone Python generado exitosamente en: {out_py}")
