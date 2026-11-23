import json
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

base_dir = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction")
meta_path = base_dir / "models" / "remtk-juicio-jev-moe-3.8b-400tools" / "catalog_metadata.json"
tensors_path = base_dir / "models" / "remtk-juicio-jev-moe-3.8b-400tools" / "catalog_e5_tensors.safetensors"
e5_dir = base_dir / "models" / "multilingual-e5-small-onnx"

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

all_tools = meta["all_tool_ids"]
catalog = meta["catalog"]

# Update schedule_task and cron_add_job in catalog with rich reminder keywords
if "schedule_task" in catalog:
    catalog["schedule_task"]["desc_es"] = "Programa recordatorios, avisos, alertas y tareas automáticas periódicas (cron job) para el futuro (executeAt o cron)."
    catalog["schedule_task"]["desc_en"] = "Schedules reminders, alerts, notices and automated recurring cron jobs for the future (executeAt or cron)."
    catalog["schedule_task"]["description"] = "Programa recordatorios, avisos o tareas periódicas cron job del agente / Schedules reminders, alerts or cron jobs"
    catalog["schedule_task"]["text_for_embedding"] = "passage: schedule_task [SCHEDULE_DEFERRED] (TEMPORAL_SCHEDULING): Programa recordatorios, alarmas, avisos y tareas programadas del agente. Crear recordatorio para mañana o fecha futura. Cron job recurrente. Schedules reminders, alerts and scheduled tasks."

if "cron_add_job" in catalog:
    catalog["cron_add_job"]["desc_es"] = "Programa recordatorios automáticos periódicos y tareas recurrentes (cron job) en el crontab del sistema operativo"
    catalog["cron_add_job"]["desc_en"] = "Schedules recurring automated reminders and cron job periodic tasks in operating system crontab"
    catalog["cron_add_job"]["description"] = "Programa recordatorios periódicos y tareas cron job en el crontab / Schedules recurring reminders and cron jobs"
    catalog["cron_add_job"]["text_for_embedding"] = "passage: cron_add_job [SCHEDULE_DEFERRED] (TEMPORAL_SCHEDULING): Programa recordatorios periódicos automáticos, cron job y alarmas programadas en el sistema. Programar recordatorio recurrente. Schedules recurring reminders and cron jobs."

tokenizer = AutoTokenizer.from_pretrained(str(e5_dir))
session = ort.InferenceSession(str(e5_dir / "model.onnx"), providers=['CPUExecutionProvider'])

cat_safetensors = load_file(str(tensors_path))
catalog_tool_tensors = cat_safetensors["catalog_tool_tensors"].clone()
max_tool_len = catalog_tool_tensors.shape[1]

def encode_tool_passage(text: str, max_len: int = max_tool_len) -> torch.Tensor:
    prefix = "passage: "
    full_text = text if text.startswith(prefix) else f"{prefix}{text}"
    toks = tokenizer([full_text], max_length=max_len, padding="max_length", truncation=True, return_tensors="np")
    inp = {
        "input_ids": toks["input_ids"].astype(np.int64),
        "attention_mask": toks["attention_mask"].astype(np.int64),
        "token_type_ids": np.zeros_like(toks["input_ids"]).astype(np.int64)
    }
    
    outs = session.run(None, inp)[0]
    raw_t = torch.tensor(outs, dtype=torch.float32)
    attn = torch.tensor(toks["attention_mask"], dtype=torch.float32)
    norm_t = F.normalize(raw_t, p=2, dim=-1)
    t_tokens = norm_t * attn.unsqueeze(-1)
    return t_tokens[0]

for target_id in ["schedule_task", "cron_add_job"]:
    if target_id in all_tools:
        idx = all_tools.index(target_id)
        emb = encode_tool_passage(catalog[target_id]["text_for_embedding"], max_len=max_tool_len)
        catalog_tool_tensors[idx] = emb
        print(f"Updated tensor for {target_id} at index {idx}")

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

save_file({"catalog_tool_tensors": catalog_tool_tensors}, str(tensors_path))
print("Successfully saved updated metadata and safetensors!")
