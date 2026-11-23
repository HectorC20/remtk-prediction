import json
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from safetensors.torch import load_file
from transformers import AutoTokenizer

onnx_path = r'C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\multilingual-e5-small-onnx\model.onnx'
meta_path = r'C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk-juicio-jev-moe-3.8b-400tools\catalog_metadata.json'
tensors_path = r'C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk-juicio-jev-moe-3.8b-400tools\catalog_e5_tensors.safetensors'
tokenizer_dir = r'C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\multilingual-e5-small-onnx'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
all_tools = meta['all_tool_ids']

cat_safetensors = load_file(tensors_path)
raw_tool_tensors = cat_safetensors['catalog_tool_tensors'].to(dtype=torch.float32)
cat_mask = (torch.norm(raw_tool_tensors, p=2, dim=-1) > 1e-5).float()
cat_lens = cat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
cat_pooled = F.normalize((raw_tool_tensors * cat_mask.unsqueeze(-1)).sum(dim=1) / cat_lens, p=2, dim=-1)

# Query
tokens = tokenizer(['query: a ver con el recordatorio'], padding="max_length", max_length=24, truncation=True, return_tensors='np')
inp = {
    'input_ids': tokens['input_ids'].astype(np.int64),
    'attention_mask': tokens['attention_mask'].astype(np.int64),
    'token_type_ids': np.zeros_like(tokens['input_ids']).astype(np.int64)
}

outs = session.run(None, inp)[0]
attn = torch.tensor(tokens['attention_mask'], dtype=torch.float32)
raw_out = torch.tensor(outs, dtype=torch.float32)
q_pooled = F.normalize((raw_out * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True), p=2, dim=-1)

sims = torch.matmul(q_pooled, cat_pooled.T)[0]
top32_idx = torch.topk(sims, k=32).indices.tolist()

print("Top 10 E5 candidate tools:")
for rank, idx in enumerate(top32_idx[:10], 1):
    t_name = all_tools[idx]
    s = sims[idx].item()
    print(f"{rank:2d}. {t_name:<35} sim={s:.4f}")

for target in ['schedule_task', 'cron_add_job', 'list_tools']:
    if target in all_tools:
        idx = all_tools.index(target)
        rank = (sims > sims[idx]).sum().item() + 1
        print(f"{target:<20}: rank={rank:3d}/400, sim={sims[idx].item():.4f}")
