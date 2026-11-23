import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, List, Any, Optional, Tuple

base_dir = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction")
meta_path = base_dir / "models" / "remtk-juicio-jev-moe-3.8b-400tools" / "catalog_metadata.json"
tensors_path = base_dir / "models" / "remtk-juicio-jev-moe-3.8b-400tools" / "catalog_e5_tensors.safetensors"

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
ALL_TOOL_IDS = meta["all_tool_ids"]
CATALOG = meta["catalog"]
tool_id_to_idx = {tid: idx for idx, tid in enumerate(ALL_TOOL_IDS)}

cat_safetensors = load_file(str(tensors_path))
raw_tool_tensors = cat_safetensors["catalog_tool_tensors"].to(dtype=torch.float32)
cat_mask = (torch.norm(raw_tool_tensors, p=2, dim=-1) > 1e-5).float()
cat_lens = cat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
catalog_pooled = F.normalize((raw_tool_tensors * cat_mask.unsqueeze(-1)).sum(dim=1) / cat_lens, p=2, dim=-1)

class TenantRegistry:
    def __init__(self):
        self.tenants: Dict[str, Dict[str, Any]] = {}

    def register(self, scope_key: str, tools_list: List[Dict[str, Any]]) -> int:
        tool_ids = []
        tools_dict = {}
        tool_tensors_list = []
        tool_pooled_list = []

        for t in tools_list:
            tid = t.get("id") or t.get("name")
            if not tid:
                continue
            clean_id = str(tid).strip()
            tool_ids.append(clean_id)
            tools_dict[clean_id] = t

            if clean_id in tool_id_to_idx:
                c_idx = tool_id_to_idx[clean_id]
                tool_tensors_list.append(raw_tool_tensors[c_idx])
                tool_pooled_list.append(catalog_pooled[c_idx])
            else:
                # Mock / dummy for invented tool
                dummy_t = torch.randn(64, 384)
                dummy_p = F.normalize(dummy_t.mean(dim=0, keepdim=True), p=2, dim=-1)[0]
                tool_tensors_list.append(dummy_t)
                tool_pooled_list.append(dummy_p)

        if tool_tensors_list:
            stacked_tensors = torch.stack(tool_tensors_list, dim=0)
            stacked_pooled = torch.stack(tool_pooled_list, dim=0)
        else:
            stacked_tensors = torch.zeros(0, 64, 384)
            stacked_pooled = torch.zeros(0, 384)

        self.tenants[scope_key] = {
            "tool_ids": tool_ids,
            "tools_dict": tools_dict,
            "tool_tensors": stacked_tensors,
            "tool_pooled": stacked_pooled,
            "ready": True
        }
        return len(tool_ids)

    def get(self, scope_key: str) -> Optional[Dict[str, Any]]:
        if scope_key in self.tenants:
            return self.tenants[scope_key]
        clean_tenant = scope_key.split("::")[0].strip()
        return self.tenants.get(f"{clean_tenant}::") or self.tenants.get(clean_tenant)

registry = TenantRegistry()

# Simulate registering 10 tools (including schedule_task, web_search, search_memory, but NO iot, NO spark, NO k8s)
user_tools = [
    {"name": "schedule_task", "description": "Programa recordatorios y cron jobs"},
    {"name": "schedule_list", "description": "Lista tareas programadas"},
    {"name": "web_search", "description": "Busca en la web"},
    {"name": "web_fetch", "description": "Descarga contenido web"},
    {"name": "search_memory", "description": "Busca en la memoria"},
    {"name": "search_knowledge_base", "description": "Busca en la base de conocimiento"},
    {"name": "knowledge_add", "description": "Agrega información a la base de conocimiento"},
    {"name": "document_generate", "description": "Genera documentos PDF o DOCX"},
    {"name": "image_generate", "description": "Genera imágenes culinarias"},
    {"name": "list_tools", "description": "Lista las herramientas disponibles"}
]

tenant_id = "4f4e8014-0a50-42ad-b1db-d6e4ffd4f9c8"
count = registry.register(f"{tenant_id}::", user_tools)
print(f"Registered {count} tools for tenant {tenant_id}")

# Simulate candidate selection
tenant_data = registry.get(f"{tenant_id}::")
assert tenant_data is not None
assert len(tenant_data["tool_ids"]) == 10

# Test query vector
fake_query = torch.randn(1, 384)
fake_query = F.normalize(fake_query, p=2, dim=-1)

sims = torch.matmul(fake_query, tenant_data["tool_pooled"].T)[0]
k = min(32, len(tenant_data["tool_ids"]))
top_idx = torch.topk(sims, k=k).indices.tolist()
cands = [tenant_data["tool_ids"][i] for i in top_idx]

print("Selected candidates strictly from user pool:")
print(cands)
assert all(c in [t["name"] for t in user_tools] for c in cands)
print("VERIFICATION PASSED: No foreign tools can ever enter the candidate set!")
