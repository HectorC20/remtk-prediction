import os
import json
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

MODEL_DIR = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk-juicio-jev-moe-3.8b-400tools")
SAFETENSORS_PATH = MODEL_DIR / "remtk_juicio_jev_moe_3_8b_hybrid_int8-001.safetensors"
CATALOG_TENSORS_PATH = MODEL_DIR / "catalog_e5_tensors.safetensors"
CATALOG_META_PATH = MODEL_DIR / "catalog_metadata.json"

print(f"Loading metadata from {CATALOG_META_PATH}...")
with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

cfg = meta["config"]
all_tool_ids = meta["all_tool_ids"]
catalog = meta["catalog"]
slang_rules = meta.get("slang_rules", [])
print(f"Config: {cfg}")
print(f"Total tools: {len(all_tool_ids)}")

print(f"Loading safetensors from {SAFETENSORS_PATH}...")
t0 = time.perf_counter()
state_dict = load_file(str(SAFETENSORS_PATH), device="cpu")
print(f"Safetensors loaded in {time.perf_counter() - t0:.2f}s ({len(state_dict)} keys)")

print(f"Loading catalog tensors from {CATALOG_TENSORS_PATH}...")
catalog_tensors = load_file(str(CATALOG_TENSORS_PATH), device="cpu")["catalog_tool_tensors"]
print(f"Catalog tensors shape: {catalog_tensors.shape}")
