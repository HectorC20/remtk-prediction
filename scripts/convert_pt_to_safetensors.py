import os
import sys
import time
import torch
from pathlib import Path
from safetensors.torch import save_file

BASE_DIR = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction")
CKPT_PATH = BASE_DIR / "models" / "remtk_juicio_jev_moe_3_82B_zero_dup_a100.pt"
OUTPUT_SAFETENSORS = BASE_DIR / "models" / "remtk_juicio_jev_moe_3_82B.safetensors"

print(f"Loading checkpoint from {CKPT_PATH} (mmap=True)...")
t0 = time.perf_counter()
ckpt = torch.load(str(CKPT_PATH), map_location="cpu", mmap=True)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

print(f"Loaded {len(state_dict)} tensors in {time.perf_counter() - t0:.2f}s")

clean_state_dict = {}
total_params = 0

for k, v in state_dict.items():
    if not isinstance(v, torch.Tensor):
        continue
    # Ensure tensor is contiguous and on CPU
    tensor = v.detach().cpu().contiguous()
    clean_state_dict[k] = tensor
    total_params += tensor.numel()

print(f"Total parameters: {total_params / 1e9:.3f}B")
print(f"Writing to {OUTPUT_SAFETENSORS}...")
t1 = time.perf_counter()
save_file(clean_state_dict, str(OUTPUT_SAFETENSORS))
t_write = time.perf_counter() - t1
file_size_gb = OUTPUT_SAFETENSORS.stat().st_size / (1024 ** 3)
print(f"Saved successfully in {t_write:.2f}s! File size: {file_size_gb:.2f} GB")
