import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

# Add remtk-prediction root
sys.path.insert(0, r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts")
import serve_3_82b_production as srv

print("Checking loaded model...")
print(f"Model type: {type(srv.model)}")

# Create dummy inputs with K=10
B = 1
K = 10
q_tokens, q_pooled = srv.encode_text_tokens(["a ver con el recordatorio"], prefix="query: ", max_len=24)
a_tokens = torch.zeros(1, 24, 384)
a_pooled = torch.zeros(1, 384)
c_global = torch.zeros(1, 384)
ledger_queries = torch.zeros(1, 8, 384)
ledger_tools = torch.zeros(1, 8, 384)
ledger_is_mutable = torch.zeros(1, 8)
ledger_mask = torch.zeros(1, 8)
rej_vec = torch.zeros(1, 384)
has_rej = torch.tensor([[0.0]])

# 10 tools from catalog
cand_indices = list(range(10))
t_tokens = srv.raw_tool_tensors[cand_indices, :32, :].unsqueeze(0)
t_pooled = srv.catalog_pooled[cand_indices].unsqueeze(0)

print(f"Input shapes: t_tokens={t_tokens.shape}, t_pooled={t_pooled.shape}")
t0 = time.perf_counter()
with torch.no_grad():
    out = srv.model(
        q_tokens, q_pooled, a_tokens, a_pooled, c_global,
        ledger_queries, ledger_tools, ledger_is_mutable, ledger_mask,
        rej_vec, has_rej, t_tokens, t_pooled
    )
dt = (time.perf_counter() - t0) * 1000
print(f"Forward completed successfully in {dt:.1f}ms!")
print("Master gate:", out["master_gate"].item())
print("p_sink:", out["p_sink"].item())
print("gated_tool_scores shape:", out["gated_tool_scores"].shape)
print("gated_tool_scores:", out["gated_tool_scores"][0].tolist())
