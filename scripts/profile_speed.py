import time
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction")
sys.path.insert(0, str(BASE_DIR / "scripts"))

from evaluate_3_8b_400tools_trained import (
    model, catalog_tensors, encode_tokens, extract_hybrid_candidates, catalog_pooled, tool_id_to_idx, DEVICE
)

q = "Quiero agendar una cita con el doctor para este viernes a las 4pm."
c = ""

q_ten = encode_tokens([q], prefix="query: ", max_len=40, apply_canonical=True)
c_ten = encode_tokens([c], prefix="passage: ", max_len=56, apply_canonical=False)
cands = extract_hybrid_candidates(q, c, q_ten, c_ten, catalog_pooled, k_filter=36)
c_idxs = torch.tensor([[tool_id_to_idx[tid] for tid in cands]], dtype=torch.long, device=DEVICE)
c_mask = torch.ones((1, len(cands)), dtype=torch.float32, device=DEVICE)

print("\n--- FINE-GRAINED LAYER PROFILING ---")

with torch.inference_mode():
    q_mask = (torch.norm(q_ten, p=2, dim=-1) > 1e-5).float()
    c_mask_in = (torch.norm(c_ten, p=2, dim=-1) > 1e-5).float()
    cat_t_mask = (torch.norm(catalog_tensors, p=2, dim=-1) > 1e-5).float()
    q_mask[:, 0] = 1.0
    c_mask_in[:, 0] = 1.0
    cat_t_mask[:, 0] = 1.0

    t0 = time.perf_counter()
    q_in = model.input_adapter(q_ten)
    c_in = model.input_adapter(c_ten)
    t_adapt_qc = (time.perf_counter() - t0) * 1000
    print(f"Adapter (Q & C): {t_adapt_qc:.2f} ms")

    t0 = time.perf_counter()
    cat_t_in = model.input_adapter(catalog_tensors)
    t_adapt_cat = (time.perf_counter() - t0) * 1000
    print(f"Adapter (ALL 400 catalog tools): {t_adapt_cat:.2f} ms")

    t0 = time.perf_counter()
    q_tilde, c_tilde, q_pool, c_pool, lang_logits, lang_probs, lang_conf, lang_bias = model.capa1_idioma(
        q_in, c_in, q_mask, c_mask_in
    )
    t_capa1 = (time.perf_counter() - t0) * 1000
    print(f"Capa 1 Idioma: {t_capa1:.2f} ms")

    capa2 = model.capa2_contexto_moe

    t0 = time.perf_counter()
    t_emb, t_att_p, t_mask_4d = capa2._encode_tools_unified(
        tools_seq=cat_t_in, t_mask=cat_t_mask, B=1, cand_idxs=c_idxs, catalog_t_in=cat_t_in, catalog_t_mask=cat_t_mask
    )
    t_encode_tools = (time.perf_counter() - t0) * 1000
    print(f"Capa 2 _encode_tools_unified (ALL 400 tools SwiGLU + pooling): {t_encode_tools:.2f} ms")

    t0 = time.perf_counter()
    # Cross attention & Colbert stats
    # let's measure the colbert & pair fusion
    B, Lq, _ = q_tilde.shape
    Lc = c_tilde.size(1)
    scale = 1.0 / (capa2.w_q.out_features ** 0.5)
    attn_scores = (
        torch.matmul(capa2.w_q(q_tilde), capa2.w_k(c_tilde).transpose(-1, -2)).float() * scale
    ).masked_fill((c_mask_in < 0.5).unsqueeze(1), -1e4)
    v_c = capa2.w_v(c_tilde)
    m_qc = torch.matmul(F.softmax(attn_scores, dim=-1).to(v_c.dtype), v_c) * q_mask.unsqueeze(-1)
    m_pool = capa2.m_pooler(m_qc, q_mask)

    ctx_in = torch.cat(
        [q_pool, c_pool, m_pool, torch.abs(q_pool - c_pool), q_pool * m_pool, lang_probs.to(q_pool.dtype)], dim=-1
    )
    ctx_logits = capa2.context_choice_head(ctx_in).float()
    ctx_probs = F.softmax(ctx_logits, dim=-1)
    gate_alpha = capa2.gate_proj(
        torch.cat([ctx_probs.to(q_pool.dtype), lang_probs.to(q_pool.dtype), q_pool, m_pool], dim=-1)
    ).float()
    ctx_polarity = capa2.ctx_polarity_head(
        torch.cat([q_pool, c_pool, m_pool, q_pool - c_pool, q_pool * c_pool, ctx_probs.to(q_pool.dtype)], dim=-1)
    ).float()
    pos_ctx_gate = F.relu(ctx_polarity) * gate_alpha
    q_fused = (
        capa2.out_norm(q_tilde + lang_bias.unsqueeze(1) + 0.55 * pos_ctx_gate.to(q_tilde.dtype).unsqueeze(1) * m_qc)
        * q_mask.unsqueeze(-1)
    )
    q_emb_raw = F.normalize(capa2.shared_q_proj(q_fused), p=2, dim=-1)
    c_emb = F.normalize(capa2.shared_c_proj(c_tilde), p=2, dim=-1)
    c_att_p_single = F.normalize(capa2.c_att_pool(c_emb, c_mask_in), p=2, dim=-1)
    q_emb = capa2.ortho_rejection(q_emb_raw, c_att_p_single, ctx_polarity, q_mask)

    # sim_qt
    sim_qt = torch.einsum("bqd,bktd->bkqt", q_emb, t_emb).float()
    t_cross_colbert = (time.perf_counter() - t0) * 1000
    print(f"Capa 2 Context + Colbert Cross-Attn: {t_cross_colbert:.2f} ms")

    # Let's measure router and MoE Bank
    router_in = torch.cat([q_pool, c_pool, m_pool, lang_probs.to(q_pool.dtype), ctx_probs.to(q_pool.dtype)], dim=-1)
    router_logits = capa2.router(router_in).float()
    topk_vals, topk_idx = torch.topk(router_logits, k=capa2.top_k_experts, dim=-1)
    topk_weights = F.softmax(topk_vals, dim=-1)

    # Pair fusion to h_base
    # Mock h_base of shape [1, 36, 2048]
    h_base = torch.randn(1, 36, 2048, device=DEVICE)

    t0 = time.perf_counter()
    h_tools = capa2.moe_bank(h_base, topk_idx, topk_weights)
    t_moe = (time.perf_counter() - t0) * 1000
    print(f"Capa 2 MoE Bank (4 experts out of 128, dequantize + bmm): {t_moe:.2f} ms")

    t0 = time.perf_counter()
    x_set = h_tools
    for layer in capa2.set_layers:
        x_set = layer(x_set, cand_mask=c_mask)
    t_set = (time.perf_counter() - t0) * 1000
    print(f"Capa 2 SetAttention (8 layers on 36 candidates): {t_set:.2f} ms")

    t0 = time.perf_counter()
    h_refined = capa2.set_norm(h_tools + torch.tanh(capa2.set_gate) * capa2.set_branch_norm(x_set))
    t_heads = (time.perf_counter() - t0) * 1000
    print(f"Capa 2 Heads & Capa 3: {t_heads:.2f} ms")
