#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación Integral de REMTK-JUICIO-JEV-MOE 3.82B (A100 Checkpoint Zero Dup)
Modelo: C:\\Users\\LENOVO\\Desktop\\ULTIMA DE 2026\\remtk-prediction\\models\\remtk_juicio_jev_moe_3_82B_zero_dup_a100.pt
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
import onnxruntime as ort

DEVICE = torch.device("cpu")
BASE_DIR = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction")
MODEL_DIR = BASE_DIR / "models" / "remtk-juicio-jev-moe-3.8b-400tools"
CKPT_PATH = BASE_DIR / "models" / "remtk_juicio_jev_moe_3_82B_zero_dup_a100.pt"

CATALOG_META_PATH = MODEL_DIR / "catalog_metadata.json"
CATALOG_TENSORS_PATH = MODEL_DIR / "catalog_e5_tensors.safetensors"
E5_DIR = BASE_DIR / "models" / "multilingual-e5-small-onnx"

print(f"[Init] Cargando metadatos del catálogo desde {CATALOG_META_PATH}...")
with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

ALL_TOOL_IDS = META["all_tool_ids"]
CATALOG = META["catalog"]
tool_id_to_idx = {tid: idx for idx, tid in enumerate(ALL_TOOL_IDS)}
print(f"[Init] Total herramientas en catálogo: {len(ALL_TOOL_IDS)}")

print(f"[Init] Cargando tensores precalculados del catálogo desde {CATALOG_TENSORS_PATH}...")
cat_safetensors = load_file(str(CATALOG_TENSORS_PATH))
raw_tool_tensors = cat_safetensors["catalog_tool_tensors"].to(dtype=torch.float32) # [400, 64, 384]
# Pooled catalog representation
cat_mask = (torch.norm(raw_tool_tensors, p=2, dim=-1) > 1e-5).float()
cat_lens = cat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
catalog_pooled = F.normalize((raw_tool_tensors * cat_mask.unsqueeze(-1)).sum(dim=1) / cat_lens, p=2, dim=-1) # [400, 384]

print(f"[Init] Inicializando Multilingual-E5-Small ONNX desde {E5_DIR}...")
e5_tok = AutoTokenizer.from_pretrained(str(E5_DIR))
e5_session = ort.InferenceSession(str(E5_DIR / "model.onnx"))

def encode_text_tokens(texts: List[str], prefix: str = "query: ", max_len: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    prefixed = [f"{prefix}{t.strip()}" if not t.strip().startswith(prefix) else t.strip() for t in texts]
    toks = e5_tok(prefixed, max_length=max_len, padding="max_length", truncation=True, return_tensors="np")
    inp = {
        "input_ids": toks["input_ids"].astype(np.int64),
        "attention_mask": toks["attention_mask"].astype(np.int64),
        "token_type_ids": (toks["token_type_ids"] if "token_type_ids" in toks else np.zeros_like(toks["input_ids"])).astype(np.int64),
    }
    out = e5_session.run(None, inp)[0]
    out_t = torch.from_numpy(out).to(DEVICE, dtype=torch.float32)
    mask = torch.from_numpy(toks["attention_mask"]).to(DEVICE, dtype=torch.float32).unsqueeze(-1)
    normed_tokens = F.normalize(out_t, p=2, dim=-1) * mask
    lens = mask.sum(dim=1).clamp(min=1.0)
    pooled = F.normalize((normed_tokens * mask).sum(dim=1) / lens, p=2, dim=-1)
    return normed_tokens, pooled

# ==============================================================================
# Definición de Arquitectura Exacta Coupled JEV MoE
# ==============================================================================
class SwiGLUProjector(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.05):
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_hidden)
        self.w_up = nn.Linear(d_in, d_hidden)
        self.w_down = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.norm(self.w_down(self.dropout(h)))

class CandidateDifferentialInjector(nn.Module):
    def __init__(self, d_embed: int, d_moe: int, dropout: float = 0.05):
        super().__init__()
        self.proj = SwiGLUProjector(d_embed * 4, d_moe, d_moe, dropout)

    def forward(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, K, D = t.shape
        q_exp = q.unsqueeze(1).expand(-1, K, -1)
        diff = torch.cat([q_exp, t, q_exp - t, q_exp * t], dim=-1)
        return self.proj(diff)

class DilatedGatedConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv_feat = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(out_c)
        self.proj_res = nn.Linear(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj_res(x)
        xt = x.transpose(1, 2)
        f = self.conv_feat(xt).transpose(1, 2)
        g = torch.sigmoid(self.conv_gate(xt).transpose(1, 2))
        return self.norm(res + f * g)

class AttentivePooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x.mean(dim=1, keepdim=True))
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v).squeeze(1)

class Capa1JevIdiomaChoice(nn.Module):
    def __init__(self, d_embed: int = 384, d_c1: int = 1024):
        super().__init__()
        self.conv1 = DilatedGatedConvBlock(d_embed, 512, 3, 1)
        self.conv2 = DilatedGatedConvBlock(512, d_c1, 3, 2)
        self.pool = AttentivePooling(d_c1)
        self.ctx_proj = nn.Linear(d_embed, d_c1)
        self.fusion_norm = nn.LayerNorm(d_c1)
        self.lang_head = nn.Linear(d_c1, 2)

    def forward(self, q_tokens: torch.Tensor, c_global: torch.Tensor):
        h = self.conv1(q_tokens)
        h = self.conv2(h)
        pooled = self.pool(h)
        ctx = self.ctx_proj(c_global)
        fused = self.fusion_norm(pooled + ctx)
        lang = self.lang_head(fused)
        return fused, lang

class Fase1Thinker(nn.Module):
    def __init__(self, d_embed: int = 384):
        super().__init__()
        self.discourse_head = nn.Sequential(
            nn.Linear(2305, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(768, 384),
            nn.SiLU(),
            nn.Linear(384, 5)
        )
        self.offer_cross_attn = nn.MultiheadAttention(
            embed_dim=d_embed, num_heads=6, dropout=0.05, batch_first=True
        )
        self.norm_pooled = nn.LayerNorm(d_embed)
        self.norm_tokens = nn.LayerNorm(d_embed)

class Fase2Redundancy(nn.Module):
    def __init__(self, d_embed: int = 384):
        super().__init__()
        self.gamma_mut = nn.Parameter(torch.tensor(2.0))
        self.redundancy_mlp = nn.Sequential(
            nn.Linear(1921, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )

class ColBERTSubwordLateInteraction(nn.Module):
    def __init__(self, d_embed: int = 384, d_proj: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_proj, bias=False)
        self.t_proj = nn.Linear(d_embed, d_proj, bias=False)

class PreLNSetAttentionLayer(nn.Module):
    def __init__(self, d_moe: int = 2048, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_moe)
        self.attn = nn.MultiheadAttention(d_moe, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_moe)
        self.ffn = nn.Sequential(
            nn.Linear(d_moe, d_moe * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_moe * 2, d_moe)
        )

class UltraSparseVectorizedMoEBank(nn.Module):
    def __init__(self, d_moe: int = 2048, d_hidden: int = 4528, num_experts: int = 128, top_k: int = 4):
        super().__init__()
        self.d_moe = d_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_moe)
        self.router = nn.Linear(d_moe, num_experts, bias=False)
        self.w_gate = nn.Parameter(torch.empty(num_experts, d_moe, d_hidden, dtype=torch.bfloat16))
        self.w_up = nn.Parameter(torch.empty(num_experts, d_moe, d_hidden, dtype=torch.bfloat16))
        self.w_down = nn.Parameter(torch.empty(num_experts, d_hidden, d_moe, dtype=torch.bfloat16))

class JudgmentCore(nn.Module):
    def __init__(self, d_in: int = 7176, d_hidden: int = 2048, d_out: int = 1024):
        super().__init__()
        self.judgment_net = SwiGLUProjector(d_in, d_hidden, d_out)
        self.judgment_head = nn.Linear(d_out, 4)

class RemtkJuicioJevMoECoupledCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        d_embed = 384
        d_layer1 = 1024
        d_moe = 2048
        d_expert_hidden = 4528

        self.noul_sink_token = nn.Parameter(torch.randn(1, 1, d_moe))
        self.colbert_weight = nn.Parameter(torch.tensor(2.5))

        self.fase1_thinker = Fase1Thinker(d_embed)
        self.capa1 = Capa1JevIdiomaChoice(d_embed, d_layer1)
        self.fase2_redundancy = Fase2Redundancy(d_embed)

        self.colbert = ColBERTSubwordLateInteraction(d_embed, 128)
        self.diff_injector = CandidateDifferentialInjector(d_embed, d_moe)
        self.h1_to_moe = nn.Linear(d_layer1, d_moe)
        self.sink_context_proj = nn.Linear(6, d_moe)

        self.set_attn_layers = nn.ModuleList([
            PreLNSetAttentionLayer(d_moe, num_heads=8, dropout=0.05)
            for _ in range(6)
        ])

        self.moe_bank = UltraSparseVectorizedMoEBank(d_moe, d_expert_hidden, num_experts=128, top_k=4)

        self.candidate_logit_head = nn.Linear(d_moe, 1)
        self.sink_logit_head = nn.Linear(d_moe, 1)

        self.judgment_core = JudgmentCore(7176, 2048, 1024)

        self.noul_intent_head = nn.Sequential(
            nn.Linear(1034, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        self.noul_coverage_head = nn.Sequential(
            nn.Linear(3074, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        self.route_head = nn.Sequential(
            nn.Linear(1037, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 4)
        )

    def forward(
        self,
        q_tokens: torch.Tensor,
        q_pooled: torch.Tensor,
        a_offer_tokens: torch.Tensor,
        a_offer_pooled: torch.Tensor,
        c_global: torch.Tensor,
        ledger_queries: torch.Tensor,
        ledger_tools: torch.Tensor,
        ledger_is_mutable: torch.Tensor,
        ledger_mask: torch.Tensor,
        rejected_vec: torch.Tensor,
        has_rejection: torch.Tensor,
        t_tokens: torch.Tensor,
        t_pooled: torch.Tensor
    ):
        B = q_tokens.size(0)
        K = t_pooled.size(1)
        M = ledger_queries.size(1)

        # 1. Fase 1: Discurso y Anáfora
        q_diff = q_pooled - a_offer_pooled
        q_prod = q_pooled * a_offer_pooled
        disc_in = torch.cat([q_pooled, a_offer_pooled, q_diff, q_prod, c_global, rejected_vec, has_rejection], dim=-1)
        disc_logits = self.fase1_thinker.discourse_head(disc_in)
        disc_probs = F.softmax(disc_logits, dim=-1)

        p_cont = disc_probs[:, 1:2] + disc_probs[:, 2:3]
        p_cancel = disc_probs[:, 3:4]
        p_modify = disc_probs[:, 4:5]

        attended_offer, _ = self.fase1_thinker.offer_cross_attn(
            query=q_tokens, key=a_offer_tokens, value=a_offer_tokens
        )

        dot = (q_pooled * rejected_vec).sum(dim=-1, keepdim=True)
        norm_sq = (rejected_vec * rejected_vec).sum(dim=-1, keepdim=True) + 1e-6
        proj = (dot / norm_sq) * rejected_vec
        q_orth = q_pooled - (p_modify * has_rejection) * proj

        q_tokens_res = self.fase1_thinker.norm_tokens(
            (1.0 - p_cancel.unsqueeze(-1)) * (q_tokens + p_cont.unsqueeze(-1) * attended_offer)
        )
        q_pooled_res = self.fase1_thinker.norm_pooled(
            (1.0 - p_cancel) * (q_orth + p_cont * attended_offer.mean(dim=1))
        )

        # 2. Capa 1: Proyección Lingüística
        h1, lang_logits = self.capa1(q_tokens_res, c_global)

        # 3. Fase 2: Redundancia y Ledger
        q_clean_exp = q_pooled_res.unsqueeze(1).expand(-1, M, -1)
        diff_ledger = torch.abs(q_clean_exp - ledger_queries)
        prod_ledger = q_clean_exp * ledger_queries
        red_in = torch.cat([
            q_clean_exp, ledger_queries, diff_ledger, prod_ledger, ledger_tools, ledger_is_mutable.unsqueeze(-1)
        ], dim=-1)
        red_logits = self.fase2_redundancy.redundancy_mlp(red_in).squeeze(-1)
        red_logits = red_logits - self.fase2_redundancy.gamma_mut * ledger_is_mutable
        red_logits = red_logits.masked_fill(ledger_mask == 0, -1e4)
        has_ledger = (ledger_mask.sum(dim=-1, keepdim=True) > 0).float()
        redundancy_prob = has_ledger * torch.sigmoid(red_logits.max(dim=-1, keepdim=True).values)

        # 4. Fase 3: ColBERT Late Interaction + MoE con [NOUL_SINK]
        q_proj = F.normalize(self.colbert.q_proj(q_tokens_res), p=2, dim=-1) # [B, L_q, 128]
        t_proj = F.normalize(self.colbert.t_proj(t_tokens), p=2, dim=-1)     # [B, K, L_t, 128]
        sim = torch.einsum('b q d, b k t d -> b k q t', q_proj, t_proj)
        colbert_scores = sim.max(dim=-1).values.mean(dim=-1) # [B, K]

        cand_moe_in = self.diff_injector(q_pooled_res, t_pooled) # [B, K, 2048]
        h1_proj = self.h1_to_moe(h1).unsqueeze(1)                # [B, 1, 2048]
        cand_moe_in = cand_moe_in + h1_proj

        sink_ctx_in = torch.cat([disc_probs, redundancy_prob], dim=-1) # [B, 6]
        sink_ctx = self.sink_context_proj(sink_ctx_in).unsqueeze(1)    # [B, 1, 2048]
        sink_token = self.noul_sink_token.expand(B, 1, -1) + h1_proj + sink_ctx # [B, 1, 2048]

        set_seq = torch.cat([cand_moe_in, sink_token], dim=1) # [B, K + 1, 2048]

        for layer in self.set_attn_layers:
            res = set_seq
            normed = layer.norm1(set_seq)
            attn_out, _ = layer.attn(normed, normed, normed, need_weights=False)
            set_seq = res + attn_out
            res = set_seq
            normed = layer.norm2(set_seq)
            set_seq = res + layer.ffn(normed)

        # MoE Top-4 Routing
        moe_res = set_seq
        normed = self.moe_bank.norm(set_seq)
        S = set_seq.size(1)
        D = set_seq.size(2)
        x_flat = normed.reshape(-1, D)
        router_logits = self.moe_bank.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_vals, top_idx = torch.topk(router_probs, k=self.moe_bank.top_k, dim=-1)
        top_weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-6)

        out_flat = torch.zeros_like(x_flat)
        for k_slot in range(self.moe_bank.top_k):
            exp_ids = top_idx[:, k_slot]
            weight_k = top_weights[:, k_slot].unsqueeze(-1)
            wg = self.moe_bank.w_gate[exp_ids].float()
            wu = self.moe_bank.w_up[exp_ids].float()
            wd = self.moe_bank.w_down[exp_ids].float()
            x_uns = x_flat.unsqueeze(1)
            hidden = F.silu(torch.bmm(x_uns, wg)) * torch.bmm(x_uns, wu)
            step_out = torch.bmm(hidden, wd).squeeze(1)
            out_flat = out_flat + weight_k * step_out

        moe_seq = moe_res + out_flat.reshape(B, S, D)

        cand_moe_out = moe_seq[:, :-1, :]
        sink_moe_out = moe_seq[:, -1, :]

        raw_cand_logits = self.candidate_logit_head(cand_moe_out).squeeze(-1) + self.colbert_weight * colbert_scores
        sink_logit = self.sink_logit_head(sink_moe_out)

        all_logits = torch.cat([raw_cand_logits, sink_logit], dim=-1)
        all_probs = F.softmax(all_logits, dim=-1)
        p_sink = all_probs[:, -1:]

        # 5. Fase 4: Judgment Core y Gating Acoplado
        cand_max_pool, _ = cand_moe_out.max(dim=1)
        cand_mean_pool = cand_moe_out.mean(dim=1)
        colbert_max, _ = colbert_scores.max(dim=-1, keepdim=True)

        j_in = torch.cat([
            cand_max_pool, cand_mean_pool, sink_moe_out, h1, disc_probs, redundancy_prob, p_sink, colbert_max
        ], dim=-1)
        judgment_h = self.judgment_core.judgment_net(j_in)
        judgment_probs = F.softmax(self.judgment_core.judgment_head(judgment_h), dim=-1)

        noul_intent_in = torch.cat([judgment_h, disc_probs, redundancy_prob, judgment_probs], dim=-1)
        p_noul_intent = torch.sigmoid(self.noul_intent_head(noul_intent_in))

        noul_cov_in = torch.cat([judgment_h, sink_moe_out, p_sink, colbert_max], dim=-1)
        p_noul_coverage = torch.sigmoid(self.noul_coverage_head(noul_cov_in)) * (1.0 - p_sink)

        route_in = torch.cat([judgment_h, disc_probs, redundancy_prob, judgment_probs, p_sink, p_noul_intent, p_noul_coverage], dim=-1)
        route_logits = self.route_head(route_in)
        route_probs = F.softmax(route_logits, dim=-1)
        p_route_tool = route_probs[:, 0:1]

        raw_affinities = torch.sigmoid(raw_cand_logits)
        master_gate = p_route_tool * (1.0 - redundancy_prob) * p_noul_intent * p_noul_coverage
        gated_tool_scores = master_gate * raw_affinities

        return {
            "gated_tool_scores": gated_tool_scores,
            "master_gate": master_gate,
            "route_probs": route_probs,
            "judgment_probs": judgment_probs,
            "disc_probs": disc_probs,
            "p_sink": p_sink,
            "redundancy_prob": redundancy_prob,
            "p_noul_intent": p_noul_intent,
            "p_noul_coverage": p_noul_coverage,
            "raw_affinities": raw_affinities,
            "raw_cand_logits": raw_cand_logits,
            "colbert_scores": colbert_scores
        }

# ==============================================================================
# Carga de Modelo y Motor de Inferencia
# ==============================================================================
print(f"[Init] Instanciando RemtkJuicioJevMoECoupledCheckpointModel...")
with torch.device("meta"):
    model = RemtkJuicioJevMoECoupledCheckpointModel()

print(f"[Init] Cargando checkpoint mmap desde {CKPT_PATH}...")
ckpt = torch.load(str(CKPT_PATH), map_location="cpu", mmap=True)
model.load_state_dict(ckpt["model_state_dict"], strict=True, assign=True)
model.eval()
print("[Init] Modelo REMTK-JUICIO-JEV-MOE 3.82B Zero Dup A100 listo para inferencia!")

DISCOURSE_LABELS = [
    "EXPLICIT_NEW",
    "AFFIRM_CONT",
    "SELECT_CONT",
    "CANCEL_REJECT",
    "MODIFY_CORRECT"
]
ROUTE_LABELS = ["TOOL", "MEMORY", "PREV_RESPONSE", "INDEPENDENT"]

def select_candidate_tools(query_pooled: torch.Tensor, k: int = 32, forced_cands: Optional[List[str]] = None) -> List[str]:
    sims = torch.matmul(query_pooled, catalog_pooled.T)[0]
    top_indices = torch.topk(sims, k=k).indices.tolist()
    cands = [ALL_TOOL_IDS[idx] for idx in top_indices]
    if forced_cands:
        for fc in reversed(forced_cands):
            if fc in tool_id_to_idx and fc not in cands:
                cands.pop()
                cands.insert(0, fc)
    return cands[:k]

def run_coupled_inference(
    query: str,
    assistant_offer: str = "",
    global_context: str = "",
    ledger_history: Optional[List[Dict[str, Any]]] = None,
    rejected_intent_text: str = "",
    forced_candidates: Optional[List[str]] = None,
    filter_out_tool: Optional[str] = None
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # 1. Encode query & assistant offer
    q_tokens, q_pooled = encode_text_tokens([query], prefix="query: ", max_len=24)
    if assistant_offer.strip():
        a_tokens, a_pooled = encode_text_tokens([assistant_offer], prefix="passage: ", max_len=24)
    else:
        a_tokens = torch.zeros(1, 24, 384)
        a_pooled = torch.zeros(1, 384)

    if global_context.strip():
        _, c_global = encode_text_tokens([global_context], prefix="passage: ", max_len=24)
    else:
        c_global = torch.zeros(1, 384)

    # 2. Rejection vector
    if rejected_intent_text.strip():
        _, rej_vec = encode_text_tokens([rejected_intent_text], prefix="query: ", max_len=24)
        has_rej = torch.tensor([[1.0]])
    else:
        rej_vec = torch.zeros(1, 384)
        has_rej = torch.tensor([[0.0]])

    # 3. Ledger (M=8 slots)
    M = 8
    ledger_queries = torch.zeros(1, M, 384)
    ledger_tools = torch.zeros(1, M, 384)
    ledger_is_mutable = torch.zeros(1, M)
    ledger_mask = torch.zeros(1, M)

    if ledger_history:
        for slot_idx, item in enumerate(ledger_history[:M]):
            q_hist = item.get("query", "")
            t_id = item.get("tool_id", "")
            is_mut = 1.0 if item.get("is_mutable", False) else 0.0
            if q_hist:
                _, q_h_p = encode_text_tokens([q_hist], prefix="query: ", max_len=24)
                ledger_queries[0, slot_idx] = q_h_p[0]
            if t_id and t_id in tool_id_to_idx:
                ledger_tools[0, slot_idx] = catalog_pooled[tool_id_to_idx[t_id]]
            ledger_is_mutable[0, slot_idx] = is_mut
            ledger_mask[0, slot_idx] = 1.0

    # 4. Candidate tools (K=32)
    # If assistant offer has content and is selective/continuation, blend intent
    search_pooled = q_pooled
    if assistant_offer.strip():
        search_pooled = F.normalize(0.40 * q_pooled + 0.60 * a_pooled, p=2, dim=-1)

    cands = select_candidate_tools(search_pooled, k=32, forced_cands=forced_candidates)
    if filter_out_tool and filter_out_tool in cands:
        cands = [c for c in cands if c != filter_out_tool]
        # pad back to 32
        for tid in ALL_TOOL_IDS:
            if tid not in cands and tid != filter_out_tool:
                cands.append(tid)
                if len(cands) == 32:
                    break

    cand_indices = [tool_id_to_idx[c] for c in cands]
    t_tokens = raw_tool_tensors[cand_indices, :32, :].unsqueeze(0) # [1, 32, 32, 384]
    t_pooled = catalog_pooled[cand_indices].unsqueeze(0)           # [1, 32, 384]

    # 5. Forward Pass
    with torch.no_grad():
        out = model(
            q_tokens, q_pooled, a_tokens, a_pooled, c_global,
            ledger_queries, ledger_tools, ledger_is_mutable, ledger_mask,
            rej_vec, has_rej, t_tokens, t_pooled
        )

    t_latency = (time.perf_counter() - t0) * 1000

    disc_probs = out["disc_probs"][0].tolist()
    disc_best_idx = int(np.argmax(disc_probs))
    route_probs = out["route_probs"][0].tolist()
    route_best_idx = int(np.argmax(route_probs))

    gated_scores = out["gated_tool_scores"][0].tolist()
    raw_affinities = out["raw_affinities"][0].tolist()
    colbert_scores = out["colbert_scores"][0].tolist()

    ranked_indices = np.argsort(gated_scores)[::-1]
    top_tools = [
        {
            "tool": cands[idx],
            "gated_score": round(gated_scores[idx], 4),
            "raw_affinity": round(raw_affinities[idx], 4),
            "colbert": round(colbert_scores[idx], 4)
        }
        for idx in ranked_indices[:3]
    ]

    return {
        "query": query,
        "latency_ms": round(t_latency, 1),
        "discourse": {
            "class": DISCOURSE_LABELS[disc_best_idx],
            "prob": round(disc_probs[disc_best_idx], 4),
            "all": {DISCOURSE_LABELS[i]: round(disc_probs[i], 4) for i in range(5)}
        },
        "route": {
            "route": ROUTE_LABELS[route_best_idx],
            "prob": round(route_probs[route_best_idx], 4),
            "all": {ROUTE_LABELS[i]: round(route_probs[i], 4) for i in range(4)}
        },
        "gates": {
            "master_gate": round(out["master_gate"].item(), 4),
            "p_noul_intent": round(out["p_noul_intent"].item(), 4),
            "p_noul_coverage": round(out["p_noul_coverage"].item(), 4),
            "p_sink": round(out["p_sink"].item(), 4),
            "redundancy_prob": round(out["redundancy_prob"].item(), 4),
        },
        "candidates": cands,
        "top_tools": top_tools
    }

# ==============================================================================
# Suite de Pruebas de Validación
# ==============================================================================
TEST_SUITE = [
    # 1. Comandos directos clave
    {
        "id": "T1_AGENDA",
        "category": "Comando Directo",
        "query": "Quiero agendar una cita con el doctor para este viernes a las 4pm.",
        "expected_tool": "agenda_book_appointment",
        "expected_route": "TOOL"
    },
    {
        "id": "T2_SQL",
        "category": "Comando Directo",
        "query": "Ejecuta una consulta SELECT sobre la tabla de clientes para listar activos.",
        "expected_tool": "db_execute_sql",
        "expected_route": "TOOL"
    },
    {
        "id": "T3_WHATSAPP",
        "category": "Comando Directo",
        "query": "Envía un mensaje de texto por WhatsApp al cliente informando su pedido.",
        "expected_tool": "ch_whatsapp_twilio",
        "expected_route": "TOOL"
    },
    {
        "id": "T4_K8S",
        "category": "Comando Directo",
        "query": "Despliega el nuevo pod del microservicio en el clúster de Kubernetes.",
        "expected_tool": "k8s_deploy_pod",
        "expected_route": "TOOL"
    },
    # 2. Caso Usuario: Anáfora y Continuación ("la carta embrujada")
    {
        "id": "T5_ANAPHORA_HALLOWEEN",
        "category": "Anáfora / Continuación",
        "query": "la carta embrujada",
        "assistant_offer": "¿Quieres que te la deje lista como página HTML imprimible (una sola hoja, con la paleta Halloween) o como PDF maquetado? 🎃",
        "expected_discourse": "SELECT_CONT",
        "expected_route": "TOOL"
    },
    # 3. Caso Usuario: Typo severo + Ausencia de herramienta en sesión ("recordatorioow")
    {
        "id": "T6_TYPO_SINK_MISSING",
        "category": "Typo + Ausencia de Herramienta ([NOUL_SINK])",
        "query": "puedes mandarme como recordatorioow para mañana a las 8AM",
        "filter_out_tool": "agenda_book_appointment", # Simulamos sesión sin scheduler
        "expect_high_sink": True
    },
    # 4. Caso Usuario: Typo severo pero con herramienta en el catálogo
    {
        "id": "T7_TYPO_WITH_TOOL",
        "category": "Typo + Presencia de Herramienta",
        "query": "puedes mandarme como recordatorioow para mañana a las 8AM",
        "forced_candidates": ["alert_send_slack_channel", "ch_whatsapp_twilio", "agenda_book_appointment"],
        "expect_tool_candidate": True
    },
    # 5. Redundancia de Sesión (Ledger con lectura previa)
    {
        "id": "T8_REDUNDANCY_SUPPRESSION",
        "category": "Redundancia de Sesión",
        "query": "dame los precios de la carta de platos y cócteles",
        "ledger_history": [
            {"query": "dame los precios de la carta de platos y cócteles", "tool_id": "knowledge_search", "is_mutable": False}
        ],
        "expect_high_redundancy": True
    },
    # 6. Cancelación / Rechazo explícito
    {
        "id": "T9_CANCELLATION",
        "category": "Cancelación / Rechazo",
        "query": "no nada, olvídalo",
        "expected_discourse": "CANCEL_REJECT",
        "expected_route": "INDEPENDENT"
    },
    # 7. Charla social / Factual
    {
        "id": "T10_CHITCHAT",
        "category": "Charla Factual",
        "query": "Hola buenos días, ¿cómo estás hoy?",
        "expected_route": "INDEPENDENT"
    }
]

print("\n" + "=" * 80)
print("INICIANDO EJECUCIÓN DE PRUEBAS SOBRE REMTK-JUICIO-JEV-MOE 3.82B")
print("=" * 80)

results = []
for t in TEST_SUITE:
    print(f"\n[Test {t['id']}] Categoría: {t['category']}")
    print(f"  Query: \"{t['query']}\"")
    if t.get("assistant_offer"):
        print(f"  Oferta Asistente: \"{t['assistant_offer']}\"")
    if t.get("ledger_history"):
        print(f"  Ledger Prev: {len(t['ledger_history'])} ejecuciones")

    res = run_coupled_inference(
        query=t["query"],
        assistant_offer=t.get("assistant_offer", ""),
        ledger_history=t.get("ledger_history"),
        forced_candidates=t.get("forced_candidates"),
        filter_out_tool=t.get("filter_out_tool")
    )
    res["test_id"] = t["id"]
    res["category"] = t["category"]
    results.append(res)

    print(f"  ⏱️  Latencia: {res['latency_ms']} ms")
    print(f"  💬 Discurso: {res['discourse']['class']} (P={res['discourse']['prob']})")
    print(f"  🛣️  Ruta: {res['route']['route']} (P={res['route']['prob']})")
    print(f"  🛡️  Compuertas -> MasterGate: {res['gates']['master_gate']} | Intent: {res['gates']['p_noul_intent']} | Cov: {res['gates']['p_noul_coverage']} | Sink: {res['gates']['p_sink']} | Red: {res['gates']['redundancy_prob']}")
    print(f"  🎯 Top 3 Herramientas:")
    for idx, th in enumerate(res["top_tools"]):
        print(f"     #{idx+1}: {th['tool']:32s} | Gated: {th['gated_score']:.4f} | Raw: {th['raw_affinity']:.4f} | ColBERT: {th['colbert']:.4f}")

out_path = BASE_DIR / "scripts" / "results_coupled_3_82b_a100.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[Fin] Resultados guardados en {out_path}")
