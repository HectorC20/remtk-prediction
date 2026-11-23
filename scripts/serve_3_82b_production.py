#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REMTK-PREDICTION PRODUCTION SERVER (Coupled JEV MoE 3.82B Zero Dup A100)
Puertos: 6776 y 6876
Sirve el nuevo checkpoint entrenado en A100 para key-remtk en tiempo real.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aiohttp import web
from safetensors.torch import load_file
from transformers import AutoTokenizer
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_DIR = BASE_DIR / "models" / "remtk-juicio-jev-moe-3.8b-400tools"
CKPT_PATH = BASE_DIR / "models" / "remtk_juicio_jev_moe_3_82B_zero_dup_a100.pt"

CATALOG_META_PATH = MODEL_DIR / "catalog_metadata.json"
CATALOG_TENSORS_PATH = MODEL_DIR / "catalog_e5_tensors.safetensors"
E5_DIR = BASE_DIR / "models" / "multilingual-e5-small-onnx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [serve-3.82b] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("remtk_prediction_3_82b")

# ==============================================================================
# Carga de Catálogo y E5
# ==============================================================================
logger.info(f"Cargando catálogo desde {CATALOG_META_PATH}...")
with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

ALL_TOOL_IDS = META["all_tool_ids"]
CATALOG = META["catalog"]
tool_id_to_idx = {tid: idx for idx, tid in enumerate(ALL_TOOL_IDS)}
logger.info(f"Catálogo cargado: {len(ALL_TOOL_IDS)} herramientas indexadas.")

logger.info(f"Cargando tensores E5 de herramientas desde {CATALOG_TENSORS_PATH}...")
cat_safetensors = load_file(str(CATALOG_TENSORS_PATH))
raw_tool_tensors = cat_safetensors["catalog_tool_tensors"].to(dtype=torch.float32)

cat_mask = (torch.norm(raw_tool_tensors, p=2, dim=-1) > 1e-5).float()
cat_lens = cat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
catalog_pooled = F.normalize((raw_tool_tensors * cat_mask.unsqueeze(-1)).sum(dim=1) / cat_lens, p=2, dim=-1)

logger.info(f"Inicializando Multilingual-E5-Small ONNX desde {E5_DIR}...")
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
    out_t = torch.from_numpy(out).to(torch.float32)
    mask = torch.from_numpy(toks["attention_mask"]).to(torch.float32).unsqueeze(-1)
    normed_tokens = F.normalize(out_t, p=2, dim=-1) * mask
    lens = mask.sum(dim=1).clamp(min=1.0)
    pooled = F.normalize((normed_tokens * mask).sum(dim=1) / lens, p=2, dim=-1)
    return normed_tokens, pooled

# ==============================================================================
# Definición de Arquitectura Exacta
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
        q_proj = F.normalize(self.colbert.q_proj(q_tokens_res), p=2, dim=-1)
        t_proj = F.normalize(self.colbert.t_proj(t_tokens), p=2, dim=-1)
        sim = torch.einsum('b q d, b k t d -> b k q t', q_proj, t_proj)
        colbert_scores = sim.max(dim=-1).values.mean(dim=-1)

        cand_moe_in = self.diff_injector(q_pooled_res, t_pooled)
        h1_proj = self.h1_to_moe(h1).unsqueeze(1)
        cand_moe_in = cand_moe_in + h1_proj

        sink_ctx_in = torch.cat([disc_probs, redundancy_prob], dim=-1)
        sink_ctx = self.sink_context_proj(sink_ctx_in).unsqueeze(1)
        sink_token = self.noul_sink_token.expand(B, 1, -1) + h1_proj + sink_ctx

        set_seq = torch.cat([cand_moe_in, sink_token], dim=1)

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

        # 5. Fase 4 Judgment Core y Gating
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
            "disc_probs": disc_probs,
            "p_sink": p_sink,
            "redundancy_prob": redundancy_prob,
            "p_noul_intent": p_noul_intent,
            "p_noul_coverage": p_noul_coverage,
            "raw_affinities": raw_affinities,
            "colbert_scores": colbert_scores
        }

logger.info(f"Instanciando RemtkJuicioJevMoECoupledCheckpointModel...")
with torch.device("meta"):
    model = RemtkJuicioJevMoECoupledCheckpointModel()

logger.info(f"Cargando checkpoint mmap desde {CKPT_PATH}...")
ckpt = torch.load(str(CKPT_PATH), map_location="cpu", mmap=True)
model.load_state_dict(ckpt["model_state_dict"], strict=True, assign=True)
model.eval()
logger.info("Motor REMTK-JUICIO-JEV-MOE 3.82B Zero Dup A100 listo y operativo!")


# ==============================================================================
# Tenant-Scoped Tool Isolation Registry (Multitenant Architecture)
# ==============================================================================
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
                desc = t.get("description", "")
                intent = t.get("intentSummary", "")
                tags = " ".join(t.get("tags") or [])
                passage = f"passage: {clean_id} ({t.get('group', 'general')}): {desc}. {intent}. {tags}".strip()
                t_ten, t_pol = encode_text_tokens([passage], prefix="passage: ", max_len=64)
                tool_tensors_list.append(t_ten[0])
                tool_pooled_list.append(t_pol[0])

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
        logger.info(f"[TenantRegistry] Scope '{scope_key}' registrado con {len(tool_ids)} herramientas activas aisladas.")
        return len(tool_ids)

    def get(self, scope_key: str) -> Optional[Dict[str, Any]]:
        if scope_key in self.tenants:
            return self.tenants[scope_key]
        clean_tenant = scope_key.split("::")[0].strip()
        return self.tenants.get(f"{clean_tenant}::") or self.tenants.get(clean_tenant)

TENANT_REGISTRY = TenantRegistry()

def extract_scope_key(request: web.Request, body: Optional[Dict[str, Any]] = None) -> str:
    b = body or {}
    tenant = str(b.get("tenant") or request.query.get("tenant") or request.headers.get("key-remtk") or request.headers.get("x-remtk-key") or "default").strip()
    agent_id = str(b.get("agentId") or request.query.get("agentId") or "").strip()
    if "::" in tenant and not agent_id:
        return tenant
    if agent_id:
        clean_tenant = tenant.split("::")[0].strip()
        return f"{clean_tenant}::{agent_id}"
    return f"{tenant}::"

# Helper de Selección de Candidatas
def select_candidates(
    search_pooled: torch.Tensor,
    pool_pooled: torch.Tensor,
    pool_ids: List[str],
    k: int = 32,
    forced_cands: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    k_actual = min(k, len(pool_ids))
    if k_actual == 0:
        return [], []
    sims = torch.matmul(search_pooled, pool_pooled.T)[0]
    top_indices = torch.topk(sims, k=k_actual).indices.tolist()
    cands = [pool_ids[idx] for idx in top_indices]
    if forced_cands:
        for fc in reversed(forced_cands):
            if fc in pool_ids and fc not in cands:
                cands.pop()
                top_indices.pop()
                cands.insert(0, fc)
                top_indices.insert(0, pool_ids.index(fc))
    return cands, top_indices

# ==============================================================================
# Handlers HTTP
# ==============================================================================
async def health_handler(request: web.Request) -> web.Response:
    scope_key = extract_scope_key(request)
    cat = TENANT_REGISTRY.get(scope_key)
    count = len(cat["tool_ids"]) if cat else len(CATALOG)
    return web.json_response({
        "status": "ok",
        "model": "remtk-juicio-jev-moe-3.82b-coupled-a100",
        "tools_count": count,
        "global_catalog_size": len(CATALOG),
        "tenant_isolated": cat is not None,
        "onnx_runtime_version": "1.20.1",
        "active_device": "cpu",
    })

async def tools_count_handler(request: web.Request) -> web.Response:
    scope_key = extract_scope_key(request)
    cat = TENANT_REGISTRY.get(scope_key)
    count = len(cat["tool_ids"]) if cat else len(CATALOG)
    return web.json_response({"count": count})

async def tools_ready_handler(request: web.Request) -> web.Response:
    scope_key = extract_scope_key(request)
    cat = TENANT_REGISTRY.get(scope_key)
    count = len(cat["tool_ids"]) if cat else len(CATALOG)
    ready = cat["ready"] if cat else True
    return web.json_response({"ready": ready, "count": count})

async def register_tools_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    scope_key = extract_scope_key(request, body)
    tools_list = body.get("tools") or []
    indexed = TENANT_REGISTRY.register(scope_key, tools_list)
    return web.json_response({"indexed": indexed})

async def predict_feedback_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "events": 1})

async def memory_predict_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "memories": [],
        "topicShift": False,
        "topicScore": 0.0,
        "modelSize": "3.82b-coupled-a100",
        "rankedScores": [],
    })

async def memory_judge_handler(request: web.Request) -> web.Response:
    return web.json_response({"isNoise": False, "noiseScore": 0.0})

async def history_regime_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    history = body.get("history") or []
    n = len(history)
    selected_indices = list(range(n))
    return web.json_response({
        "regime": "PRESERVE_ALL" if n <= 4 else "PRESERVE_RECENT",
        "topicShift": 0.0,
        "anaphoraDetected": False,
        "dependencyScore": 0.5,
        "selectedIndices": selected_indices,
        "reason": "coupled_3_82b_history_preservation",
    })

async def skills_predict_handler(request: web.Request) -> web.Response:
    return web.json_response({"skills": []})

async def skills_learn_handler(request: web.Request) -> web.Response:
    return web.json_response({"learned": True})

async def entities_track_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})

async def spatial_predict_handler(request: web.Request) -> web.Response:
    return web.json_response({"tool": "none", "confidence": 0.0, "x": 0.0, "y": 0.0})

async def fast_path_route_handler(request: web.Request) -> web.Response:
    return web.json_response({"action": "plan", "skip_planner": False})

async def interest_handler(request: web.Request) -> web.Response:
    return web.json_response({"hasInterest": False, "score": 0.0, "matches": []})

async def predict_handler(request: web.Request) -> web.Response:
    t0 = time.perf_counter()
    try:
        raw_bytes = await request.read()
        try:
            text_body = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_body = raw_bytes.decode("latin-1", errors="replace")
        body = json.loads(text_body)
    except Exception as e:
        return web.json_response({"error": f"JSON inválido: {e}"}, status=400)

    query = str(body.get("text", "")).strip()
    if not query:
        return web.json_response({"error": "El campo 'text' es requerido"}, status=400)

    exclude_list = set(body.get("exclude") or [])

    # Extraer contexto y oferta previa del asistente si existe
    context = str(body.get("context", "")).strip()
    history = body.get("history") or []
    assistant_offer = ""
    if history and isinstance(history, list):
        last_asst = [m.get("content", "") for m in history if isinstance(m, dict) and m.get("role") == "assistant"]
        if last_asst:
            assistant_offer = last_asst[-1].strip()
    if not assistant_offer and context:
        assistant_offer = context

    # 1. Encode query y oferta
    q_tokens, q_pooled = encode_text_tokens([query], prefix="query: ", max_len=24)
    if assistant_offer:
        a_tokens, a_pooled = encode_text_tokens([assistant_offer], prefix="passage: ", max_len=24)
    else:
        a_tokens = torch.zeros(1, 24, 384)
        a_pooled = torch.zeros(1, 384)

    c_global = torch.zeros(1, 384)

    # 2. Rejection vector
    rejected_intent = str(body.get("rejectedIntent", "")).strip()
    if rejected_intent:
        _, rej_vec = encode_text_tokens([rejected_intent], prefix="query: ", max_len=24)
        has_rej = torch.tensor([[1.0]])
    else:
        rej_vec = torch.zeros(1, 384)
        has_rej = torch.tensor([[0.0]])

    # 3. Ledger de sesión
    M = 8
    ledger_queries = torch.zeros(1, M, 384)
    ledger_tools = torch.zeros(1, M, 384)
    ledger_is_mutable = torch.zeros(1, M)
    ledger_mask = torch.zeros(1, M)

    ledger_data = body.get("ledger") or []
    if isinstance(ledger_data, list):
        for slot_idx, item in enumerate(ledger_data[:M]):
            if isinstance(item, dict):
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

    # 4. Selección de Candidatas con Aislamiento Estricto por Tenant
    scope_key = extract_scope_key(request, body)
    tenant_cat = TENANT_REGISTRY.get(scope_key)

    if tenant_cat and len(tenant_cat["tool_ids"]) > 0:
        pool_pooled = tenant_cat["tool_pooled"]
        pool_tensors = tenant_cat["tool_tensors"]
        pool_ids = tenant_cat["tool_ids"]
        pool_dict = tenant_cat["tools_dict"]
        logger.info(f"[predict] Usando catálogo exclusivo del tenant ({len(pool_ids)} herramientas activas)")
    else:
        pool_pooled = catalog_pooled
        pool_tensors = raw_tool_tensors
        pool_ids = ALL_TOOL_IDS
        pool_dict = CATALOG

    search_pooled = q_pooled
    if assistant_offer:
        search_pooled = F.normalize(0.40 * q_pooled + 0.60 * a_pooled, p=2, dim=-1)

    forced_keywords = body.get("keywords") or []
    cands, cand_indices = select_candidates(search_pooled, pool_pooled, pool_ids, k=32, forced_cands=forced_keywords)
    t_tokens = pool_tensors[cand_indices, :32, :].unsqueeze(0)
    t_pooled = pool_pooled[cand_indices].unsqueeze(0)

    # 5. Inferencia en el Motor 3.82B
    with torch.no_grad():
        out = model(
            q_tokens, q_pooled, a_tokens, a_pooled, c_global,
            ledger_queries, ledger_tools, ledger_is_mutable, ledger_mask,
            rej_vec, has_rej, t_tokens, t_pooled
        )

    lat_ms = (time.perf_counter() - t0) * 1000.0

    gated_scores = out["gated_tool_scores"][0].tolist()
    raw_affinities = out["raw_affinities"][0].tolist()
    master_gate = float(out["master_gate"].item())
    p_sink = float(out["p_sink"].item())
    p_noul_cov = float(out["p_noul_coverage"].item())
    p_noul_intent = float(out["p_noul_intent"].item())
    red_prob = float(out["redundancy_prob"].item())

    ranked_indices = np.argsort(gated_scores)[::-1]
    top_cands = [cands[i] for i in ranked_indices[:3]]
    top_scores = [gated_scores[i] for i in ranked_indices[:3]]

    # Criterio de Ejecución / Abstención
    is_abstain = (master_gate < 0.20) or (p_sink > 0.70) or (p_noul_cov < 0.15) or (top_scores[0] < 0.25)
    decision = "ABSTAIN" if is_abstain else "EXECUTE"

    predicted_tools: List[Dict[str, Any]] = []
    ranked_scores: List[float] = []
    calibrated_scores: List[float] = []

    if not is_abstain:
        for idx, tid in enumerate(top_cands):
            if (tid in pool_dict or tid in CATALOG) and tid not in exclude_list:
                score = top_scores[idx]
                if score >= 0.20:
                    t_data = pool_dict.get(tid) or CATALOG.get(tid, {})
                    predicted_tools.append({
                        "id": tid,
                        "name": t_data.get("name", tid),
                        "group": t_data.get("group") or t_data.get("domain_name", "general"),
                        "category": t_data.get("category") or t_data.get("affordance", "QUERY"),
                        "description": t_data.get("desc_es") or t_data.get("description", ""),
                        "tags": t_data.get("tags") or [t_data.get("domain_name", ""), t_data.get("affordance", "")],
                        "intentSummary": t_data.get("intentSummary") or t_data.get("desc_es", ""),
                        "inputSchema": t_data.get("inputSchema") or {"type": "object", "properties": {}},
                        "prerequisites": t_data.get("prerequisites", []),
                        "conflicts": t_data.get("conflicts", []),
                    })
                    ranked_scores.append(round(score, 4))
                    calibrated_scores.append(round(max(0.30, score * master_gate), 4))

    tool_names = [t["name"] for t in predicted_tools]
    graph = {
        "nodes": tool_names,
        "edges": [],
        "executionOrder": tool_names,
    }

    is_file_tool = any("file" in t.lower() or "workspace" in t.lower() for t in tool_names)
    file_command_mode = {
        "enabled": is_file_tool,
        "confidence": master_gate if is_file_tool else 0.0,
        "reason": "predicted_tool" if is_file_tool else "none",
    }

    response_data = {
        "tools": predicted_tools,
        "complexity": "simple" if len(predicted_tools) <= 1 else "moderate",
        "modelSize": "remtk-juicio-jev-moe-3.82b-coupled-a100",
        "rankedScores": ranked_scores,
        "calibratedScores": calibrated_scores,
        "graph": graph,
        "fileCommandMode": file_command_mode,
        "debug": {
            "decision": decision,
            "top_tool": top_cands[0] if top_cands else "none",
            "top_3_tools": top_cands,
            "master_gate": round(master_gate, 4),
            "p_sink": round(p_sink, 4),
            "p_noul_intent": round(p_noul_intent, 4),
            "p_noul_coverage": round(p_noul_cov, 4),
            "redundancy_prob": round(red_prob, 4),
            "is_abstain": is_abstain,
            "latency_ms": round(lat_ms, 1)
        },
    }

    logger.info(
        f"[/predict] 200 OK ({lat_ms:.1f}ms) | decision='{decision}' | tools={tool_names} | prompt='{query[:45]}'"
    )
    return web.json_response(response_data)

# ==============================================================================
# Servidor Principal Dual-Port (6776 y 6876)
# ==============================================================================
def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_post("/predict", predict_handler)
    app.router.add_post("/predict/feedback", predict_feedback_handler)
    app.router.add_get("/tools/count", tools_count_handler)
    app.router.add_get("/tools/ready", tools_ready_handler)
    app.router.add_post("/tools", register_tools_handler)
    app.router.add_post("/memory/predict", memory_predict_handler)
    app.router.add_post("/memory/judge", memory_judge_handler)
    app.router.add_post("/history/regime", history_regime_handler)
    app.router.add_post("/skills/predict", skills_predict_handler)
    app.router.add_post("/skills/learn", skills_learn_handler)
    app.router.add_post("/entities/track", entities_track_handler)
    app.router.add_post("/spatial/predict", spatial_predict_handler)
    app.router.add_post("/predict/spatial", spatial_predict_handler)
    app.router.add_post("/predict/route", fast_path_route_handler)
    app.router.add_post("/interest", interest_handler)
    return app

import asyncio

async def start_dual_server():
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    host = "0.0.0.0"
    for p in [6776, 6876]:
        site = web.TCPSite(runner, host, p)
        await site.start()
        logger.info(f"REMTK 3.82B Prediction Service activo en http://{host}:{p}")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    logger.info("Iniciando REMTK Coupled 3.82B Production Server...")
    try:
        asyncio.run(start_dual_server())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Servidor detenido.")
