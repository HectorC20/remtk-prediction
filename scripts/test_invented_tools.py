#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba de Generalización Zero-Shot con Herramientas Nuevas e Inventadas
Modelo: REMTK-JUICIO-JEV-MOE 3.82B (Checkpoint Zero Dup A100)
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

print(f"[Init] Cargando metadatos y catálogo desde {CATALOG_META_PATH}...")
with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

ALL_TOOL_IDS = META["all_tool_ids"]
cat_safetensors = load_file(str(CATALOG_TENSORS_PATH))
raw_tool_tensors = cat_safetensors["catalog_tool_tensors"].to(dtype=torch.float32)

print(f"[Init] Inicializando Multilingual-E5-Small ONNX desde {E5_DIR}...")
e5_tok = AutoTokenizer.from_pretrained(str(E5_DIR))
e5_session = ort.InferenceSession(str(E5_DIR / "model.onnx"))

def encode_text(texts: List[str], prefix: str = "query: ", max_len: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
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
# Arquitectura Exacta
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

        # 1. Fase 1
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

        # 2. Capa 1
        h1, lang_logits = self.capa1(q_tokens_res, c_global)

        # 3. Fase 2 Redundancy
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

        # 4. Fase 3 Late Interaction (ColBERT) + MoE con [NOUL_SINK]
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

# Carga de modelo
print(f"[Init] Instanciando RemtkJuicioJevMoECoupledCheckpointModel...")
with torch.device("meta"):
    model = RemtkJuicioJevMoECoupledCheckpointModel()

print(f"[Init] Cargando checkpoint mmap desde {CKPT_PATH}...")
ckpt = torch.load(str(CKPT_PATH), map_location="cpu", mmap=True)
model.load_state_dict(ckpt["model_state_dict"], strict=True, assign=True)
model.eval()
print("[Init] Modelo listo para evaluar herramientas nuevas e inventadas!\n")

# ==============================================================================
# DEFINICIÓN DE HERRAMIENTAS INVENTADAS (100% Nuevas / Nunca Vistas)
# ==============================================================================
INVENTED_TOOLS = [
    {
        "id": "dron_cosecha_maiz_agricola",
        "description": "Despliega una flota de drones autónomos agrícolas con sensores multiespectrales para cosechar y recolectar maíz en campos de cultivo abiertos."
    },
    {
        "id": "holograma_proyector_3d_activar",
        "description": "Enciende y calibra el proyector de hologramas tridimensionales de la sala de reuniones para visualización interactiva de maquetas y arquitectura 3D."
    },
    {
        "id": "tiki_cocteleria_halloween_receta",
        "description": "Genera fórmulas y recetas secretas de mixología para cócteles temáticos de Halloween con hielo seco y licores artesanales en Tiki Rooftop."
    },
    {
        "id": "criogenia_temperatura_regular",
        "description": "Ajusta la temperatura criogénica y el nivel de nitrógeno líquido en los tanques de biopreservación celular a menos de ciento noventa y seis grados."
    },
    {
        "id": "bunker_escudo_cuantico_sellar",
        "description": "Activa los protocolos de defensa de emergencia, sellado de compuertas blindadas y barrera de plasma cuántico en las instalaciones del búnker."
    }
]

print("=" * 80)
print("CODIFICANDO LAS 5 HERRAMIENTAS INVENTADAS CON E5...")
print("=" * 80)
invented_texts = [f"{t['id']}: {t['description']}" for t in INVENTED_TOOLS]
inv_tokens, inv_pooled = encode_text(invented_texts, prefix="passage: ", max_len=32)

for i, t in enumerate(INVENTED_TOOLS):
    print(f"  [{i+1}] {t['id']}")
    print(f"      Desc: {t['description']}")

# Tomar 27 herramientas distractoras del catálogo existente para armar un pool K=32
distractor_indices = list(range(0, 400, 15))[:27]
distractor_ids = [ALL_TOOL_IDS[idx] for idx in distractor_indices]
distractor_tokens = raw_tool_tensors[distractor_indices, :32, :] # [27, 32, 384]

cat_mask = (torch.norm(distractor_tokens, p=2, dim=-1) > 1e-5).float()
cat_lens = cat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
distractor_pooled = F.normalize((distractor_tokens * cat_mask.unsqueeze(-1)).sum(dim=1) / cat_lens, p=2, dim=-1) # [27, 384]

# Pool de 32 herramientas: 5 inventadas + 27 distractoras
pool_32_ids = [t["id"] for t in INVENTED_TOOLS] + distractor_ids
pool_32_tokens = torch.cat([inv_tokens, distractor_tokens], dim=0).unsqueeze(0) # [1, 32, 32, 384]
pool_32_pooled = torch.cat([inv_pooled, distractor_pooled], dim=0).unsqueeze(0) # [1, 32, 384]

print(f"\nPool activo de evaluación armado: 32 herramientas (5 inventadas + 27 distractoras).\n")

# ==============================================================================
# ESCENARIOS DE PRUEBA: Afinidad, Descarte y [NOUL_SINK]
# ==============================================================================
CASOS_PRUEBA = [
    # 1. Match Directo sobre Herramienta Inventada 1 (Drones Maíz)
    {
        "nombre": "Caso 1: Drones de cosecha de maíz",
        "query": "Mandá inmediatamente la flota de drones para cosechar todo el maíz del lote norte antes de la lluvia.",
        "herramienta_esperada": "dron_cosecha_maiz_agricola",
        "debe_ejecutar": True
    },
    # 2. Match Directo sobre Herramienta Inventada 2 (Holograma 3D)
    {
        "nombre": "Caso 2: Proyector de holograma",
        "query": "Prendé el proyector de hologramas en 3D en el auditorio para mostrar la maqueta del edificio.",
        "herramienta_esperada": "holograma_proyector_3d_activar",
        "debe_ejecutar": True
    },
    # 3. Match con Sinónimos y Jerga sobre Herramienta Inventada 3 (Cócteles Halloween)
    {
        "nombre": "Caso 3: Cóctel Halloween (Tiki)",
        "query": "¿Qué trago o pócima con hielo seco podemos preparar para la noche de brujas en el rooftop?",
        "herramienta_esperada": "tiki_cocteleria_halloween_receta",
        "debe_ejecutar": True
    },
    # 4. Match Técnico sobre Herramienta Inventada 4 (Criogenia)
    {
        "nombre": "Caso 4: Tanque Criogénico",
        "query": "Baja la temperatura a menos de -196 grados y regula el nitrógeno líquido de las cápsulas celulares.",
        "herramienta_esperada": "criogenia_temperatura_regular",
        "debe_ejecutar": True
    },
    # 5. Match Crítico sobre Herramienta Inventada 5 (Búnker Cuántico)
    {
        "nombre": "Caso 5: Blindaje y Búnker",
        "query": "Alerta roja en el complejo: sellen todas las compuertas blindadas y activen el escudo cuántico del búnker.",
        "herramienta_esperada": "bunker_escudo_cuantico_sellar",
        "debe_ejecutar": True
    },
    # 6. Descarte Ortogonal: Ninguna de las 5 inventadas ni distractoras cubre la intención
    {
        "nombre": "Caso 6: Solicitud No Cubierta -> Debe ir al [NOUL_SINK]",
        "query": "Necesito reservar un vuelo comercial de ida y vuelta a París para el próximo mes.",
        "herramienta_esperada": "[NOUL_SINK]",
        "debe_ejecutar": False
    },
    # 7. Consulta Ambigua / Conversacional
    {
        "nombre": "Caso 7: Agradecimiento social -> Debe abstenerse",
        "query": "Muchísimas gracias por explicarme todo tan claro, que tengas buena tarde.",
        "herramienta_esperada": "[ABSTAIN]",
        "debe_ejecutar": False
    }
]

print("=" * 80)
print("EJECUTANDO EVALUACIÓN ZERO-SHOT EN TIEMPO REAL")
print("=" * 80)

resultados = []

for idx, caso in enumerate(CASOS_PRUEBA):
    print(f"\n[{idx+1}/{len(CASOS_PRUEBA)}] {caso['nombre']}")
    print(f"  Query: \"{caso['query']}\"")
    print(f"  Esperado: {caso['herramienta_esperada']} (Debe ejecutar: {caso['debe_ejecutar']})")

    t0 = time.perf_counter()
    q_tokens, q_pooled = encode_text([caso["query"]], prefix="query: ", max_len=24)

    # Entradas cero para oferta asistente, historial y rechazo
    a_tokens = torch.zeros(1, 24, 384)
    a_pooled = torch.zeros(1, 384)
    c_global = torch.zeros(1, 384)
    ledger_queries = torch.zeros(1, 8, 384)
    ledger_tools = torch.zeros(1, 8, 384)
    ledger_is_mutable = torch.zeros(1, 8)
    ledger_mask = torch.zeros(1, 8)
    rej_vec = torch.zeros(1, 384)
    has_rej = torch.tensor([[0.0]])

    with torch.no_grad():
        out = model(
            q_tokens, q_pooled, a_tokens, a_pooled, c_global,
            ledger_queries, ledger_tools, ledger_is_mutable, ledger_mask,
            rej_vec, has_rej, pool_32_tokens, pool_32_pooled
        )

    t_ms = (time.perf_counter() - t0) * 1000

    gated_scores = out["gated_tool_scores"][0].tolist()
    raw_affinities = out["raw_affinities"][0].tolist()
    colbert_scores = out["colbert_scores"][0].tolist()
    master_gate = out["master_gate"].item()
    p_sink = out["p_sink"].item()
    p_noul_cov = out["p_noul_coverage"].item()
    p_noul_intent = out["p_noul_intent"].item()

    ranked = np.argsort(gated_scores)[::-1]
    top_cand = pool_32_ids[ranked[0]]
    top_score = gated_scores[ranked[0]]
    top_raw = raw_affinities[ranked[0]]
    top_colbert = colbert_scores[ranked[0]]

    print(f"  ⏱️  Latencia: {t_ms:.1f} ms")
    print(f"  🛡️  MasterGate: {master_gate:.4f} | Sink: {p_sink:.4f} | Intent: {p_noul_intent:.4f} | Cov: {p_noul_cov:.4f}")
    print(f"  🏆 Top-1 Seleccionado: {top_cand}")
    print(f"     GatedScore: {top_score:.4f} | RawAffinity: {top_raw:.4f} | ColBERT: {top_colbert:.4f}")

    if caso["debe_ejecutar"]:
        acierto = (top_cand == caso["herramienta_esperada"]) and (top_score > 0.50)
        veredicto = "✅ ACIERTO TOTAL" if acierto else "❌ FALLO"
    else:
        acierto = (master_gate < 0.10) or (p_sink > 0.80) or (top_score < 0.10)
        veredicto = "✅ ABSTENCIÓN CORRECTA ([NOUL_SINK])" if acierto else "❌ FALSA ALARMA"

    print(f"  👉 Veredicto: {veredicto}")
    sys.stdout.flush()

    resultados.append({
        "caso": caso["nombre"],
        "query": caso["query"],
        "esperado": caso["herramienta_esperada"],
        "top_seleccionado": top_cand,
        "gated_score": round(top_score, 4),
        "raw_affinity": round(top_raw, 4),
        "colbert": round(top_colbert, 4),
        "master_gate": round(master_gate, 4),
        "p_sink": round(p_sink, 4),
        "veredicto": veredicto
    })

res_file = BASE_DIR / "scripts" / "results_invented_tools_test.json"
with open(res_file, "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print(f"\n[Completado] Reporte guardado en {res_file}")
