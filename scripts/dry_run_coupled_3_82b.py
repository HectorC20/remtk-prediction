import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            "colbert_scores": colbert_scores
        }

if __name__ == "__main__":
    ckpt_path = r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk_juicio_jev_moe_3_82B_zero_dup_a100.pt"
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    sd = ckpt["model_state_dict"]

    with torch.device("meta"):
        model = RemtkJuicioJevMoECoupledCheckpointModel()
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval()
    print("Model ready on CPU!")

    print("Running dry-run forward pass with 1 synthetic episode...")
    B, L_q, L_a, K, L_t, M, D = 1, 24, 24, 32, 32, 8, 384
    q_tokens = torch.randn(B, L_q, D)
    q_pooled = torch.randn(B, D)
    a_offer_tokens = torch.randn(B, L_a, D)
    a_offer_pooled = torch.randn(B, D)
    c_global = torch.randn(B, D)
    ledger_queries = torch.zeros(B, M, D)
    ledger_tools = torch.zeros(B, M, D)
    ledger_is_mutable = torch.zeros(B, M)
    ledger_mask = torch.zeros(B, M)
    rejected_vec = torch.zeros(B, D)
    has_rejection = torch.zeros(B, 1)
    t_tokens = torch.randn(B, K, L_t, D)
    t_pooled = torch.randn(B, K, D)

    with torch.no_grad():
        out = model(
            q_tokens, q_pooled, a_offer_tokens, a_offer_pooled, c_global,
            ledger_queries, ledger_tools, ledger_is_mutable, ledger_mask,
            rejected_vec, has_rejection, t_tokens, t_pooled
        )

    print("Dry-run SUCCESS!")
    for k, v in out.items():
        print(f"  {k:20s}: shape {list(v.shape)}, mean={v.mean().item():.4f}")
