import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

print("Torch version:", torch.__version__)

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
        # q: [B, D], t: [B, K, D]
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
        # x: [B, L, C]
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

print("Instantiating RemtkJuicioJevMoECoupledCheckpointModel on meta device...")
with torch.device('meta'):
    model = RemtkJuicioJevMoECoupledCheckpointModel()
model_keys = set(dict(model.named_parameters()).keys())
print(f"Model instantiated with {len(model_keys)} parameters.")

ckpt_path = r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk_juicio_jev_moe_3_82B_zero_dup_a100.pt"
print(f"Loading checkpoint mmap from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
sd = ckpt["model_state_dict"]

missing, unexpected = model.load_state_dict(sd, strict=True, assign=True)
print("Missing keys:", len(missing), missing[:5])
print("Unexpected keys:", len(unexpected), unexpected[:5])
if len(missing) == 0 and len(unexpected) == 0:
    print("SUCCESS! 100% PERFECT MATCH WITH 0 MISSING AND 0 UNEXPECTED KEYS!")
