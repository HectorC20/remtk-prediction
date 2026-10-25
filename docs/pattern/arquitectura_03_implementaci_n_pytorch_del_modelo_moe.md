# Arquitectura 03: Código Fuente de la Red `RemtkJuicioMoE` (300M Parámetros) en PyTorch

Este módulo implementa la arquitectura corregida de **~302.4M de parámetros** (`d_proj = 1024`, `d_out = 512`, `num_experts = 6`), con **Dilated Gated Convolutions**, **Multi-Head Attentive Pooling**, **Enmascaramiento Automático de Padding en MaxSim** y **Set-Attention Transformer de 3 capas**, 100% exportable a ONNX (`opset_version = 17`).

```python
# training/02_architecture_moe.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedGatedConvBlock(nn.Module):
    """
    Codificador secuencial multiescala con dilatación configurable.
    Con 3 capas (dilation=1, 2, 4) alcanza un campo receptivo de 15 tokens,
    capturando negaciones distantes ("no quiero X... mejor haz Y") y condicionales.
    """
    def __init__(self, d_model: int = 384, d_hidden: int = 768, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_hidden * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=4
        )
        self.out_proj = nn.Linear(d_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) en {0.0, 1.0}
        residual = x
        x_masked = x * mask.unsqueeze(-1)
        h = x_masked.transpose(1, 2)          # (B, D, L)
        h = self.conv(h).transpose(1, 2)      # (B, L, 2*d_hidden)
        a, b = h.chunk(2, dim=-1)
        h = a * F.silu(b)                     # SwiGLU Gating
        h = self.out_proj(h)
        h = self.norm(residual + self.dropout(h))
        return h * mask.unsqueeze(-1)


class AttentivePooling(nn.Module):
    """
    Reemplaza el mean(dim=1) plano. Aprende a concentrar el peso en verbos de acción,
    partículas de negación y sustantivos clave, ignorando relleno y conectores vacíos.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) en {0.0, 1.0}
        scores = self.attn(x).squeeze(-1)     # (B, L)
        scores = scores.masked_fill(mask < 0.5, -1e4)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return torch.sum(x * weights, dim=1)  # (B, D)


class Etapa1CompuertaEpisodicaNLI(nn.Module):
    """
    Etapa 1 (~22.4M params): Evalúa (query_tokens, context_tokens) con dilatación multiescala,
    atención cruzada enmascarada y Attentive Pooling.
    """
    def __init__(self, d_model: int = 384, d_attn: int = 768):
        super().__init__()
        self.q_blocks = nn.ModuleList([
            DilatedGatedConvBlock(d_model, d_hidden=768, dilation=1),
            DilatedGatedConvBlock(d_model, d_hidden=768, dilation=2),
            DilatedGatedConvBlock(d_model, d_hidden=768, dilation=4),
        ])
        self.c_blocks = nn.ModuleList([
            DilatedGatedConvBlock(d_model, d_hidden=768, dilation=1),
            DilatedGatedConvBlock(d_model, d_hidden=768, dilation=2),
            DilatedGatedConvBlock(d_model, d_hidden=768, dilation=4),
        ])

        self.w_q = nn.Linear(d_model, d_attn, bias=False)
        self.w_k = nn.Linear(d_model, d_attn, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.q_pooler = AttentivePooling(d_model)
        self.c_pooler = AttentivePooling(d_model)
        self.m_pooler = AttentivePooling(d_model)

        self.nli_classifier = nn.Sequential(
            nn.Linear(d_model * 4, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 4)  # 0:TRANSICION, 1:CONTINUACION, 2:CORRECCION, 3:ABANDONO
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(4 + d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, q_seq: torch.Tensor, c_seq: torch.Tensor, q_mask: torch.Tensor, c_mask: torch.Tensor):
        q_tilde = q_seq
        for block in self.q_blocks:
            q_tilde = block(q_tilde, q_mask)

        c_tilde = c_seq
        for block in self.c_blocks:
            c_tilde = block(c_tilde, c_mask)

        # Cross-Attention enmascarando el padding de context_tokens
        scale = 1.0 / math.sqrt(self.w_q.out_features)
        attn_scores = torch.matmul(self.w_q(q_tilde), self.w_k(c_tilde).transpose(-1, -2)) * scale
        c_pad_mask = (c_mask < 0.5).unsqueeze(1)  # (1, 1, Lc)
        attn_scores = attn_scores.masked_fill(c_pad_mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        m_qc = torch.matmul(attn_weights, self.w_v(c_tilde)) * q_mask.unsqueeze(-1)

        # Attentive Pooling (preserva negaciones y verbos clave)
        q_pool = self.q_pooler(q_tilde, q_mask)   # (1, 384)
        c_pool = self.c_pooler(c_tilde, c_mask)   # (1, 384)
        m_pool = self.m_pooler(m_qc, q_mask)      # (1, 384)

        nli_features = torch.cat([
            q_pool,
            c_pool,
            torch.abs(q_pool - c_pool),
            q_pool * m_pool
        ], dim=-1)  # (1, 1536)

        estado_logits = self.nli_classifier(nli_features)
        estado_probs = F.softmax(estado_logits, dim=-1)
        topic_shift_score = estado_probs[:, 0:1] + estado_probs[:, 3:4]

        gate_in = torch.cat([estado_probs, q_pool, m_pool], dim=-1)
        alpha = self.gate_proj(gate_in).unsqueeze(1)  # (1, 1, 1)

        q_fused = self.out_norm(q_tilde + alpha * m_qc) * q_mask.unsqueeze(-1)
        return q_fused, q_pool, estado_logits, estado_probs, topic_shift_score, alpha.squeeze(-1)


classSwiGLUProjector(nn.Module):
    """Proyector residual denso de alta capacidad para cada experto."""
    def __init__(self, d_in: int = 384, d_proj: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(d_in, d_proj * 2)
        self.fc_mid = nn.Linear(d_proj, d_proj)
        self.norm1 = nn.LayerNorm(d_proj)
        self.fc_res = nn.Linear(d_proj, d_proj * 2)
        self.fc_out = nn.Linear(d_proj, d_proj)
        self.norm2 = nn.LayerNorm(d_proj)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc_in(x).chunk(2, dim=-1)
        h = self.norm1(self.fc_mid(a * F.silu(b)))
        c, d = self.fc_res(h).chunk(2, dim=-1)
        h2 = self.fc_out(c * F.silu(d))
        return self.norm2(h + self.dropout(h2))


class ExpertoInteraccionTool300M(nn.Module):
    """
    Macro-Experto (~44M params por experto):
    Proyecta a d_proj=1024, ejecuta MaxSim Bidireccional Enmascarado e Interacción Bilineal Profunda.
    """
    def __init__(self, d_model: int = 384, d_proj: int = 1024, d_out: int = 512):
        super().__init__()
        self.q_proj = classSwiGLUProjector(d_model, d_proj)
        self.t_proj = classSwiGLUProjector(d_model, d_proj)
        self.q_att_pool = AttentivePooling(d_proj)
        self.t_att_pool = AttentivePooling(d_proj)

        # Entrada al scorer: q_pool(1024) + t_pool(1024) + prod(1024) + diff(1024) + 6 estadísticos MaxSim = 4102
        self.scorer_mlp = nn.Sequential(
            nn.Linear(d_proj * 4 + 6, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, d_out),
            nn.LayerNorm(d_out),
            nn.GELU()
        )

    def forward(
        self,
        q_fused: torch.Tensor,
        tools_seq: torch.Tensor,
        q_mask: torch.Tensor,
        t_mask: torch.Tensor
    ) -> torch.Tensor:
        # q_fused: (1, Lq, 384), tools_seq: (K, Lt, 384)
        # q_mask: (1, Lq), t_mask: (K, Lt)
        K = tools_seq.size(0)
        q_emb = F.normalize(self.q_proj(q_fused), p=2, dim=-1)    # (1, Lq, 1024)
        t_emb = F.normalize(self.t_proj(tools_seq), p=2, dim=-1)  # (K, Lt, 1024)

        q_exp = q_emb.expand(K, -1, -1)                           # (K, Lq, 1024)
        sim_matrix = torch.bmm(q_exp, t_emb.transpose(1, 2))      # (K, Lq, Lt)

        # Enmascarar posiciones de padding antes de calcular MaxSim
        valid_pair_mask = torch.bmm(q_mask.expand(K, -1).unsqueeze(2), t_mask.unsqueeze(1))  # (K, Lq, Lt)
        sim_masked_for_max = sim_matrix.masked_fill(valid_pair_mask < 0.5, -1.0)

        # 1. MaxSim Forward Enmascarado (Query -> Tool)
        max_sim_fwd, _ = sim_masked_for_max.max(dim=-1)           # (K, Lq)
        q_len_valid = q_mask.sum(dim=-1, keepdim=True).clamp(min=1.0).expand(K, -1)
        mean_max_fwd = (max_sim_fwd * q_mask.expand(K, -1)).sum(dim=-1, keepdim=True) / q_len_valid
        top_max_fwd, _ = max_sim_fwd.max(dim=-1, keepdim=True)
        min_max_fwd, _ = max_sim_fwd.masked_fill(q_mask.expand(K, -1) < 0.5, 1.0).min(dim=-1, keepdim=True)

        # 2. MaxSim Backward Enmascarado (Tool -> Query: Especificidad sin sesgo por longitud de tool)
        max_sim_bwd, _ = sim_masked_for_max.max(dim=1)            # (K, Lt)
        t_len_valid = t_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_max_bwd = (max_sim_bwd * t_mask).sum(dim=-1, keepdim=True) / t_len_valid
        top_max_bwd, _ = max_sim_bwd.max(dim=-1, keepdim=True)
        coverage_bwd = ((max_sim_bwd > 0.65).float() * t_mask).sum(dim=-1, keepdim=True) / t_len_valid

        # 3. Attentive Pooling por secuencia
        q_pool = self.q_att_pool(q_emb, q_mask).expand(K, -1)     # (K, 1024)
        t_pool = self.t_att_pool(t_emb, t_mask)                   # (K, 1024)

        combined = torch.cat([
            q_pool,
            t_pool,
            q_pool * t_pool,
            torch.abs(q_pool - t_pool),
            mean_max_fwd,
            top_max_fwd,
            min_max_fwd,
            mean_max_bwd,
            top_max_bwd,
            coverage_bwd
        ], dim=-1)  # (K, 4102)

        return self.scorer_mlp(combined)  # (K, 512)


class Etapa2MixtureOfExperts300M(nn.Module):
    """
    Etapa 2 (~264M params): Router + 6 Macro-Expertos Especializados.
    """
    def __init__(self, num_experts: int = 6, d_model: int = 384, d_proj: int = 1024, d_out: int = 512):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(d_model * 2 + 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_experts)
        )
        self.experts = nn.ModuleList([
            ExpertoInteraccionTool300M(d_model, d_proj, d_out) for _ in range(num_experts)
        ])

    def forward(
        self,
        q_fused: torch.Tensor,
        q_att_summary: torch.Tensor,
        estado_probs: torch.Tensor,
        tools_seq: torch.Tensor,
        q_mask: torch.Tensor,
        t_mask: torch.Tensor
    ):
        q_mean_valid = (q_fused * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        router_in = torch.cat([q_att_summary, q_mean_valid, estado_probs], dim=-1)  # (1, 772)
        router_logits = self.router(router_in)  # (1, 6)

        top2_vals, top2_idx = torch.topk(router_logits, k=2, dim=-1)
        top2_weights = F.softmax(top2_vals, dim=-1)
        sparse_gates = torch.zeros_like(router_logits).scatter(-1, top2_idx, top2_weights)  # (1, 6)

        expert_outputs = []
        for expert in self.experts:
            h_i = expert(q_fused, tools_seq, q_mask, t_mask)  # (K, 512)
            expert_outputs.append(h_i.unsqueeze(0))           # (1, K, 512)

        stacked = torch.cat(expert_outputs, dim=0)            # (6, K, 512)
        gates_expanded = sparse_gates.squeeze(0).view(self.num_experts, 1, 1)
        h_tools = torch.sum(stacked * gates_expanded, dim=0)  # (K, 512)

        return h_tools, sparse_gates, F.softmax(router_logits, dim=-1)


class Etapa3CabezaSetDecisionAbstencion300M(nn.Module):
    """
    Etapa 3 (~16M params): Pila de 3 capas Transformer Set-Attention entre las K herramientas
    + Cabeza de Abstención consciente de la cardinalidad K.
    """
    def __init__(self, d_in: int = 512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.set_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(d_in)

        self.score_head = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 14 estadísticos (incluyendo log(K)) + 2 * d_in (max_pool + mean_pool) = 1038
        self.abstencion_head = nn.Sequential(
            nn.Linear(14 + d_in * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, h_tools: torch.Tensor, estado_probs: torch.Tensor, gate_alpha: torch.Tensor):
        # h_tools: (K, 512)
        h_set = h_tools.unsqueeze(0)                              # (1, K, 512)
        h_refined = self.norm(self.set_transformer(h_set)).squeeze(0)  # (K, 512)

        raw_logits = self.score_head(h_refined).squeeze(-1)       # (K,)
        ranked_scores = torch.sigmoid(raw_logits)                 # (K,)

        K_float = torch.tensor(float(max(1, ranked_scores.size(0))), device=raw_logits.device)
        log_k_norm = (torch.log(K_float + 1.0) / math.log(201.0)).unsqueeze(0)

        sorted_scores, _ = torch.sort(ranked_scores, descending=True)
        top1 = sorted_scores[0:1]
        top2 = sorted_scores[1:2] if sorted_scores.size(0) > 1 else torch.zeros_like(top1)
        top3 = sorted_scores[2:3] if sorted_scores.size(0) > 2 else torch.zeros_like(top1)
        margen_12 = top1 - top2
        margen_23 = top2 - top3

        mean_s = ranked_scores.mean().unsqueeze(0)
        std_s = ranked_scores.std().unsqueeze(0) if ranked_scores.size(0) > 1 else torch.zeros_like(top1)

        probs_dist = F.softmax(raw_logits * 4.0, dim=-1)
        entropia = -torch.sum(probs_dist * torch.log(probs_dist + 1e-8)).unsqueeze(0)
        denom = torch.log(torch.clamp(K_float, min=2.0))
        entropia_norm = entropia / denom

        max_tool_feat, _ = h_refined.max(dim=0)                   # (512,)
        mean_tool_feat = h_refined.mean(dim=0)                    # (512,)

        abs_features = torch.cat([
            top1,
            top2,
            top3,
            margen_12,
            margen_23,
            entropia_norm,
            mean_s,
            std_s,
            log_k_norm,
            estado_probs.squeeze(0),
            gate_alpha.squeeze(0),
            max_tool_feat,
            mean_tool_feat
        ], dim=-1).unsqueeze(0)  # (1, 1038)

        abstencion_prob = self.abstencion_head(abs_features)      # (1, 1)
        return ranked_scores, raw_logits, abstencion_prob


class RemtkJuicioMoE(nn.Module):
    """
    Grafo Completo de ~302.4M de Parámetros para exportar a un único archivo ONNX.
    Detecta automáticamente máscaras de padding a partir de la norma L2 de los tokens.
    """
    def __init__(self, num_experts: int = 6, d_model: int = 384, d_proj: int = 1024, d_out: int = 512):
        super().__init__()
        self.etapa1 = Etapa1CompuertaEpisodicaNLI(d_model=d_model, d_attn=768)
        self.etapa2 = Etapa2MixtureOfExperts300M(
            num_experts=num_experts, d_model=d_model, d_proj=d_proj, d_out=d_out
        )
        self.etapa3 = Etapa3CabezaSetDecisionAbstencion300M(d_in=d_out)

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor, tools_tokens: torch.Tensor):
        # Detección diferenciable/simbólica de máscaras reales (ignora vectores cero de padding)
        q_mask = (torch.norm(query_tokens, p=2, dim=-1) > 1e-5).float()    # (1, Lq)
        c_mask = (torch.norm(context_tokens, p=2, dim=-1) > 1e-5).float()  # (1, Lc)
        t_mask = (torch.norm(tools_tokens, p=2, dim=-1) > 1e-5).float()    # (K, Lt)

        # Garantizar al menos el primer token activo si un tensor entra vacío
        q_mask[:, 0] = 1.0
        c_mask[:, 0] = 1.0
        t_mask[:, 0] = 1.0

        q_fused, q_att_summary, estado_logits, estado_probs, topic_shift_score, gate_alpha = self.etapa1(
            query_tokens, context_tokens, q_mask, c_mask
        )
        h_tools, sparse_gates, router_probs = self.etapa2(
            q_fused, q_att_summary, estado_probs, tools_tokens, q_mask, t_mask
        )
        ranked_scores, raw_logits, abstencion_prob = self.etapa3(
            h_tools, estado_probs, gate_alpha
        )
        return ranked_scores, estado_probs, abstencion_prob, topic_shift_score, sparse_gates