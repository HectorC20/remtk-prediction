# Arquitectura 03: Código Fuente de la Red `RemtkJuicioMoE` en PyTorch (Exportable a ONNX)

Este módulo contiene la implementación completa y autocontenida de las **3 etapas de decisión** lista para entrenarse en CPU y exportarse al estándar ONNX (`opset_version = 17`).

```python
# training/02_architecture_moe.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConvBlock(nn.Module):
    """Codificador secuencial rápido (reemplaza capas pesadas de Self-Attention en Etapa 1)."""
    def __init__(self, d_model: int = 384, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=4
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        h = x.transpose(1, 2)             # (B, D, L)
        h = self.conv(h).transpose(1, 2)  # (B, L, 2*D)
        a, b = h.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)          # Gated Linear Unit (GLU)
        return self.norm(residual + self.dropout(h))


class Etapa1CompuertaEpisodicaNLI(nn.Module):
    """
    Etapa 1: Evalúa (query_tokens, context_tokens) y produce:
      - estado_probs: [TRANSICION, CONTINUACION, CORRECCION, ABANDONO]
      - topic_shift_score: probabilidad de cambio de tema
      - query_contextualizada: secuencia de query fusionada dinámicamente con el contexto
      - gate_alpha: apertura de la compuerta de contexto
    """
    def __init__(self, d_model: int = 384, d_attn: int = 256):
        super().__init__()
        self.q_enc = GatedConvBlock(d_model)
        self.c_enc = GatedConvBlock(d_model)

        self.w_q = nn.Linear(d_model, d_attn, bias=False)
        self.w_k = nn.Linear(d_model, d_attn, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.nli_classifier = nn.Sequential(
            nn.Linear(d_model * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 4)  # 0:TRANSICION, 1:CONTINUACION, 2:CORRECCION, 3:ABANDONO
        )

        # Compuerta que decide cuánto contexto inyectar según los logits de estado y afinidad
        self.gate_proj = nn.Sequential(
            nn.Linear(4 + d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, q_seq: torch.Tensor, c_seq: torch.Tensor):
        # q_seq: (1, Lq, 384), c_seq: (1, Lc, 384)
        q_tilde = self.q_enc(q_seq)
        c_tilde = self.c_enc(c_seq)

        # Cross-Attention de Query hacia Contexto
        scale = 1.0 / math.sqrt(self.w_q.out_features)
        attn_scores = torch.matmul(self.w_q(q_tilde), self.w_k(c_tilde).transpose(-1, -2)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        m_qc = torch.matmul(attn_weights, self.w_v(c_tilde))  # (1, Lq, 384)

        # Pooling para decisión NLI
        q_pool = q_tilde.mean(dim=1)  # (1, 384)
        c_pool = c_tilde.mean(dim=1)  # (1, 384)
        m_pool = m_qc.mean(dim=1)     # (1, 384)

        nli_features = torch.cat([
            q_pool,
            c_pool,
            torch.abs(q_pool - c_pool),
            q_pool * m_pool
        ], dim=-1)  # (1, 1536)

        estado_logits = self.nli_classifier(nli_features)     # (1, 4)
        estado_probs = F.softmax(estado_logits, dim=-1)       # (1, 4)

        # Topic shift es la suma de probabilidad de TRANSICION (0) + ABANDONO (3)
        topic_shift_score = estado_probs[:, 0:1] + estado_probs[:, 3:4]

        # Compuerta diferenciable alpha
        gate_in = torch.cat([estado_probs, m_pool], dim=-1)
        alpha = self.gate_proj(gate_in).unsqueeze(1)          # (1, 1, 1)

        q_fused = self.out_norm(q_tilde + alpha * m_qc)       # (1, Lq, 384)
        return q_fused, estado_logits, estado_probs, topic_shift_score, alpha.squeeze(-1)


class ExpertoInteraccionTool(nn.Module):
    """
    Un Experto especializado dentro de la Etapa 2.
    Ejecuta MaxSim Forward, MaxSim Backward (Especificidad) e Interacción Densa.
    """
    def __init__(self, d_model: int = 384, d_proj: int = 256, d_out: int = 128):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj)
        )
        self.t_proj = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj)
        )
        # Fusiona: q_pool (256) + t_pool (256) + prod (256) + diff (256) + 4 estadísticos MaxSim
        self.scorer_mlp = nn.Sequential(
            nn.Linear(d_proj * 4 + 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_out),
            nn.GELU()
        )

    def forward(self, q_fused: torch.Tensor, tools_seq: torch.Tensor) -> torch.Tensor:
        # q_fused: (1, Lq, 384)
        # tools_seq: (K, Lt, 384)
        K = tools_seq.size(0)
        q_emb = F.normalize(self.q_proj(q_fused), p=2, dim=-1)     # (1, Lq, 256)
        t_emb = F.normalize(self.t_proj(tools_seq), p=2, dim=-1)   # (K, Lt, 256)

        # Expandir q_emb a (K, Lq, 256) para cruce paralelo con las K tools
        q_exp = q_emb.expand(K, -1, -1)

        # Matriz de similitud token-a-token: (K, Lq, Lt)
        sim_matrix = torch.bmm(q_exp, t_emb.transpose(1, 2))

        # 1. MaxSim Forward (ColBERT: para cada token de la query, el mejor token de la tool)
        max_sim_fwd, _ = sim_matrix.max(dim=-1)          # (K, Lq)
        mean_max_fwd = max_sim_fwd.mean(dim=-1, keepdim=True)
        min_max_fwd, _ = max_sim_fwd.min(dim=-1, keepdim=True)

        # 2. MaxSim Backward (Especificidad: cobertura de los tokens de la tool en la query)
        max_sim_bwd, _ = sim_matrix.max(dim=1)           # (K, Lt)
        mean_max_bwd = max_sim_bwd.mean(dim=-1, keepdim=True)
        top_max_bwd, _ = max_sim_bwd.max(dim=-1, keepdim=True)

        # 3. Representaciones globales por secuencia
        q_pool = q_exp.mean(dim=1)                       # (K, 256)
        t_pool = t_emb.mean(dim=1)                       # (K, 256)

        combined = torch.cat([
            q_pool,
            t_pool,
            q_pool * t_pool,
            torch.abs(q_pool - t_pool),
            mean_max_fwd,
            min_max_fwd,
            mean_max_bwd,
            top_max_bwd
        ], dim=-1)                                       # (K, 1028)

        return self.scorer_mlp(combined)                 # (K, 128)


class Etapa2MixtureOfExperts(nn.Module):
    """
    Etapa 2: Router + 6 Expertos Especializados de Interacción Query-Tool.
    Compatible con ONNX mediante despacho ponderado vectorizado.
    """
    def __init__(self, num_experts: int = 6, d_model: int = 384, d_proj: int = 256, d_out: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(d_model + 4, 128),
            nn.GELU(),
            nn.Linear(128, num_experts)
        )
        self.experts = nn.ModuleList([
            ExpertoInteraccionTool(d_model, d_proj, d_out) for _ in range(num_experts)
        ])

    def forward(self, q_fused: torch.Tensor, estado_probs: torch.Tensor, tools_seq: torch.Tensor):
        # q_fused: (1, Lq, 384), estado_probs: (1, 4), tools_seq: (K, Lt, 384)
        q_summary = q_fused.mean(dim=1)                           # (1, 384)
        router_in = torch.cat([q_summary, estado_probs], dim=-1)  # (1, 388)
        router_logits = self.router(router_in)                    # (1, 6)

        # Selección Top-2 diferenciable y exportable a ONNX sin ramificaciones rotas
        top2_vals, top2_idx = torch.topk(router_logits, k=2, dim=-1)
        top2_weights = F.softmax(top2_vals, dim=-1)               # (1, 2)

        # Máscara densa con ceros en los expertos inactivos y pesos softmax en Top-2
        sparse_gates = torch.zeros_like(router_logits).scatter(-1, top2_idx, top2_weights)  # (1, 6)

        # Evaluar expertos y combinar según compuerta sparse_gates
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            h_i = expert(q_fused, tools_seq)                      # (K, 128)
            expert_outputs.append(h_i.unsqueeze(0))               # (1, K, 128)

        stacked = torch.cat(expert_outputs, dim=0)                # (6, K, 128)
        gates_expanded = sparse_gates.squeeze(0).view(self.num_experts, 1, 1)
        h_tools = torch.sum(stacked * gates_expanded, dim=0)      # (K, 128)

        return h_tools, sparse_gates, F.softmax(router_logits, dim=-1)


class Etapa3CabezaSetDecisionAbstencion(nn.Module):
    """
    Etapa 3: Compara las K herramientas ENTRE SÍ (Set-Attention) y emite:
      - ranked_scores (K,)
      - abstencion_prob (1, 1)
    """
    def __init__(self, d_in: int = 128):
        super().__init__()
        self.set_attn = nn.MultiheadAttention(embed_dim=d_in, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_in)
        self.score_head = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # Entrada a abstención: top1, top2, margen, entropía, media, std, 4 estados, gate_alpha + max_pool(128)
        self.abstencion_head = nn.Sequential(
            nn.Linear(11 + d_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h_tools: torch.Tensor, estado_probs: torch.Tensor, gate_alpha: torch.Tensor):
        # h_tools: (K, 128) -> (1, K, 128) para Set-Attention entre candidatas
        h_set = h_tools.unsqueeze(0)
        attn_out, _ = self.set_attn(h_set, h_set, h_set)
        h_refined = self.norm(h_set + attn_out).squeeze(0)        # (K, 128)

        raw_logits = self.score_head(h_refined).squeeze(-1)       # (K,)
        ranked_scores = torch.sigmoid(raw_logits)                 # (K,) en (0, 1)

        # Estadísticos de distribución para Abstención (seguros cuando K >= 1)
        sorted_scores, _ = torch.sort(ranked_scores, descending=True)
        top1 = sorted_scores[0:1]
        top2 = sorted_scores[1:2] if sorted_scores.size(0) > 1 else torch.zeros_like(top1)
        margen = top1 - top2

        mean_s = ranked_scores.mean().unsqueeze(0)
        std_s = ranked_scores.std().unsqueeze(0) if ranked_scores.size(0) > 1 else torch.zeros_like(top1)

        # Entropía normalizada de Shannon
        probs_dist = F.softmax(raw_logits * 4.0, dim=-1)
        entropia = -torch.sum(probs_dist * torch.log(probs_dist + 1e-8)).unsqueeze(0)
        denom = torch.log(torch.tensor(float(max(2, ranked_scores.size(0))), device=raw_logits.device))
        entropia_norm = entropia / denom

        global_tool_feat, _ = h_refined.max(dim=0)                # (128,)

        abs_features = torch.cat([
            top1,
            top2,
            margen,
            entropia_norm,
            mean_s,
            std_s,
            estado_probs.squeeze(0),
            gate_alpha.squeeze(0),
            global_tool_feat
        ], dim=-1).unsqueeze(0)                                   # (1, 139)

        abstencion_prob = self.abstencion_head(abs_features)      # (1, 1)
        return ranked_scores, raw_logits, abstencion_prob


class RemtkJuicioMoE(nn.Module):
    """Grafo Completo de 3 Etapas para exportar a un único archivo ONNX."""
    def __init__(self):
        super().__init__()
        self.etapa1 = Etapa1CompuertaEpisodicaNLI(d_model=384, d_attn=256)
        self.etapa2 = Etapa2MixtureOfExperts(num_experts=6, d_model=384, d_proj=256, d_out=128)
        self.etapa3 = Etapa3CabezaSetDecisionAbstencion(d_in=128)

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor, tools_tokens: torch.Tensor):
        q_fused, estado_logits, estado_probs, topic_shift_score, gate_alpha = self.etapa1(
            query_tokens, context_tokens
        )
        h_tools, sparse_gates, router_probs = self.etapa2(
            q_fused, estado_probs, tools_tokens
        )
        ranked_scores, raw_logits, abstencion_prob = self.etapa3(
            h_tools, estado_probs, gate_alpha
        )
        return ranked_scores, estado_probs, abstencion_prob, topic_shift_score, sparse_gates
```