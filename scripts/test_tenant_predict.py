import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts")
import serve_3_82b_production as srv

tenant_id = "4f4e8014-0a50-42ad-b1db-d6e4ffd4f9c8"
scope_key = f"{tenant_id}::"

user_tools_payload = [
    {"id": "schedule_task", "name": "schedule_task", "group": "scheduler", "description": "Programa recordatorios y cron jobs del agente"},
    {"id": "schedule_list", "name": "schedule_list", "group": "scheduler", "description": "Lista tareas programadas y recordatorios"},
    {"id": "web_search", "name": "web_search", "group": "search", "description": "Busca en la web"},
    {"id": "web_fetch", "name": "web_fetch", "group": "search", "description": "Descarga contenido web"},
    {"id": "search_memory", "name": "search_memory", "group": "knowledge", "description": "Busca en la memoria conversacional"},
    {"id": "search_knowledge_base", "name": "search_knowledge_base", "group": "knowledge", "description": "Busca en la base de conocimiento"},
    {"id": "knowledge_add", "name": "knowledge_add", "group": "knowledge", "description": "Agrega información a la base de conocimiento"},
    {"id": "document_generate", "name": "document_generate", "group": "documents", "description": "Genera documentos PDF o DOCX"},
    {"id": "image_generate", "name": "image_generate", "group": "image", "description": "Genera imágenes culinarias del restaurante"},
    {"id": "list_tools", "name": "list_tools", "group": "own_tools", "description": "Lista las herramientas disponibles"}
]

print("1. Registering tools in TENANT_REGISTRY...")
srv.TENANT_REGISTRY.register(scope_key, user_tools_payload)

tenant_cat = srv.TENANT_REGISTRY.get(scope_key)
print("Registered:", len(tenant_cat["tool_ids"]))

import numpy as np

test_cases = [
    {
        "query": "a ver con el recordatorio",
        "offer": ""
    },
    {
        "query": "a ver con el recordatorio",
        "offer": "¿Retomamos el recordatorio o la carta embrujada?"
    },
    {
        "query": "programa un recordatorio para mañana a las 8am",
        "offer": ""
    },
    {
        "query": "agrega un cron job para revisar la carta a las 8am",
        "offer": ""
    }
]

c_global = torch.zeros(1, 384)
ledger_queries = torch.zeros(1, 8, 384)
ledger_tools = torch.zeros(1, 8, 384)
ledger_is_mutable = torch.zeros(1, 8)
ledger_mask = torch.zeros(1, 8)
rej_vec = torch.zeros(1, 384)
has_rej = torch.tensor([[0.0]])

pool_pooled = tenant_cat["tool_pooled"]
pool_tensors = tenant_cat["tool_tensors"]
pool_ids = tenant_cat["tool_ids"]
pool_dict = tenant_cat["tools_dict"]

for tc in test_cases:
    q = tc["query"]
    off = tc["offer"]
    print(f"\n==========================================")
    print(f"QUERY: '{q}' | OFFER: '{off}'")
    q_tokens, q_pooled = srv.encode_text_tokens([q], prefix="query: ", max_len=24)
    if off:
        a_tokens, a_pooled = srv.encode_text_tokens([off], prefix="passage: ", max_len=24)
        search_pooled = F.normalize(0.40 * q_pooled + 0.60 * a_pooled, p=2, dim=-1)
    else:
        a_tokens = torch.zeros(1, 24, 384)
        a_pooled = torch.zeros(1, 384)
        search_pooled = q_pooled

    cands, cand_indices = srv.select_candidates(search_pooled, pool_pooled, pool_ids, k=32, forced_cands=[])
    t_tokens = pool_tensors[cand_indices, :32, :].unsqueeze(0)
    t_pooled = pool_pooled[cand_indices].unsqueeze(0)

    with torch.no_grad():
        out = srv.model(
            q_tokens, q_pooled, a_tokens, a_pooled, c_global,
            ledger_queries, ledger_tools, ledger_is_mutable, ledger_mask,
            rej_vec, has_rej, t_tokens, t_pooled
        )

    mg = out["master_gate"].item()
    ps = out["p_sink"].item()
    gated_scores = out["gated_tool_scores"][0].tolist()
    ranked_indices = np.argsort(gated_scores)[::-1]
    top_cands = [cands[i] for i in ranked_indices[:3]]
    top_scores = [round(gated_scores[i], 4) for i in ranked_indices[:3]]
    print(f"Master Gate: {mg:.4f} | p_sink: {ps:.4f}")
    print(f"Top 3 Tools: {list(zip(top_cands, top_scores))}")
