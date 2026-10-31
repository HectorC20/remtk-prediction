from importlib.machinery import SourceFileLoader
import torch

eval_m = SourceFileLoader('eval_m', r'C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts\evaluate_3_8b_400tools_trained.py').load_module()

queries = [
    ("Por favor cancela mi cita de las 5 de la tarde.", "agenda_cancel"),
    ("Fíjate qué citas tengo programadas con mi número 987654321.", "agenda_get_client_appointments"),
    ("Necesito reagendar o mover mi turno para el lunes próximo.", "agenda_reschedule"),
    ("Muéstrame la lista de agentes disponibles que puedo delegar subtareas.", "am_list_agents"),
    ("Manda un broadcast a todos los agentes para sincronizar su estado.", "am_agent_broadcast"),
    ("Recupera el resultado de la tarea que delegué al agente de diseño.", "am_get_agent_result"),
]

for q, exp in queries:
    q_ten = eval_m.encode_tokens([q], prefix="query: ", max_len=40, apply_canonical=True)
    q_mask = (torch.norm(q_ten, p=2, dim=-1) > 1e-5).float()
    q_lens = q_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    q_pooled = torch.nn.functional.normalize((q_ten * q_mask.unsqueeze(-1)).sum(dim=1) / q_lens, p=2, dim=-1)
    sims = torch.matmul(q_pooled, eval_m.catalog_pooled.T)[0]
    top10_idx = torch.topk(sims, k=10).indices.tolist()
    top10_tools = [eval_m.ALL_TOOL_IDS[i] for i in top10_idx]
    in_top10 = exp in top10_tools
    rank = top10_tools.index(exp) if in_top10 else (eval_m.ALL_TOOL_IDS.index(exp) if exp in eval_m.ALL_TOOL_IDS else -1)
    print(f"Q: '{q}' | Exp: {exp}")
    print(f"   In Top-10: {in_top10} (rank: {rank}) | Top 3: {top10_tools[:3]}")
