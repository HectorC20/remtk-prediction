from pathlib import Path
import re

server_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts\serve_3_82b_production.py")
code = server_path.read_text(encoding="utf-8")

# 1. Add TenantRegistry and helper functions
tenant_registry_code = '''
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
'''

# 2. Replace select_candidates & handlers
old_block = '''# ==============================================================================
# Helper de Selección de Candidatas
# ==============================================================================
def select_candidates(query_pooled: torch.Tensor, k: int = 32, forced_cands: Optional[List[str]] = None) -> List[str]:
    sims = torch.matmul(query_pooled, catalog_pooled.T)[0]
    top_indices = torch.topk(sims, k=k).indices.tolist()
    cands = [ALL_TOOL_IDS[idx] for idx in top_indices]
    if forced_cands:
        for fc in reversed(forced_cands):
            if fc in tool_id_to_idx and fc not in cands:
                cands.pop()
                cands.insert(0, fc)
    return cands[:k]

# ==============================================================================
# Handlers HTTP
# ==============================================================================
async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "model": "remtk-juicio-jev-moe-3.82b-coupled-a100",
        "tools_count": len(CATALOG),
        "onnx_runtime_version": "1.20.1",
        "active_device": "cpu",
    })

async def tools_count_handler(request: web.Request) -> web.Response:
    return web.json_response({"count": len(CATALOG)})

async def tools_ready_handler(request: web.Request) -> web.Response:
    return web.json_response({"ready": True, "count": len(CATALOG)})

async def register_tools_handler(request: web.Request) -> web.Response:
    return web.json_response({"indexed": len(CATALOG)})'''

new_block = '''# ==============================================================================
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
    return web.json_response({"indexed": indexed})'''

assert old_block in code, "old_block not found in code!"
code = code.replace(old_block, tenant_registry_code + "\n" + new_block, 1)

# 3. In predict_handler, replace Step 4 and tool resolution
old_predict_step = '''    # 4. Selección de Candidatas K=32
    search_pooled = q_pooled
    if assistant_offer:
        search_pooled = F.normalize(0.40 * q_pooled + 0.60 * a_pooled, p=2, dim=-1)

    cands = select_candidates(search_pooled, k=32)
    cand_indices = [tool_id_to_idx[c] for c in cands]
    t_tokens = raw_tool_tensors[cand_indices, :32, :].unsqueeze(0)
    t_pooled = catalog_pooled[cand_indices].unsqueeze(0)'''

new_predict_step = '''    # 4. Selección de Candidatas con Aislamiento Estricto por Tenant
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
    t_pooled = pool_pooled[cand_indices].unsqueeze(0)'''

assert old_predict_step in code, "old_predict_step not found in code!"
code = code.replace(old_predict_step, new_predict_step, 1)

# 4. Update the dictionary lookup in predicted_tools
old_dict_lookup = '''        for idx, tid in enumerate(top_cands):
            if tid in CATALOG and tid not in exclude_list:
                score = top_scores[idx]
                if score >= 0.20:
                    t_data = CATALOG[tid]
                    predicted_tools.append({
                        "id": tid,
                        "name": t_data.get("name", tid),
                        "group": t_data.get("domain_name", "general"),
                        "category": t_data.get("affordance", "QUERY"),
                        "description": t_data.get("desc_es") or t_data.get("description", ""),
                        "tags": [t_data.get("domain_name", ""), t_data.get("affordance", "")],
                        "intentSummary": t_data.get("desc_es", ""),
                        "inputSchema": t_data.get("inputSchema") or {"type": "object", "properties": {}},
                        "prerequisites": t_data.get("prerequisites", []),
                        "conflicts": t_data.get("conflicts", []),
                    })'''

new_dict_lookup = '''        for idx, tid in enumerate(top_cands):
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
                    })'''

assert old_dict_lookup in code, "old_dict_lookup not found in code!"
code = code.replace(old_dict_lookup, new_dict_lookup, 1)

server_path.write_text(code, encoding="utf-8")
print("serve_3_82b_production.py successfully updated with TenantRegistry!")
