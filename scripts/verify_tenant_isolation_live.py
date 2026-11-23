import urllib.request
import json

base_url = "http://localhost:6776"
tenant_id = "4f4e8014-0a50-42ad-b1db-d6e4ffd4f9c8"
headers = {
    "Content-Type": "application/json",
    "key-remtk": f"{tenant_id}::"
}

# 1. Test GET /tools/count before registration
req = urllib.request.Request(f"{base_url}/tools/count?tenant={tenant_id}", headers=headers)
with urllib.request.urlopen(req) as resp:
    print("Initial count:", resp.read().decode('utf-8'))

# 2. Register user's active tools (simulating what NestJS sends)
user_tools_payload = {
    "tenant": tenant_id,
    "tools": [
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
}

req = urllib.request.Request(f"{base_url}/tools", data=json.dumps(user_tools_payload).encode('utf-8'), headers=headers)
with urllib.request.urlopen(req) as resp:
    print("Register response:", resp.read().decode('utf-8'))

# 3. Check GET /tools/count after registration
req = urllib.request.Request(f"{base_url}/tools/count?tenant={tenant_id}", headers=headers)
with urllib.request.urlopen(req) as resp:
    count_data = json.loads(resp.read().decode('utf-8'))
    print("Post-register count for tenant:", count_data)
    assert count_data["count"] == 10, f"Expected 10, got {count_data['count']}"

# 4. Check GET /health with tenant header
req = urllib.request.Request(f"{base_url}/health", headers=headers)
with urllib.request.urlopen(req) as resp:
    health_data = json.loads(resp.read().decode('utf-8'))
    print("Health with tenant scope:", health_data)
    assert health_data["tenant_isolated"] is True
    assert health_data["tools_count"] == 10

# 5. Send POST /predict for 'a ver con el recordatorio'
predict_payload = {
    "text": "a ver con el recordatorio",
    "tenant": tenant_id,
    "source": "agent"
}
req = urllib.request.Request(f"{base_url}/predict", data=json.dumps(predict_payload).encode('utf-8'), headers=headers)
with urllib.request.urlopen(req) as resp:
    pred_res = json.loads(resp.read().decode('utf-8'))
    print("\n=== Predict Result for Tenant ===")
    print("Decision:", pred_res.get("decision"))
    print("Tools predicted:", [t["name"] for t in pred_res.get("tools", [])])
    print("Top 3 evaluated:", pred_res.get("debug", {}).get("top_3_tools"))
    print("Master gate:", pred_res.get("debug", {}).get("master_gate"))
    print("p_sink:", pred_res.get("debug", {}).get("p_sink"))
    print("p_noul_intent:", pred_res.get("debug", {}).get("p_noul_intent"))
    print("p_noul_cov:", pred_res.get("debug", {}).get("p_noul_coverage"))

    # Verify that NO foreign tools are in top 3
    top3 = pred_res.get("debug", {}).get("top_3_tools", [])
    assert all(t in [x["name"] for x in user_tools_payload["tools"]] for t in top3), f"Foreign tool detected in top 3: {top3}"
    print("\nSUCCESS: All candidates were strictly drawn from the user's active tenant pool!")
