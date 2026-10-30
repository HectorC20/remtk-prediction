import json
from pathlib import Path

meta_p = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk-juicio-jev-moe-3.8b-400tools\catalog_metadata.json")
with open(meta_p, "r", encoding="utf-8") as f:
    cat = json.load(f)["catalog"]

scenarios = [
    "agenda_book_appointment",
    "agenda_check_availability",
    "agenda_cancel",
    "agenda_get_client_appointments",
    "agenda_reschedule",
    "agenda_get_today_summary",
    "am_list_agents",
    "am_delegate_to_agent",
    "am_agent_broadcast",
    "am_get_agent_result",
    "database_run_read_query",
    "mongodb_find_documents",
    "s3_bucket_create",
    "redis_cache_flush",
    "database_create_index",
    "k8s_deploy_service",
    "aws_ec2_start_instance",
    "docker_container_restart",
    "cloudflare_waf_block_ip",
    "airflow_trigger_dag",
    "stripe_create_payment_intent",
    "invoice_mark_as_paid",
    "accounting_close_fiscal_period",
    "accounting_get_balance_sheet",
    "whatsapp_send_message",
    "slack_send_notification",
    "email_send_smtp",
    "alert_send_pagerduty",
    "workspace_write_file",
    "workspace_read_file",
    "workspace_list",
]

print(f"Total tools in catalog: {len(cat)}")
missing = []
for s in scenarios:
    if s not in cat:
        close = [k for k in cat.keys() if any(part in k.lower() for part in s.split("_") if len(part) > 3)]
        missing.append((s, close[:5]))
    else:
        print(f"EXISTS: {s}")

print("\n--- HERRAMIENTAS QUE NO EXISTEN EN EL CATALOGO DE 400 HERRAMIENTAS ---")
for m, c in missing:
    print(f"FALTA: '{m}' -> Similares en catálogo: {c}")
