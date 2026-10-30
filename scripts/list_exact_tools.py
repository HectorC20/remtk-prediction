import json
from pathlib import Path

meta_p = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk-juicio-jev-moe-3.8b-400tools\catalog_metadata.json")
with open(meta_p, "r", encoding="utf-8") as f:
    cat = json.load(f)["catalog"]

for cat_name, kw in [
    ("Payment", "pay"),
    ("File", "file"),
    ("Redis", "redis"),
    ("Slack", "slack"),
    ("K8s", "k8s"),
    ("Database", "db"),
    ("WhatsApp", "what"),
]:
    tools = [k for k in cat.keys() if kw in k.lower()]
    print(f"[{cat_name}]: {tools}")
