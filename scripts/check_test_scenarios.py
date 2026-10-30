import json
from pathlib import Path

meta_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\models\remtk-juicio-jev-moe-3.8b-400tools\catalog_metadata.json")
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
catalog_tools = set(meta["catalog"].keys())

eval_script = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts\evaluate_3_8b_400tools_trained.py")
with open(eval_script, "r", encoding="utf-8") as f:
    code = f.read()

# Extraer TEST_SCENARIOS
import re
scenarios_match = re.search(r"TEST_SCENARIOS = \[(.*?)\]\n\n#", code, re.DOTALL)
if scenarios_match:
    scenarios_text = "[" + scenarios_match.group(1) + "]"
    scenarios = eval(scenarios_text)
    print(f"Total escenarios: {len(scenarios)}")
    missing = []
    for i, s in enumerate(scenarios):
        exp = s.get("expected_tool")
        if exp and exp != "ABSTAIN" and exp not in catalog_tools:
            missing.append((i+1, exp, s["query"]))
    print(f"Herramientas esperadas que NO EXISTEN en el catálogo (falsos negativos del test): {len(missing)}")
    for idx, exp, q in missing:
        print(f"  #{idx}: esperaba '{exp}' | query: '{q}'")
else:
    print("No se pudo extraer TEST_SCENARIOS")
