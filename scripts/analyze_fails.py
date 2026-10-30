import json
from pathlib import Path

p = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts\benchmark_3_8b_trained_final.json")
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

fails = [r for r in data["results"] if not r["pass"]]
print(f"Total fails: {len(fails)} de {data['total_cases']}")
for f in fails:
    print(f"\n#{f['id']} | Query: \"{f['query']}\"")
    print(f"   Context: \"{f.get('context', '')}\"")
    print(f"   Expected: {f['expected_tool']} | Exp Dec: {f['expected_decision']}")
    print(f"   Predicted: {f['predicted_tool']} | Top3: {f['predicted_top3']}")
    print(f"   Route: {f['route']} | Noul: {f['noul']:.3f} | Decision: {f['decision']}")
