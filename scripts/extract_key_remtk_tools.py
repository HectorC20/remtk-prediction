import os
import re
import json

tools = {}
for root, dirs, files in os.walk(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\key-remtk\src"):
    for f in files:
        if f.endswith(".ts"):
            p = os.path.join(root, f)
            content = open(p, "r", encoding="utf-8", errors="ignore").read()
            matches = re.findall(r'name:\s*["\']([a-zA-Z0-9_\-]+)["\'].*?description:\s*["\']([^"\']+)["\']', content, re.DOTALL)
            for name, desc in matches:
                if name not in ["string", "number", "boolean", "array", "object", "id", "type", "error"]:
                    desc_clean = re.sub(r'\s+', ' ', desc).strip()
                    if name not in tools or len(desc_clean) > len(tools[name]["description"]):
                        tools[name] = {
                            "name": name,
                            "description": desc_clean,
                            "source_file": f,
                        }

out_path = r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts\key_remtk_extracted_tools.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(tools, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(tools)} tools into {out_path}")
