import json
import ast
import py_compile
from pathlib import Path

nb_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_colab.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)
print(f"Notebook valid JSON: {len(nb['cells'])} cells.")

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = "".join(c['source'])
        ast.parse(src)
        print(f"Cell {i} parsed successfully ({len(src.splitlines())} lines).")

py_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_3_8b_standalone.py")
py_compile.compile(str(py_path), doraise=True)
print("Standalone py script compiled cleanly without any SyntaxError.")
