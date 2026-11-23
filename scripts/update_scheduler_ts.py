from pathlib import Path

path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\key-remtk\src\shared\constants\mcp\scheduler.mcp.ts")
content = path.read_text(encoding="utf-8")

old_desc = '"Programa una tarea del agente para el futuro: única (executeAt) o periódica (cron). " +'
new_desc = '"Programa un recordatorio, alarma, aviso o tarea periódica (cron job) del agente para el futuro: única (executeAt) o periódica (cron). " +'

old_tag = 'tags: ["scheduler", "cron", "recordatorio",'
new_tag = 'tags: ["scheduler", "cron", "cron job", "recordatorio", "alarma", "aviso",'

old_intent = 'intentSummary:\n      "Programa una tarea del agente (única o recurrente con cron): recordatorios, seguimientos, reportes y delegaciones autónomas",'
new_intent = 'intentSummary:\n      "Programa un recordatorio, alarma, aviso o tarea periódica (cron job) del agente: única (executeAt) o recurrente con cron",'

if old_desc in content:
    content = content.replace(old_desc, new_desc, 1)
    content = content.replace(old_tag, new_tag, 1)
    content = content.replace(old_intent, new_intent, 1)
    path.write_text(content, encoding="utf-8")
    print("SUCCESS")
else:
    print("NOT FOUND")
