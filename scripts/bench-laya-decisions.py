"""Prueba real de Laya (motor de decisiones) como SELECCIONADOR DE TOOL.

No es un modelo de embeddings: se usa la primitiva `choice` con las 232 tools del
catalogo real como criterios, sobre las MISMAS 15 consultas del banco de embeddings.
Se mide en dos modos:

  A) DIRECTO   : 1 sola pregunta `choice` con las 232 opciones (alta cardinalidad).
  B) SHORTLIST : recuperacion top-k por coseno sobre el propio encoder de Laya
                 (predict_shortlist + embed_fn_from_agent) y luego la cabeza de decision.

Fuente del catalogo: catalog-232.json (volcado de TOTAL_ACTIVE_TOOLS).
"""
import json
import time

from laya import Agent, predict_shortlist, embed_fn_from_agent, cached_embed_fn

CATALOG = json.load(open("catalog-232.json", encoding="utf-8"))
QUERIES = [
    ("lista las diapositivas activas del hero de la portada", "mitumbes_hero_obtener"),
    ("muestrame la configuracion actual de las diapositivas del hero de la portada", "mitumbes_hero_obtener"),
    ("obten todas las diapositivas del hero con su orden y estado", "mitumbes_hero_obtener"),
    ("cambia las diapositivas que se muestran en la portada principal", "mitumbes_hero_configurar"),
    ("configura las diapositivas del hero de la portada con nuevos items y lugares", "mitumbes_hero_configurar"),
    ("quiero mostrar otros eventos destacados en el carrusel de la pagina de eventos", "mitumbes_evento_hero_configurar"),
    ("programa que un evento destacado solo aparezca los fines de semana", "mitumbes_evento_hero_programar"),
    ("anade una nueva seccion al menu de navegacion de arriba", "mitumbes_navbar_configurar"),
    ("devuelve el menu de navegacion a su estado original", "mitumbes_navbar_restablecer"),
    ("crea un lugar turistico nuevo en el sistema", "mitumbes_lugar_crear"),
    ("lista todos los lugares registrados", "mitumbes_lugar_listar"),
    ("elimina definitivamente un aliado del directorio", "mitumbes_aliado_eliminar"),
    ("fija los mejores items al inicio de una categoria", "mitumbes_categoria_ordenar_items"),
    ("consulta cuantos clics y visitas han tenido los anuncios publicitarios", "mitumbes_publicidad_estadisticas"),
    ("adjunta una imagen a la ficha de un item", "mitumbes_item_imagen_adjuntar"),
]

CRITERIA = {c["name"]: c["description"] for c in CATALOG}  # 232 criterios
QUESTION = {
    "tool": {
        "type": "choice",
        "instructions": "Elige la UNICA herramienta (tool) que debe ejecutarse para atender la peticion del usuario.",
        "criteria": CRITERIA,
    }
}


def top1(results):
    return sum(1 for _, exp, got in results if got == exp) / len(results)


def run_direct(agent):
    print("\n===== A) DIRECTO: 1 pregunta choice con 232 opciones =====")
    rows, t_all = [], 0.0
    collapse_reports = []
    for q, expected in QUERIES:
        t = time.perf_counter()
        res = agent.system_one(q, QUESTION)
        dt = time.perf_counter() - t
        t_all += dt
        ans = res["answers"]["tool"]
        got = ans["choice"]
        rows.append((q, expected, got))
        if "options" in res["usage"]:
            collapse_reports.append((q, res["usage"]["options"]["tool"]))
        print(f"  {'OK ' if got == expected else 'X  '} exp={expected:42s} got={got:42s} conf={ans['answer_confidence']:.3f} {dt*1000:.0f}ms")
    print(f"  -> top1 = {top1(rows)*100:.0f}%  latencia media = {t_all/len(QUERIES)*1000:.0f}ms")
    if collapse_reports:
        q0, info = collapse_reports[0]
        print(f"  -> COLAPSO de opciones en el presupuesto de la cabeza: {info} (p.ej. '{q0[:30]}...')")
    else:
        print("  -> sin colapso reportado")
    return top1(rows)


def run_shortlist(agent, k):
    print(f"\n===== B) SHORTLIST k={k}: recuperacion coseno + cabeza choice =====")
    embed_fn = cached_embed_fn(embed_fn_from_agent(agent))
    rows, hits = [], 0
    t_all = 0.0
    for q, expected in QUERIES:
        t = time.perf_counter()
        res = predict_shortlist(agent, q, QUESTION, embed_fn, k=k)
        dt = time.perf_counter() - t
        t_all += dt
        labels = res["shortlist"]["tool"]["labels"]
        in_list = expected in labels
        hits += 1 if in_list else 0
        got = res["answers"]["tool"]["choice"]
        rows.append((q, expected, got))
        print(f"  {'OK ' if got == expected else 'X  '} exp={expected:42s} got={got:42s} en_top{k}={in_list} {dt*1000:.0f}ms")
    print(f"  -> recall@{k} = {hits}/{len(QUERIES)} ({hits/len(QUERIES)*100:.0f}%)  top1 = {top1(rows)*100:.0f}%  latencia media = {t_all/len(QUERIES)*1000:.0f}ms")
    return top1(rows)


if __name__ == "__main__":
    print(f"Catalogo: {len(CATALOG)} tools · Consultas: {len(QUERIES)}")
    t0 = time.perf_counter()
    agent = Agent("convaiinnovations/laya", subfolder="multilingual", device="cpu")
    print(f"Laya multilingual cargado en {time.perf_counter()-t0:.1f}s · device={agent.device} · max_len={agent.cfg.get('max_len')} head_max_len={agent.cfg.get('head_max_len')}")

    a = run_direct(agent)
    b20 = run_shortlist(agent, 20)
    b5 = run_shortlist(agent, 5)

    print("\n===== RESUMEN LAYA =====")
    print(f"  A) directo 232 opciones : top1 = {a*100:.0f}%")
    print(f"  B) shortlist k=20       : top1 = {b20*100:.0f}%")
    print(f"  B) shortlist k=5        : top1 = {b5*100:.0f}%")
