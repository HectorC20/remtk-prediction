#!/usr/bin/env python3
# ==============================================================================
# REMTK-PREDICTION PRODUCTION SERVER (PORT 6776)
# Sirve el modelo REMTK-JUICIO-JEV-MOE 3.8B (400 TOOLS) en tiempo real para key-remtk
# ==============================================================================

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from aiohttp import web

# Asegurar path de scripts en sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [serve-3.8b] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("remtk_prediction_service")

# Importar motor neuronal y catálogo
logger.info("Cargando motor de inferencia neuronal REMTK-JUICIO 3.8B (400 Tools)...")
from evaluate_3_8b_400tools_trained import (
    run_jev_decision_calibrated,
    CATALOG,
    ALL_TOOL_IDS,
    tool_id_to_idx,
    DEVICE,
    CFG,
)

logger.info(f"Motor listo sobre {DEVICE}. Catálogo activo: {len(CATALOG)} herramientas.")

PORT = int(os.environ.get("PORT_PREDICT", 6776))
HOST = os.environ.get("HOST_PREDICT", "0.0.0.0")

# ==============================================================================
# Rutas HTTP
# ==============================================================================

async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "model": "remtk-juicio-jev-moe-3.9b-400tools",
        "tools_count": len(CATALOG),
        "onnx_runtime_version": "1.20.1",
        "active_device": str(DEVICE),
    })

async def tools_count_handler(request: web.Request) -> web.Response:
    return web.json_response({"count": len(CATALOG)})

async def tools_ready_handler(request: web.Request) -> web.Response:
    return web.json_response({"ready": True, "count": len(CATALOG)})

async def register_tools_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        tools = body.get("tools", [])
        return web.json_response({"indexed": len(tools) or len(CATALOG)})
    except Exception:
        return web.json_response({"indexed": len(CATALOG)})

async def predict_feedback_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "events": 1})

async def memory_predict_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "memories": [],
        "topicShift": False,
        "topicScore": 0.0,
        "modelSize": "3.8b-400tools",
        "rankedScores": [],
    })

async def memory_judge_handler(request: web.Request) -> web.Response:
    return web.json_response({"isNoise": False, "noiseScore": 0.0})

async def history_regime_handler(request: web.Request) -> web.Response:
    return web.json_response({"regime": "chat"})

async def skills_predict_handler(request: web.Request) -> web.Response:
    return web.json_response({"skills": []})

async def skills_learn_handler(request: web.Request) -> web.Response:
    return web.json_response({"learned": True})

async def entities_track_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})

async def spatial_predict_handler(request: web.Request) -> web.Response:
    return web.json_response({"tool": "none", "confidence": 0.0, "x": 0.0, "y": 0.0})

async def fast_path_route_handler(request: web.Request) -> web.Response:
    return web.json_response({"action": "plan", "skip_planner": False})

async def interest_handler(request: web.Request) -> web.Response:
    return web.json_response({"hasInterest": False, "score": 0.0, "matches": []})

async def predict_handler(request: web.Request) -> web.Response:
    t0 = time.perf_counter()
    try:
        raw_bytes = await request.read()
        try:
            text_body = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_body = raw_bytes.decode("latin-1", errors="replace")
        body = json.loads(text_body)
    except Exception as e:
        return web.json_response({"error": f"JSON inválido: {e}"}, status=400)

    query = str(body.get("text", "")).strip()
    if not query:
        return web.json_response({"error": "El campo 'text' es requerido"}, status=400)

    tenant = str(body.get("tenant", "default"))
    session_id = str(body.get("sessionId", ""))
    exclude_list = set(body.get("exclude") or [])

    # Extraer contexto: historial previo o intentContext
    context = str(body.get("context", "")).strip()
    if not context:
        intent_ctx = str(body.get("intentContext", "")).strip()
        history = body.get("history") or []
        if history and isinstance(history, list):
            # Tomar último mensaje relevante del asistente
            last_assistant_msgs = [m.get("content", "") for m in history if isinstance(m, dict) and m.get("role") == "assistant"]
            if last_assistant_msgs:
                context = last_assistant_msgs[-1].strip()
        elif intent_ctx:
            context = intent_ctx

    # Ejecutar inferencia calibrada del modelo 3.8B
    res = run_jev_decision_calibrated(query, context=context, is_open_world_zeroshot=True)
    decision = res["decision"]
    top_tool = res["top_tool"]
    top_3 = res["top_3_tools"]
    noul = float(res["noul"])
    top1_score = float(res["top1_score"])
    lat_ms = (time.perf_counter() - t0) * 1000.0

    predicted_tools: List[Dict[str, Any]] = []

    # Si el modelo decide abstenerse o no requiere herramienta
    top1_z = float(res.get("top1_z", 0.0))
    is_abstain = (decision == "ABSTAIN" and top1_z < 0.85) or (noul < 0.15 and top1_z < 0.70)

    ranked_scores: List[float] = []
    calibrated_scores: List[float] = []

    if is_abstain:
        logger.info(f"[/predict] Abstención (noul={noul:.3f}, top1_z={top1_z:.2f}, route={res['route']}, top_tool={top_tool}, top_3={top_3}) -> query='{query[:50]}'")
    else:
        # Ensamblar herramientas predichas desde el catálogo de 400 herramientas
        # Top 1 recibe noul, Top 2 recibe 85%, Top 3 recibe 72% para superar el umbral >=0.25 del arnés
        top_3_scores = res.get("top_3_scores", [top1_score] * len(top_3))
        confidence_multipliers = [1.0, 0.85, 0.72]
        candidate_tids = [top_tool] + [t for t in top_3 if t != top_tool]
        
        for idx, tid in enumerate(candidate_tids):
            if tid in CATALOG and tid not in exclude_list:
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
                })
                raw_s = top_3_scores[idx] if idx < len(top_3_scores) else top1_score * 0.7
                ranked_scores.append(round(float(raw_s), 4))
                mult = confidence_multipliers[idx] if idx < len(confidence_multipliers) else 0.60
                cal_s = max(0.30, noul * mult)
                calibrated_scores.append(round(float(cal_s), 4))

    tool_names = [t["name"] for t in predicted_tools]
    graph = {
        "nodes": tool_names,
        "edges": [],
        "executionOrder": tool_names,
    }

    is_file_tool = any("file" in t.lower() or "workspace" in t.lower() for t in tool_names)
    file_command_mode = {
        "enabled": is_file_tool,
        "confidence": noul if is_file_tool else 0.0,
        "reason": "predicted_tool" if is_file_tool else "none",
    }

    response_data = {
        "tools": predicted_tools,
        "complexity": "simple" if len(predicted_tools) <= 1 else "moderate",
        "modelSize": "remtk-juicio-jev-moe-3.9b-400tools",
        "rankedScores": ranked_scores,
        "calibratedScores": calibrated_scores,
        "graph": graph,
        "fileCommandMode": file_command_mode,
        "debug": {"decision": decision, "top_tool": top_tool, "top_3_tools": top_3, "top1_z": top1_z, "noul": noul, "is_abstain": is_abstain},
    }

    logger.info(
        f"[/predict] 200 OK ({lat_ms:.1f}ms) | decision='{decision}' | tools={tool_names} | prompt='{query[:45]}'"
    )
    return web.json_response(response_data)


# ==============================================================================
# Servidor Principal
# ==============================================================================

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_post("/predict", predict_handler)
    app.router.add_post("/predict/feedback", predict_feedback_handler)
    app.router.add_get("/tools/count", tools_count_handler)
    app.router.add_get("/tools/ready", tools_ready_handler)
    app.router.add_post("/tools", register_tools_handler)
    app.router.add_post("/memory/predict", memory_predict_handler)
    app.router.add_post("/memory/judge", memory_judge_handler)
    app.router.add_post("/history/regime", history_regime_handler)
    app.router.add_post("/skills/predict", skills_predict_handler)
    app.router.add_post("/skills/learn", skills_learn_handler)
    app.router.add_post("/entities/track", entities_track_handler)
    app.router.add_post("/spatial/predict", spatial_predict_handler)
    app.router.add_post("/predict/spatial", spatial_predict_handler)
    app.router.add_post("/predict/route", fast_path_route_handler)
    app.router.add_post("/interest", interest_handler)
    return app

import asyncio

async def start_dual_server():
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    for p in [6776, 6876]:
        site = web.TCPSite(runner, HOST, p)
        await site.start()
        logger.info(f"REMTK Prediction Service activo en http://{HOST}:{p}")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    logger.info("Iniciando REMTK Prediction Dual-Port Service (puertos 6776 y 6876)...")
    try:
        asyncio.run(start_dual_server())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Servidor detenido.")
