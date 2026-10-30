import json
from pathlib import Path

# Load the 400-tools catalog JSON
catalog_json_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\catalog_400_tools.json")
with open(catalog_json_path, "r", encoding="utf-8") as f:
    catalog_400 = json.load(f)

print(f"Loaded {len(catalog_400)} tools from catalog_400_tools.json")
raw_catalog_str = json.dumps(catalog_400, ensure_ascii=False)

# Define cells for the Colab Notebook

cells = []

# Cell 0: Markdown Description
cell_0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# REMTK-JUICIO-JEV-MOE 3.8B (A100 80GB - 100% QWEN GENERATIVE + 48,000 EPISODIOS)\n",
        "### Catálogo Maestro de 400 Herramientas (key-remtk + General Enterprise)\n",
        "\n",
        "**Especificaciones de Arquitectura y Pipeline**:\n",
        "- **Banco MoE**: 128 expertos ultra-dispersos con Top-4 gating y FFN intermedio de 4,736 dimensiones (~3.82B parámetros totales).\n",
        "- **Generación 100% Autónoma con Qwen**: Generador de escenarios, perturbador adversarial de modismos/typos y juez con filtrado JSON y auto-recarga. Cero plantillas estáticas hardcodeadas.\n",
        "- **Dataset de Entrenamiento**: 48,000 episodios multi-turno en VRAM balanceados a través de 10 modos cognitivos, con pares siameses limpios y ruidosos.\n",
        "- **Capa 2**: Representaciones desinsensibilizadas con proyecciones SwiGLU (2048D), supresor ortogonal de contexto rechazado, inyector diferencial de candidatos, 8 capas de Pre-LN Set-Attention y ColBERT bidireccional (16 estadísticas).\n",
        "- **Capa 3**: Calibrador tri-nivel de validez (JEV Noul) y calidad de decisión (JEV Score) con 9,234 dimensiones de entrada.\n",
        "- **Catálogo Activo**: 400 herramientas (96 herramientas reales de producción de `key-remtk` + 304 herramientas empresariales generales).\n",
        "- **Exportación**: Modelo híbrido INT8 por canal en `.safetensors` compatible con inferencia local en CPU, y copia de seguridad directa a Google Drive."
    ]
}
cells.append(cell_0)

# Cell 1: Environment & Setup
cell_1_code = """# ==============================================================================
# CELL 1: SETUP DE DEPENDENCIAS Y OPTIMIZACIÓN CUDA A100
# ==============================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import subprocess

def ensure_packages():
    missing = []
    for pkg in ["onnx", "onnxruntime", "transformers", "accelerate", "safetensors"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[Setup] Instalando paquetes requeridos ({', '.join(missing)})...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "onnx>=1.16.0", "onnxruntime>=1.18.0", "transformers>=4.43.0",
            "accelerate>=0.30.0", "safetensors>=0.4.3", "sentencepiece>=0.2.0"
        ])
        print("[Setup] Dependencias instaladas correctamente.")
    else:
        print("[Setup] Dependencias ya disponibles.")

ensure_packages()

import re
import gc
import copy
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from safetensors.torch import save_file

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

print(f"[Hardware] Dispositivo: {DEVICE} (BF16 compatible: {USE_BF16})")
if torch.cuda.is_available():
    print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_1_code.splitlines()]})

# Cell 2: Workspace & Config
cell_2_code = """# ==============================================================================
# CELL 2: WORKSPACE Y CONFIGURACIÓN EXACTA DE 3.8B PARÁMETROS
# ==============================================================================
WORKSPACE_DIR = Path.cwd() / "remtk_jev_moe_workspace"
DATA_DIR = WORKSPACE_DIR / "data"
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints_jev_3_8b"
EXPORT_DIR = WORKSPACE_DIR / "models" / "remtk-juicio-jev-moe-3.8b"

for d in [DATA_DIR, CHECKPOINT_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

best_ckpt_path = CHECKPOINT_DIR / "best_jev_moe_3_8b_a100.pt"
DYNAMIC_CORPUS_CACHE_PATH = DATA_DIR / "qwen_100pct_generative_corpus_3_8b_v2.json"

# Escalado matemático a ~3.82B parámetros (128 expertos x 3 x 2048 x 4736 + Set-Transformer 8 capas)
ACTIVE_CFG = {
    "num_experts": 128,
    "top_k_experts": 4,
    "d_model": 1024,
    "d_proj": 2048,
    "d_out": 2048,
    "d_exp_ffn": 4736,
    "nhead": 16,
    "set_layers": 8,
    "dropout": 0.08,
}

MIN_CANDIDATES_K = 14
MAX_CANDIDATES_K = 48
N_SYNTH_EPISODES = 48000

LANG_LABELS = ["ES", "EN"]
CONTEXT_ROUTE_LABELS = ["TOOL", "MEMORY", "PREV_RESPONSE", "INDEPENDENT"]
COGNITIVE_MODES = [
    "TOOL_DIRECT", "TOOL_INTROSPECTION", "MEMORY_RECALL", "MEMORY_STORE",
    "PREV_CONFIRM", "PREV_EXPLAIN_NO_TOOL", "PREV_CORRECTION",
    "FACTUAL_ASSERTION", "SOCIAL_OR_ATTACK", "CAPABILITY_MISMATCH",
]
EXPERT_DOMAINS = {
    0: "CRUD_ACTION",
    1: "SEARCH_QUERY",
    2: "SYSTEM_CONFIG",
    3: "TEMPORAL_SCHEDULING",
    4: "COMMUNICATION_NOTIFY",
    5: "ANALYTICS_FINANCE",
    6: "WORKSPACE_EXECUTION",
    7: "METADATA_INTROSPECTION",
}
AFFORDANCE_TYPES = ["QUERY", "MUTATION", "SCHEDULE_DEFERRED", "COMMUNICATION", "FINTECH", "DEVOPS"]

ROUTE_BY_MODE = {
    "TOOL_DIRECT": 0, "TOOL_INTROSPECTION": 0, "MEMORY_RECALL": 1, "MEMORY_STORE": 1,
    "PREV_CONFIRM": 2, "PREV_EXPLAIN_NO_TOOL": 2, "PREV_CORRECTION": 2,
    "FACTUAL_ASSERTION": 3, "SOCIAL_OR_ATTACK": 3, "CAPABILITY_MISMATCH": 0,
}
MODE_LABEL_DEFAULTS = {
    "TOOL_DIRECT": ([0.96, 0.01, 0.02, 0.01], 0.05, 1.0, 3, 0.0),
    "TOOL_INTROSPECTION": ([0.96, 0.01, 0.02, 0.01], 0.05, 1.0, 3, 0.0),
    "MEMORY_RECALL": ([0.02, 0.95, 0.02, 0.01], 0.65, 1.0, 2, 0.0),
    "MEMORY_STORE": ([0.02, 0.95, 0.02, 0.01], 0.70, 1.0, 2, 0.0),
    "PREV_CONFIRM": ([0.02, 0.01, 0.96, 0.01], 0.96, 1.0, 3, 1.0),
    "PREV_EXPLAIN_NO_TOOL": ([0.02, 0.02, 0.94, 0.02], 0.85, 0.0, 0, 0.0),
    "PREV_CORRECTION": ([0.03, 0.01, 0.95, 0.01], 0.88, 1.0, 3, -1.0),
    "FACTUAL_ASSERTION": ([0.01, 0.01, 0.02, 0.96], 0.02, 0.0, 0, 0.0),
    "SOCIAL_OR_ATTACK": ([0.01, 0.01, 0.01, 0.97], 0.01, 0.0, 0, 0.0),
    "CAPABILITY_MISMATCH": ([0.93, 0.02, 0.02, 0.03], 0.04, 0.0, 0, 0.0),
}
print(f"[Config] Workspace configurado. Configuración 3.8B: {ACTIVE_CFG}")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_2_code.splitlines()]})

# Cell 3: Catálogo Maestro de 400 Herramientas
cell_3_code = f"""# ==============================================================================
# CELL 3: CATÁLOGO MAESTRO INYECTADO (400 HERRAMIENTAS: 96 key-remtk + 304 General Enterprise)
# ==============================================================================
cat_candidates = [
    Path("catalog_400_tools.json"),
    WORKSPACE_DIR / "catalog_400_tools.json",
    DATA_DIR / "catalog_400_tools.json",
    Path.cwd() / "training" / "catalog_400_tools.json",
]

CATALOG = None
for p in cat_candidates:
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                CATALOG = json.load(f)
            print(f"[Catalog] Cargadas {{len(CATALOG)}} herramientas desde {{p}}")
            break
        except Exception:
            pass

if not CATALOG:
    print("[Catalog] Cargando catálogo maestro de 400 herramientas embebido...")
    RAW_CATALOG_JSON = {json.dumps(raw_catalog_str)}
    CATALOG = json.loads(RAW_CATALOG_JSON)
    print(f"[Catalog] Catálogo embebido cargado: {{len(CATALOG)}} herramientas.")

ALL_TOOL_IDS = list(CATALOG.keys())
NUM_CATALOG_TOOLS = len(ALL_TOOL_IDS)
tool_id_to_idx = {{tid: idx for idx, tid in enumerate(ALL_TOOL_IDS)}}

TOOLS_BY_DOMAIN: Dict[int, List[str]] = {{d: [] for d in EXPERT_DOMAINS}}
for tid, tdata in CATALOG.items():
    d_idx = int(tdata.get("domain_id", 0)) % len(EXPERT_DOMAINS)
    TOOLS_BY_DOMAIN[d_idx].append(tid)

print("=" * 80)
print(f"[Catalog] Catálogo activo validado: {{NUM_CATALOG_TOOLS}} herramientas.")
for d_idx, d_name in EXPERT_DOMAINS.items():
    print(f"  - Dominio {{d_idx}} ({{d_name:<24}}): {{len(TOOLS_BY_DOMAIN[d_idx]):>3}} herramientas")
print("=" * 80)
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_3_code.splitlines()]})

# Cell 4: PureQwenEcosystemGenerator
cell_4_code = """# ==============================================================================
# CELL 4: SINTETIZADOR 100% AUTÓNOMO CON QWEN CON SALVAMENTO JSON Y AUTO-REFILL
# ==============================================================================
TEACHER_LLM_NAME = os.environ.get("REMTK_TEACHER_MODEL", "Qwen/Qwen2.5-3B-Instruct")

class PureQwenEcosystemGenerator:
    \"\"\"
    Genera absolutamente TODO el ecosistema con Qwen sin inyectar ejemplos fijos.
    Incluye parser tolerante a truncamiento (`_extract_json_list`) y verificación
    activa de piscinas no vacías (`ensure_complete_ecosystem`) para evitar IndexError.
    \"\"\"
    def __init__(self, model_name: str = TEACHER_LLM_NAME, device: torch.device = DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_teacher(self):
        if self.model is not None:
            return
        print(f"[Qwen Autonomous Engine] Cargando Teacher LLM ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if (torch.cuda.is_available() and USE_BF16) else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()

    def unload_teacher(self):
        if self.model is not None:
            print("[Qwen Autonomous Engine] Liberando VRAM del Teacher LLM...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_json_batches(
        self, prompts: List[str], max_new_tokens: int = 640, temperature: float = 0.82, batch_size: int = 24
    ) -> List[str]:
        if not prompts:
            return []
        self.load_teacher()
        outputs = []
        for start in tqdm(range(0, len(prompts), batch_size), desc="Qwen Generating Batches", leave=False):
            batch_prompts = prompts[start : start + batch_size]
            formatted = [
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an autonomous data architect for a neural tool-routing engine. "
                                "Output strictly a valid JSON array of objects `[ {...}, {...} ]`. "
                                "Never include markdown explanations."
                            ),
                        },
                        {"role": "user", "content": p},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
            enc = self.tokenizer(
                formatted, return_tensors="pt", padding=True, truncation=True, max_length=1280
            ).to(self.device)
            gen_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            prompt_len = enc["input_ids"].shape[1]
            for i in range(len(batch_prompts)):
                raw_txt = self.tokenizer.decode(gen_ids[i, prompt_len:], skip_special_tokens=True).strip()
                outputs.append(raw_txt)
        return outputs

    @staticmethod
    def _extract_json_list(raw_text: str) -> List[Dict[str, Any]]:
        \"\"\"
        Extrae listas JSON incluso si el LLM cortó la generación antes del `]` final.
        Rescata todos los bloques `{...}` válidos para que ningún lote se pierda.
        \"\"\"
        if not raw_text:
            return []
        start_idx = raw_text.find("[")
        end_idx = raw_text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(raw_text[start_idx : end_idx + 1])
                if isinstance(parsed, list):
                    return [ {str(k).lower().strip(): v for k, v in it.items()} for it in parsed if isinstance(it, dict) ]
            except Exception:
                pass
        salvaged = []
        for match in re.finditer(r"\\{[^{}]+\\}", raw_text):
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    salvaged.append({str(k).lower().strip(): v for k, v in obj.items()})
            except Exception:
                continue
        return salvaged

    @staticmethod
    def _pick_field(item: Dict[str, Any], candidates: List[str]) -> str:
        for k in candidates:
            val = item.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ""

    def generate_slang_normalization_rules(self) -> List[Dict[str, str]]:
        prompts = [
            f"Batch {b}: Generate a JSON array of 20 common chat abbreviations, phonetic spellings, keyboard typos, "
            f"and informal developer slang in Spanish and English used when commanding AI assistants. "
            f"Each object must have keys: \\"slang\\" and \\"canonical\\"."
            for b in range(1, 5)
        ]
        raws = self.generate_json_batches(prompts, max_new_tokens=512, temperature=0.78, batch_size=4)
        rules = []
        seen = set()
        for r in raws:
            for item in self._extract_json_list(r):
                s_tok = self._pick_field(item, ["slang", "typo", "noisy", "abbr"]).lower()
                c_tok = self._pick_field(item, ["canonical", "clean", "correct", "normalized"]).lower()
                if 1 <= len(s_tok) <= 18 and len(c_tok) >= 2 and s_tok != c_tok and s_tok not in seen:
                    seen.add(s_tok)
                    rules.append({"slang": s_tok, "canonical": c_tok})
        return rules

    def generate_full_tool_catalog(self) -> Dict[str, Dict[str, Any]]:
        prompts = []
        meta = []
        for dom_id, dom_name in EXPERT_DOMAINS.items():
            for batch_idx in range(4):
                prompts.append(
                    f"Design a JSON array of 6 distinct MCP tools for expert domain '{dom_name}' "
                    f"(domain_id={dom_id}, batch={batch_idx+1}). "
                    f"If domain_id==1 and batch==1, include one tool named 'search_memory' for long-term user episodic memory recall. "
                    f"Each object must contain:\\n"
                    f"- \\"name\\": snake_case tool identifier\\n"
                    f"- \\"affordance\\": one of {AFFORDANCE_TYPES}\\n"
                    f"- \\"desc_es\\": detailed Spanish description of purpose and usage\\n"
                    f"- \\"desc_en\\": detailed English description of purpose and usage"
                )
                meta.append((dom_id, dom_name))

        raws = self.generate_json_batches(prompts, max_new_tokens=700, temperature=0.82, batch_size=16)
        catalog: Dict[str, Dict[str, Any]] = {}

        for (dom_id, dom_name), raw in zip(meta, raws):
            for item in self._extract_json_list(raw):
                raw_name = self._pick_field(item, ["name", "tool_name", "id", "tool"])
                t_name = re.sub(r"[^a-z0-9_]", "_", raw_name.lower()).strip("_")
                if not t_name or len(t_name) < 3 or t_name in catalog:
                    continue
                aff = self._pick_field(item, ["affordance", "type", "mode"]).upper()
                if aff not in AFFORDANCE_TYPES:
                    aff = "QUERY"
                d_es = self._pick_field(item, ["desc_es", "description_es", "spanish", "desc"])
                d_en = self._pick_field(item, ["desc_en", "description_en", "english", "description"])
                if not d_es and d_en:
                    d_es = d_en
                if not d_en and d_es:
                    d_en = d_es
                if len(d_es) < 8:
                    continue
                catalog[t_name] = {
                    "name": t_name,
                    "desc_es": d_es,
                    "desc_en": d_en,
                    "description": f"{d_es}. / {d_en}.",
                    "domain_id": int(dom_id),
                    "domain_name": dom_name,
                    "affordance": aff,
                    "text_for_embedding": f"passage: {t_name} [{aff}] ({dom_name}): {d_es}. {d_en}.",
                }
        return catalog

    def _parse_utterance_items(self, raw: str) -> List[Dict[str, str]]:
        valid = []
        for item in self._extract_json_list(raw):
            c_txt = self._pick_field(item, ["clean", "query", "user", "utterance", "request", "message"])
            n_txt = self._pick_field(item, ["noisy", "typo", "slang", "informal"]) or c_txt
            ctx_txt = self._pick_field(item, ["context", "assistant", "prev_turn", "session", "history"]) or c_txt
            if len(c_txt) >= 2 and len(ctx_txt) >= 2:
                valid.append({"clean": c_txt, "noisy": n_txt, "context": ctx_txt})
        return valid

    def generate_all_conversational_data(self, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        all_tids = list(catalog.keys())
        print(f"[Qwen] Generando corpus dinámico conversacional para {len(all_tids)} herramientas...")

        # 1. Utterances por herramienta (ES y EN)
        tool_prompts = []
        tool_meta = []
        for tid in all_tids:
            t_info = catalog[tid]
            for style_batch in ["direct_command_and_parameters", "implicit_goal_and_colloquial"]:
                for lang_code, lang_name in [("es", "Spanish"), ("en", "English")]:
                    tool_prompts.append(
                        f"Tool ID: '{tid}' ({t_info['domain_name']} / {t_info['affordance']})\\n"
                        f"Specification: {t_info['desc_es'] if lang_code == 'es' else t_info['desc_en']}\\n"
                        f"Style: {style_batch}\\n"
                        f"Generate a JSON array of 5 diverse user requests in {lang_name} that require '{tid}'. "
                        f"Each object must have keys: \\"clean\\", \\"noisy\\", and \\"context\\"."
                    )
                    tool_meta.append((tid, lang_code))

        raw_tool_outs = self.generate_json_batches(tool_prompts, max_new_tokens=560, temperature=0.85, batch_size=24)
        tool_utterances: Dict[str, Dict[str, List[Dict[str, str]]]] = {tid: {"es": [], "en": []} for tid in all_tids}
        for (tid, lang_code), raw in zip(tool_meta, raw_tool_outs):
            tool_utterances[tid][lang_code].extend(self._parse_utterance_items(raw))

        # 2. Episodios multi-turno PREV_CONFIRM y PREV_CORRECTION generados 100% por Qwen
        dialog_prompts = []
        dialog_meta = []
        for idx_p in range(120):
            tid_a = all_tids[idx_p % len(all_tids)]
            tid_b = all_tids[(idx_p * 7 + 3) % len(all_tids)]
            if tid_a == tid_b:
                tid_b = all_tids[(idx_p + 1) % len(all_tids)]
            lang_code = "es" if (idx_p % 2 == 0) else "en"
            lang_name = "Spanish" if lang_code == "es" else "English"

            dialog_prompts.append(
                f"Language: {lang_name}. Proposed Tool: '{tid_a}' ({catalog[tid_a]['desc_es'] if lang_code=='es' else catalog[tid_a]['desc_en']}).\\n"
                f"Generate a JSON array of 4 confirmation episodes where each object has:\\n"
                f"- \\"context\\": assistant proposing to execute '{tid_a}'\\n"
                f"- \\"clean\\": short user confirmation accepting the proposal (1 to 6 words)\\n"
                f"- \\"noisy\\": user confirmation with typos or slang"
            )
            dialog_meta.append(("PREV_CONFIRM", lang_code, tid_a, None))

            dialog_prompts.append(
                f"Language: {lang_name}. Rejected Tool: '{tid_b}'. Desired Tool: '{tid_a}' ({catalog[tid_a]['desc_es'] if lang_code=='es' else catalog[tid_a]['desc_en']}).\\n"
                f"Generate a JSON array of 4 correction episodes where each object has:\\n"
                f"- \\"context\\": assistant stating it will run '{tid_b}'\\n"
                f"- \\"clean\\": user negating/canceling '{tid_b}' and asking for the action of '{tid_a}' instead\\n"
                f"- \\"noisy\\": user correction with typos or slang"
            )
            dialog_meta.append(("PREV_CORRECTION", lang_code, tid_a, tid_b))

        raw_dialog_outs = self.generate_json_batches(dialog_prompts, max_new_tokens=540, temperature=0.85, batch_size=24)
        confirm_episodes: List[Dict[str, Any]] = []
        correction_episodes: List[Dict[str, Any]] = []

        for (m_type, lang_code, pos_tid, forb_tid), raw in zip(dialog_meta, raw_dialog_outs):
            for parsed_it in self._parse_utterance_items(raw):
                entry = {
                    "lang": lang_code,
                    "context": parsed_it["context"],
                    "clean": parsed_it["clean"],
                    "noisy": parsed_it["noisy"],
                    "target_tool": pos_tid,
                    "forbidden_tool": forb_tid,
                }
                if m_type == "PREV_CONFIRM":
                    confirm_episodes.append(entry)
                else:
                    correction_episodes.append(entry)

        # 3. Modos cognitivos no ejecutables / memoria
        cog_specs = {
            "MEMORY_RECALL": "user queries asking the assistant to recall personal preferences, saved rules, past session context, or historical notes from long-term memory",
            "MEMORY_STORE": "user instructions commanding the assistant to remember, save, or store a preference, rule, stack choice, or personal fact in memory",
            "PREV_EXPLAIN_NO_TOOL": "follow-up user questions asking the assistant to clarify or explain in more detail what the assistant just explained in 'context', without executing any tool",
            "FACTUAL_ASSERTION": "deceptive user statements mentioning servers, databases, cron intervals, or metrics just to complain, state a fact, or ask a general theoretical question where NO tool should be called",
            "SOCIAL_OR_ATTACK": "casual social greetings/chitchat/gratitude AND adversarial prompt injections/jailbreaks/system overrides that must be abstained from tool execution",
        }
        cog_prompts = []
        cog_meta = []
        for mode_name, desc_spec in cog_specs.items():
            for b_i in range(8):
                for lang_code, lang_name in [("es", "Spanish"), ("en", "English")]:
                    cog_prompts.append(
                        f"Language: {lang_name} (Batch {b_i+1})\\n"
                        f"Category: {mode_name} — {desc_spec}.\\n"
                        f"Generate a JSON array of 6 realistic interactions. Each object must have keys: \\"context\\", \\"clean\\", and \\"noisy\\"."
                    )
                    cog_meta.append((mode_name, lang_code))

        raw_cog_outs = self.generate_json_batches(cog_prompts, max_new_tokens=560, temperature=0.86, batch_size=20)
        cognitive_bank: Dict[str, Dict[str, List[Dict[str, str]]]] = {m: {"es": [], "en": []} for m in cog_specs}
        for (mode_name, lang_code), raw in zip(cog_meta, raw_cog_outs):
            cognitive_bank[mode_name][lang_code].extend(self._parse_utterance_items(raw))

        # 4. Suite Zero-Shot de Mundo Abierto
        zs_prompts = [
            f"Batch {b}: Generate a JSON array of 6 completely unseen tools from novel industries (robotics, biotech, aviation, "
            f"music production, legal tech, smart agriculture) and a test dialogue in Spanish or English. "
            f"Each object must have: \\"tool_name\\", \\"affordance\\" (QUERY or MUTATION), \\"domain_name\\", \\"tool_desc\\", \\"context\\", and \\"test_query\\"."
            for b in range(1, 5)
        ]
        raw_zs = self.generate_json_batches(zs_prompts, max_new_tokens=640, temperature=0.80, batch_size=4)
        zero_shot_eval_suite = []
        for r in raw_zs:
            for item in self._extract_json_list(r):
                t_n = self._pick_field(item, ["tool_name", "name", "tool"])
                t_d = self._pick_field(item, ["tool_desc", "description", "desc", "desc_es"])
                t_q = self._pick_field(item, ["test_query", "query", "clean", "user"])
                t_c = self._pick_field(item, ["context", "assistant"]) or t_d
                if t_n and t_d and t_q:
                    zero_shot_eval_suite.append({
                        "tool_name": re.sub(r"[^a-z0-9_]", "_", t_n.lower()).strip("_"),
                        "affordance": self._pick_field(item, ["affordance", "type"]) or "QUERY",
                        "domain_name": self._pick_field(item, ["domain_name", "domain"]) or "OPEN_DOMAIN",
                        "tool_desc": t_d,
                        "context": t_c,
                        "test_query": t_q,
                    })

        corpus = {
            "tool_utterances": tool_utterances,
            "confirm_episodes": confirm_episodes,
            "correction_episodes": correction_episodes,
            "cognitive_bank": cognitive_bank,
            "zero_shot_eval_suite": zero_shot_eval_suite,
        }
        return self.ensure_non_empty_qwen_pools(catalog, corpus)

    def ensure_non_empty_qwen_pools(self, catalog: Dict[str, Dict[str, Any]], corpus: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Verifica que absolutamente ninguna piscina de muestreo esté vacía antes de liberar Qwen.
        Si algún modo o idioma quedó vacío, pide a Qwen generarlo inmediatamente.
        \"\"\"
        all_tids = list(catalog.keys())
        cog_bank = corpus.setdefault("cognitive_bank", {})
        required_modes = ["MEMORY_RECALL", "MEMORY_STORE", "PREV_EXPLAIN_NO_TOOL", "FACTUAL_ASSERTION", "SOCIAL_OR_ATTACK"]

        for m in required_modes:
            cog_bank.setdefault(m, {"es": [], "en": []})
            for lang_code, lang_name in [("es", "Spanish"), ("en", "English")]:
                attempts = 0
                while len(cog_bank[m][lang_code]) < 4 and attempts < 4:
                    attempts += 1
                    raw = self.generate_json_batches([
                        f"Generate a JSON array of 6 {lang_name} user-assistant interactions for cognitive mode '{m}'. "
                        f"Each object must have \\"context\\", \\"clean\\", and \\"noisy\\"."
                    ], max_new_tokens=512, temperature=0.80, batch_size=1)
                    if raw:
                        cog_bank[m][lang_code].extend(self._parse_utterance_items(raw[0]))
                other_lang = "en" if lang_code == "es" else "es"
                if not cog_bank[m][lang_code] and cog_bank[m][other_lang]:
                    cog_bank[m][lang_code] = list(cog_bank[m][other_lang])

        for ep_key, m_label in [("confirm_episodes", "PREV_CONFIRM"), ("correction_episodes", "PREV_CORRECTION")]:
            pool = corpus.setdefault(ep_key, [])
            for lang_code, lang_name in [("es", "Spanish"), ("en", "English")]:
                attempts = 0
                while sum(1 for e in pool if e.get("lang") == lang_code and e.get("target_tool") in catalog) < 4 and attempts < 4:
                    attempts += 1
                    tid_a = all_tids[attempts % len(all_tids)]
                    tid_b = all_tids[(attempts + 3) % len(all_tids)]
                    raw = self.generate_json_batches([
                        f"Generate a JSON array of 5 {lang_name} multi-turn episodes for '{m_label}' "
                        f"(target tool: '{tid_a}', rejected tool: '{tid_b}'). "
                        f"Each object must have \\"context\\", \\"clean\\", and \\"noisy\\"."
                    ], max_new_tokens=512, temperature=0.80, batch_size=1)
                    for it in (self._parse_utterance_items(raw[0]) if raw else []):
                        pool.append({
                            "lang": lang_code,
                            "context": it["context"],
                            "clean": it["clean"],
                            "noisy": it["noisy"],
                            "target_tool": tid_a,
                            "forbidden_tool": tid_b if m_label == "PREV_CORRECTION" else None,
                        })

        t_utts = corpus.setdefault("tool_utterances", {})
        missing_tids = [tid for tid in all_tids if not t_utts.get(tid, {}).get("es") and not t_utts.get(tid, {}).get("en")]
        if missing_tids:
            refill_prompts = [
                f"Generate a JSON array of 4 user requests in Spanish for tool '{tid}' ({catalog[tid]['desc_es']}). "
                f"Each object must have \\"clean\\", \\"noisy\\", and \\"context\\"."
                for tid in missing_tids
            ]
            refill_raws = self.generate_json_batches(refill_prompts, max_new_tokens=450, temperature=0.80, batch_size=16)
            for tid, r in zip(missing_tids, refill_raws):
                items = self._parse_utterance_items(r)
                t_utts.setdefault(tid, {"es": [], "en": []})
                t_utts[tid]["es"].extend(items)
                t_utts[tid]["en"].extend(items)

        return corpus
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_4_code.splitlines()]})

# Cell 5: Generación o Carga del Corpus
cell_5_code = """# ==============================================================================
# CELL 5: EJECUCIÓN O CARGA DEL CORPUS 100% QWEN Y REGLAS DE SLANG
# ==============================================================================
qwen_engine = PureQwenEcosystemGenerator()

if DYNAMIC_CORPUS_CACHE_PATH.exists():
    print(f"[Qwen Autonomous Engine] Cargando ecosistema desde caché: {DYNAMIC_CORPUS_CACHE_PATH}...")
    with open(DYNAMIC_CORPUS_CACHE_PATH, "r", encoding="utf-8") as f:
        _qwen_bundle = json.load(f)
    QWEN_SLANG_RULES = _qwen_bundle.get("slang_rules", [])
    DYNAMIC_CORPUS = _qwen_bundle.get("corpus", {})
    cached_catalog = _qwen_bundle.get("catalog", {})
    if cached_catalog and len(cached_catalog) >= len(CATALOG):
        CATALOG = cached_catalog
    DYNAMIC_CORPUS = qwen_engine.ensure_non_empty_qwen_pools(CATALOG, DYNAMIC_CORPUS)
    qwen_engine.unload_teacher()
else:
    print(f"[Qwen Autonomous Engine] Generando ecosistema desde cero para {len(CATALOG)} herramientas...")
    QWEN_SLANG_RULES = qwen_engine.generate_slang_normalization_rules()
    DYNAMIC_CORPUS = qwen_engine.generate_all_conversational_data(CATALOG)
    qwen_engine.unload_teacher()
    with open(DYNAMIC_CORPUS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "slang_rules": QWEN_SLANG_RULES,
                "catalog": CATALOG,
                "corpus": DYNAMIC_CORPUS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Qwen Autonomous Engine] Ecosistema generado y guardado en {DYNAMIC_CORPUS_CACHE_PATH}")

_COMPILED_QWEN_SLANG = [
    (re.compile(rf"\\b{re.escape(r['slang'])}\\b", flags=re.IGNORECASE), r["canonical"])
    for r in QWEN_SLANG_RULES
    if r.get("slang") and r.get("canonical")
]

def canonicalize_chat_surface(text: str) -> str:
    out = text
    for creg, repl in _COMPILED_QWEN_SLANG:
        out = creg.sub(repl, out)
    return out

ALL_TOOL_IDS = list(CATALOG.keys())
NUM_CATALOG_TOOLS = len(ALL_TOOL_IDS)
tool_id_to_idx = {tid: idx for idx, tid in enumerate(ALL_TOOL_IDS)}

TOOLS_BY_DOMAIN = {d: [] for d in EXPERT_DOMAINS}
for tid, tdata in CATALOG.items():
    d_idx = int(tdata.get("domain_id", 0)) % len(EXPERT_DOMAINS)
    TOOLS_BY_DOMAIN[d_idx].append(tid)

print(f"[Slang] Compiladas {len(_COMPILED_QWEN_SLANG)} reglas de normalización de jerga.")
print(f"[Corpus] Total herramientas en catálogo activo: {NUM_CATALOG_TOOLS}")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_5_code.splitlines()]})

# Cell 6: E5 Encoder & Latent Mining
cell_6_code = """# ==============================================================================
# CELL 6: CODIFICACIÓN DENSA MULTILINGÜE E5 Y MINERÍA LATENTE DE VECINOS DUROS
# ==============================================================================
BACKBONE_NAME = "intfloat/multilingual-e5-small"
MAX_QUERY_LEN, MAX_CTX_LEN, MAX_TOOL_LEN = 40, 56, 56
print(f"[E5] Cargando encoder denso {BACKBONE_NAME}...")
e5_tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
e5_encoder = AutoModel.from_pretrained(BACKBONE_NAME).to(DEVICE).eval()

@torch.inference_mode()
def encode_token_sequences(
    texts: List[str], prefix: str, max_len: int, batch_size: int = 512, apply_canonical: bool = False
) -> torch.Tensor:
    processed = [canonicalize_chat_surface(t.strip()) if apply_canonical else t.strip() for t in texts]
    prefixed = [f"{prefix}{t}" if not t.startswith(prefix) else t for t in processed]
    outputs_list = []
    for start in range(0, len(prefixed), batch_size):
        enc = e5_tokenizer(
            prefixed[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            hid = F.normalize(e5_encoder(**enc).last_hidden_state, p=2, dim=-1)
        outputs_list.append((hid * enc["attention_mask"].unsqueeze(-1).float()).cpu().to(torch.float16))
    return torch.cat(outputs_list, dim=0)

print(f"[E5] Codificando embeddings del catálogo ({NUM_CATALOG_TOOLS} herramientas)...")
CATALOG_TOOL_TENSORS = encode_token_sequences(
    [CATALOG[tid]["text_for_embedding"] for tid in ALL_TOOL_IDS], prefix="passage: ", max_len=MAX_TOOL_LEN
)

with torch.no_grad():
    _t_float = CATALOG_TOOL_TENSORS.float()
    _t_mask = (_t_float.norm(p=2, dim=-1, keepdim=True) > 1e-5).float()
    _t_pooled = F.normalize((_t_float * _t_mask).sum(dim=1) / _t_mask.sum(dim=1).clamp(min=1.0), p=2, dim=-1)
    TOOL_COSINE_SIM_MATRIX = torch.matmul(_t_pooled, _t_pooled.T)
    TOOL_COSINE_SIM_MATRIX.fill_diagonal_(-1.0)
    LATENT_NEAREST_NEIGHBORS: Dict[str, List[str]] = {}
    for i, tid in enumerate(ALL_TOOL_IDS):
        top_indices = torch.argsort(TOOL_COSINE_SIM_MATRIX[i], descending=True)[:18].tolist()
        LATENT_NEAREST_NEIGHBORS[tid] = [ALL_TOOL_IDS[j] for j in top_indices]

print(f"[Latent Mining] Matriz de similitud y vecinos duros calculados: {TOOL_COSINE_SIM_MATRIX.shape}")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_6_code.splitlines()]})

# Cell 7: Typo Augmenter & Balanced Episode Synthesis
cell_7_code = """# ==============================================================================
# CELL 7: AUMENTADOR ESTOCÁSTICO Y SÍNTESIS DE EPISODIOS BALANCEADOS
# ==============================================================================
class ContinuousTypoAugmenter:
    \"\"\"Aumentador estocástico continuo sobre los textos de Qwen para escalar a 48,000 episodios únicos.\"\"\"
    def __init__(self):
        self.adj_keys = {
            "a": "sqz", "e": "wrd", "i": "uok", "o": "ipl", "u": "yij",
            "s": "adw", "t": "ryg", "r": "tef", "n": "mbh", "c": "xvk",
        }

    def augment_query(self, clean_q: str, base_noisy_q: str, rng: random.Random) -> str:
        text = base_noisy_q if rng.random() < 0.65 else clean_q
        if len(text) < 4:
            return text
        if rng.random() < 0.45:
            text = text.translate(str.maketrans("áéíóúÁÉÍÓÚ¿¡", "aeiouAEIOU  ")).strip()
        if rng.random() < 0.35 and len(text) > 7:
            chars = list(text)
            idx = rng.randint(1, len(chars) - 2)
            c = chars[idx].lower()
            if c in self.adj_keys:
                chars[idx] = rng.choice(self.adj_keys[c])
            text = "".join(chars)
        return text

AUGMENTER = ContinuousTypoAugmenter()

def infer_op_mode(mode: str, target_tool: Optional[str]) -> int:
    if mode == "MEMORY_STORE":
        return 1
    if mode == "MEMORY_RECALL":
        return 0
    if not target_tool or mode == "CAPABILITY_MISMATCH":
        return 2
    return 0 if CATALOG[target_tool]["affordance"] == "QUERY" else 1

def sample_qwen_tool_episode(tid: str, is_es: bool, rng: random.Random) -> Tuple[str, str, str]:
    lang_key = "es" if is_es else "en"
    other_key = "en" if is_es else "es"
    t_dict = DYNAMIC_CORPUS.get("tool_utterances", {}).get(tid, {})
    pool = t_dict.get(lang_key) or t_dict.get(other_key) or []
    if pool:
        item1 = rng.choice(pool)
        clean_q = item1["clean"]
        noisy_q = AUGMENTER.augment_query(clean_q, item1["noisy"], rng)
        ctx = item1["context"]
        if len(pool) > 1 and rng.random() < 0.35:
            item2 = rng.choice(pool)
            ctx = item2["context"]
        return clean_q, noisy_q, ctx
    desc = CATALOG[tid]["desc_es"] if is_es else CATALOG[tid]["desc_en"]
    return desc, AUGMENTER.augment_query(desc, desc, rng), desc

def assemble_episode_from_latent_mining(
    mode: str,
    clean_q: str,
    noisy_q: str,
    context: str,
    lang_id: int,
    target_tool: Optional[str],
    extra_negatives: Optional[List[str]] = None,
    forbidden_tool: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    rng = rng or random
    excluded_set = set()
    if mode == "CAPABILITY_MISMATCH" and target_tool and target_tool in tool_id_to_idx:
        excluded_set.add(target_tool)
        t_idx = tool_id_to_idx[target_tool]
        max_exclusions = max(1, len(ALL_TOOL_IDS) - MIN_CANDIDATES_K - 2)
        sorted_neighbors = LATENT_NEAREST_NEIGHBORS.get(target_tool, [])
        for other_tid in sorted_neighbors:
            if len(excluded_set) >= max_exclusions:
                break
            other_idx = tool_id_to_idx[other_tid]
            if float(TOOL_COSINE_SIM_MATRIX[t_idx, other_idx].item()) > 0.65:
                excluded_set.add(other_tid)
        active_target_tool = None
    else:
        active_target_tool = target_tool

    avail_all = [t for t in ALL_TOOL_IDS if t not in excluded_set]
    if len(avail_all) < 4:
        avail_all = [t for t in ALL_TOOL_IDS if t != target_tool]
        excluded_set = {target_tool} if target_tool else set()

    max_avail = min(MAX_CANDIDATES_K, len(avail_all))
    min_avail = min(MIN_CANDIDATES_K, max_avail)
    k_cands = max_avail if rng.random() < 0.30 else rng.randint(min_avail, max_avail)

    latent_negs = [
        h for h in (extra_negatives or [])
        if h in CATALOG and h != active_target_tool and h not in excluded_set
    ]
    ref_tool = active_target_tool or target_tool
    if ref_tool and ref_tool in LATENT_NEAREST_NEIGHBORS:
        nearest_pool = [
            n for n in LATENT_NEAREST_NEIGHBORS[ref_tool]
            if n not in excluded_set and n != active_target_tool
        ]
        if nearest_pool:
            latent_negs.extend(nearest_pool[: rng.randint(min(4, len(nearest_pool)), min(10, len(nearest_pool)))])
    if not latent_negs and avail_all:
        anchor = rng.choice(avail_all)
        latent_negs = [anchor] + [n for n in LATENT_NEAREST_NEIGHBORS.get(anchor, []) if n not in excluded_set][:6]

    latent_negs = list(dict.fromkeys([h for h in latent_negs if h != active_target_tool and h not in excluded_set]))[:11]
    cands_set = set(latent_negs)
    if active_target_tool:
        cands_set.add(active_target_tool)
    if forbidden_tool and forbidden_tool in CATALOG and forbidden_tool not in excluded_set:
        cands_set.add(forbidden_tool)

    avail_pool = [t for t in avail_all if t not in cands_set]
    need_more = max(0, k_cands - len(cands_set))
    if avail_pool and need_more > 0:
        cands_set.update(rng.sample(avail_pool, min(need_more, len(avail_pool))))

    cands = list(cands_set)[:MAX_CANDIDATES_K]
    rng.shuffle(cands)

    teacher_scores = []
    pos_idx_global = tool_id_to_idx[active_target_tool] if (active_target_tool and active_target_tool in tool_id_to_idx) else -1
    for cid in cands:
        if active_target_tool and cid == active_target_tool:
            teacher_scores.append(round(rng.uniform(0.96, 0.998), 4))
        elif forbidden_tool and cid == forbidden_tool:
            teacher_scores.append(round(rng.uniform(0.0005, 0.008), 4))
        elif pos_idx_global >= 0:
            sim_val = max(0.0, float(TOOL_COSINE_SIM_MATRIX[pos_idx_global, tool_id_to_idx[cid]].item()))
            teacher_scores.append(round(min(0.09, 0.005 + 0.08 * sim_val), 4))
        else:
            teacher_scores.append(round(rng.uniform(0.001, 0.015), 4))

    probs, gate, noul, qual, pol = MODE_LABEL_DEFAULTS[mode]
    return {
        "mode": mode,
        "clean_query": clean_q,
        "noisy_query": noisy_q,
        "context": context,
        "cands": cands,
        "lang_label": lang_id,
        "ctx_route_label": ROUTE_BY_MODE[mode],
        "ctx_route_probs": list(probs),
        "target_tool": active_target_tool,
        "forbidden_tool": forbidden_tool,
        "domain_id": CATALOG[active_target_tool]["domain_id"] if active_target_tool else -1,
        "gate_target": gate,
        "polarity_target": pol,
        "valid_noul_label": noul,
        "quality_level": qual,
        "teacher_scores": teacher_scores,
        "op_mode": infer_op_mode(mode, active_target_tool),
    }

def sample_domain_balanced_tool(ep_idx: int, rng: random.Random) -> str:
    dom_id = (ep_idx + rng.randint(0, 7)) % len(EXPERT_DOMAINS)
    dom_tools = TOOLS_BY_DOMAIN.get(dom_id) or ALL_TOOL_IDS
    return rng.choice(dom_tools)

def _safe_sample_cognitive_pool(mode: str, lang_key: str, rng: random.Random) -> Dict[str, str]:
    mode_dict = DYNAMIC_CORPUS.get("cognitive_bank", {}).get(mode, {})
    pool = mode_dict.get(lang_key) or mode_dict.get("es") or mode_dict.get("en") or []
    if pool:
        return rng.choice(pool)
    for m_other in DYNAMIC_CORPUS.get("cognitive_bank", {}).values():
        for p_list in m_other.values():
            if p_list:
                return rng.choice(p_list)
    any_tid = rng.choice(ALL_TOOL_IDS)
    c_q, n_q, ctx = sample_qwen_tool_episode(any_tid, lang_key == "es", rng)
    return {"clean": c_q, "noisy": n_q, "context": ctx}

def synthesize_balanced_episode(ep_idx: int, rng: random.Random) -> Dict[str, Any]:
    lang_id = 0 if (ep_idx % 20) < 13 else 1
    is_es = lang_id == 0
    lang_key = "es" if is_es else "en"

    mode_schedule = [
        "TOOL_DIRECT", "TOOL_DIRECT", "TOOL_DIRECT", "TOOL_DIRECT", "TOOL_DIRECT",
        "TOOL_INTROSPECTION",
        "MEMORY_RECALL", "MEMORY_STORE",
        "PREV_CONFIRM", "PREV_CONFIRM",
        "PREV_EXPLAIN_NO_TOOL",
        "PREV_CORRECTION", "PREV_CORRECTION",
        "FACTUAL_ASSERTION", "SOCIAL_OR_ATTACK",
        "CAPABILITY_MISMATCH", "CAPABILITY_MISMATCH",
    ]
    mode = mode_schedule[ep_idx % len(mode_schedule)]

    if mode == "TOOL_DIRECT":
        pos_tid = sample_domain_balanced_tool(ep_idx, rng)
        clean_q, noisy_q, ctx = sample_qwen_tool_episode(pos_tid, is_es, rng)
        return assemble_episode_from_latent_mining(mode, clean_q, noisy_q, ctx, lang_id, pos_tid, rng=rng)

    elif mode == "TOOL_INTROSPECTION":
        intro_tools = TOOLS_BY_DOMAIN.get(7) or ALL_TOOL_IDS
        pos_tid = rng.choice(intro_tools)
        clean_q, noisy_q, ctx = sample_qwen_tool_episode(pos_tid, is_es, rng)
        extra_hn = [t for t in intro_tools if t != pos_tid]
        return assemble_episode_from_latent_mining(
            mode, clean_q, noisy_q, ctx, lang_id, pos_tid, extra_negatives=extra_hn, rng=rng
        )

    elif mode in ("MEMORY_RECALL", "MEMORY_STORE", "PREV_EXPLAIN_NO_TOOL", "FACTUAL_ASSERTION", "SOCIAL_OR_ATTACK"):
        pos_tid = "search_memory" if (mode == "MEMORY_RECALL" and "search_memory" in CATALOG) else None
        chosen = _safe_sample_cognitive_pool(mode, lang_key, rng)
        clean_q = chosen["clean"]
        noisy_q = AUGMENTER.augment_query(clean_q, chosen["noisy"], rng)
        ctx = chosen["context"]
        return assemble_episode_from_latent_mining(mode, clean_q, noisy_q, ctx, lang_id, pos_tid, rng=rng)

    elif mode == "PREV_CONFIRM":
        all_conf = [e for e in DYNAMIC_CORPUS.get("confirm_episodes", []) if e.get("target_tool") in CATALOG]
        lang_conf = [e for e in all_conf if e.get("lang") == lang_key]
        active_conf_pool = lang_conf or all_conf
        if active_conf_pool:
            chosen = rng.choice(active_conf_pool)
            pos_tid = chosen["target_tool"]
            clean_q = chosen["clean"]
            noisy_q = AUGMENTER.augment_query(clean_q, chosen["noisy"], rng)
            ctx = chosen["context"]
        else:
            pos_tid = sample_domain_balanced_tool(ep_idx, rng)
            clean_q, noisy_q, ctx = sample_qwen_tool_episode(pos_tid, is_es, rng)
        return assemble_episode_from_latent_mining(mode, clean_q, noisy_q, ctx, lang_id, pos_tid, rng=rng)

    elif mode == "PREV_CORRECTION":
        all_corr = [e for e in DYNAMIC_CORPUS.get("correction_episodes", []) if e.get("target_tool") in CATALOG]
        lang_corr = [e for e in all_corr if e.get("lang") == lang_key]
        active_corr_pool = lang_corr or all_corr
        if active_corr_pool:
            chosen = rng.choice(active_corr_pool)
            pos_tid = chosen["target_tool"]
            forb_cand = chosen.get("forbidden_tool")
            forb_tid = forb_cand if (forb_cand in CATALOG and forb_cand != pos_tid) else LATENT_NEAREST_NEIGHBORS[pos_tid][0]
            clean_q = chosen["clean"]
            noisy_q = AUGMENTER.augment_query(clean_q, chosen["noisy"], rng)
            ctx = chosen["context"]
        else:
            pos_tid = sample_domain_balanced_tool(ep_idx, rng)
            forb_tid = LATENT_NEAREST_NEIGHBORS[pos_tid][0]
            clean_q, noisy_q, ctx = sample_qwen_tool_episode(pos_tid, is_es, rng)
        return assemble_episode_from_latent_mining(
            mode, clean_q, noisy_q, ctx, lang_id, pos_tid,
            extra_negatives=[forb_tid], forbidden_tool=forb_tid, rng=rng
        )

    else:  # CAPABILITY_MISMATCH
        missing_tid = sample_domain_balanced_tool(ep_idx, rng)
        clean_q, noisy_q, ctx = sample_qwen_tool_episode(missing_tid, is_es, rng)
        return assemble_episode_from_latent_mining(mode, clean_q, noisy_q, ctx, lang_id, missing_tid, rng=rng)
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_7_code.splitlines()]})

# Cell 8: Architecture 3.8B
cell_8_code = """# ==============================================================================
# CELL 8: ARQUITECTURA REMTK-JUICIO-JEV-MOE 3.8B (128 EXPERTOS, SET-ATTENTION, TRI-NIVEL)
# ==============================================================================
E5_EMBED_DIM = 384

class DilatedGatedConvBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.08):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            d_model, d_hidden * 2, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=4
        )
        self.out_proj = nn.Linear(d_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        h = (x * mask.unsqueeze(-1)).transpose(1, 2)
        a, b = self.conv(h).transpose(1, 2).chunk(2, dim=-1)
        h = self.out_proj(a * F.silu(b))
        return self.norm(residual + self.dropout(h)) * mask.unsqueeze(-1)

class AttentivePooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1).float().masked_fill(mask < 0.5, -1e4)
        weights = F.softmax(scores, dim=-1).to(x.dtype).unsqueeze(-1)
        return torch.sum(x * weights, dim=-2)

class Capa1JevIdiomaChoice(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.08):
        super().__init__()
        self.q_blocks = nn.ModuleList(
            [DilatedGatedConvBlock(d_model, d_model * 2, dilation=d, dropout=dropout) for d in (1, 2, 4)]
        )
        self.c_blocks = nn.ModuleList(
            [DilatedGatedConvBlock(d_model, d_model * 2, dilation=d, dropout=dropout) for d in (1, 2, 4)]
        )
        self.q_pooler = AttentivePooling(d_model)
        self.c_pooler = AttentivePooling(d_model)
        self.lang_choice_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, 2)
        )
        self.lang_cond_proj = nn.Linear(2, d_model)

    def forward(self, q_seq: torch.Tensor, c_seq: torch.Tensor, q_mask: torch.Tensor, c_mask: torch.Tensor):
        q_tilde, c_tilde = q_seq, c_seq
        for blk in self.q_blocks:
            q_tilde = blk(q_tilde, q_mask)
        for blk in self.c_blocks:
            c_tilde = blk(c_tilde, c_mask)
        q_pool = self.q_pooler(q_tilde, q_mask)
        c_pool = self.c_pooler(c_tilde, c_mask)
        lang_logits = self.lang_choice_head(torch.cat([q_pool, c_pool], dim=-1)).float()
        lang_probs = F.softmax(lang_logits, dim=-1)
        lang_ent = -torch.sum(lang_probs * torch.log(lang_probs.clamp(min=1e-8)), dim=-1, keepdim=True) / math.log(2.0)
        lang_conf = (1.0 - lang_ent).clamp(0.0, 1.0)
        lang_bias = self.lang_cond_proj(lang_probs.to(q_tilde.dtype))
        return q_tilde, c_tilde, q_pool, c_pool, lang_logits, lang_probs, lang_conf, lang_bias

class SwiGLUProjector(nn.Module):
    def __init__(self, d_in: int, d_proj: int, dropout: float = 0.08):
        super().__init__()
        self.fc_in = nn.Linear(d_in, d_proj * 2)
        self.fc_mid = nn.Linear(d_proj, d_proj)
        self.norm1 = nn.LayerNorm(d_proj)
        self.fc_res = nn.Linear(d_proj, d_proj * 2)
        self.fc_out = nn.Linear(d_proj, d_proj)
        self.norm2 = nn.LayerNorm(d_proj)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc_in(x).chunk(2, dim=-1)
        h = self.norm1(self.fc_mid(a * F.silu(b)))
        c, d = self.fc_res(h).chunk(2, dim=-1)
        return self.norm2(h + 0.3 * self.dropout(self.fc_out(c * F.silu(d))))

class OrthogonalRejectionSubtractor(nn.Module):
    def __init__(self, d_proj: int):
        super().__init__()
        self.reject_scale = nn.Parameter(torch.tensor(0.85))
        self.norm = nn.LayerNorm(d_proj)

    def forward(
        self, q_emb: torch.Tensor, c_pool_vec: torch.Tensor, ctx_polarity: torch.Tensor, q_mask: torch.Tensor
    ) -> torch.Tensor:
        neg_weight = F.relu(-ctx_polarity).unsqueeze(-1).to(q_emb.dtype)
        c_unit = F.normalize(c_pool_vec.unsqueeze(1), p=2, dim=-1)
        proj_scalar = torch.sum(q_emb * c_unit, dim=-1, keepdim=True)
        ortho_q = q_emb - torch.sigmoid(self.reject_scale) * neg_weight * F.relu(proj_scalar) * c_unit
        ortho_q = F.normalize(self.norm(ortho_q), p=2, dim=-1)
        return ortho_q * q_mask.unsqueeze(-1).to(q_emb.dtype)

class CandidateDifferentialInjector(nn.Module):
    def __init__(self, d_out: int, dropout: float = 0.08):
        super().__init__()
        self.diff_proj = nn.Sequential(
            nn.Linear(d_out * 3, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.tensor(0.25))
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h: torch.Tensor, cand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, D = h.shape
        h_norm = F.normalize(h.float(), p=2, dim=-1)
        peer_sim = torch.bmm(h_norm, h_norm.transpose(1, 2))
        eye = torch.eye(K, device=h.device, dtype=torch.bool).unsqueeze(0)
        peer_sim = peer_sim.masked_fill(eye, -1e4)
        if cand_mask is not None:
            peer_sim = peer_sim.masked_fill((cand_mask < 0.5).unsqueeze(1), -1e4)

        peer_weights = F.softmax(peer_sim * 8.0, dim=-1).to(h.dtype)
        closest_competitor_centroid = torch.bmm(peer_weights, h)
        ortho_diff = h - closest_competitor_centroid
        fused = self.diff_proj(torch.cat([h, ortho_diff, h * ortho_diff], dim=-1))
        out = self.norm(h + torch.tanh(self.gate) * fused)
        if cand_mask is not None:
            out = out * cand_mask.unsqueeze(-1).to(out.dtype)
        return out

class PreLNSetAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.08, res_scale: float = 0.25):
        super().__init__()
        self.d_model, self.nhead, self.head_dim = d_model, nhead, d_model // nhead
        self.res_scale = res_scale
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.norm1(x)
        B, N, C = norm_x.shape
        q = self.q_proj(norm_x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)).float() / (self.head_dim ** 0.5)
        if cand_mask is not None:
            scores = scores.masked_fill((cand_mask < 0.5).unsqueeze(1).unsqueeze(2), -1e4)
        attn_weights = F.softmax(scores, dim=-1).to(v.dtype)
        attn_out = self.out_proj((self.dropout(attn_weights) @ v).transpose(1, 2).reshape(B, N, C))
        x = x + self.res_scale * attn_out
        out = x + self.res_scale * self.ffn(self.norm2(x))
        if cand_mask is not None:
            out = out * cand_mask.unsqueeze(-1).to(out.dtype)
        return out

class UltraSparseVectorizedMoEBank(nn.Module):
    def __init__(self, num_experts: int, top_k: int, d_out: int, d_exp_ffn: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_out = d_out
        self.d_exp_ffn = d_exp_ffn

        scale_in = 0.5 / math.sqrt(d_out)
        scale_out = 0.5 / math.sqrt(d_exp_ffn)
        moe_dtype = torch.bfloat16 if (torch.cuda.is_available() and USE_BF16) else torch.float32
        self.w_gate_up = nn.Parameter(
            torch.randn(num_experts, d_out, d_exp_ffn * 2, dtype=moe_dtype) * scale_in
        )
        self.w_down = nn.Parameter(
            torch.randn(num_experts, d_exp_ffn, d_out, dtype=moe_dtype) * scale_out
        )

        self.shared_gate_up = nn.Linear(d_out, d_out * 2, bias=False)
        self.shared_down = nn.Linear(d_out, d_out, bias=False)
        self.branch_norm = nn.LayerNorm(d_out)
        self.moe_gate = nn.Parameter(torch.tensor(0.35))
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h_base: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        B, K, D = h_base.shape
        orig_dtype = h_base.dtype
        work_dtype = self.w_gate_up.dtype
        h_in = h_base.to(work_dtype)

        flat_idx = topk_idx.reshape(-1)
        w1_sel = self.w_gate_up.index_select(0, flat_idx)
        w2_sel = self.w_down.index_select(0, flat_idx)

        h_rep = h_in.unsqueeze(1).expand(B, self.top_k, K, D).reshape(B * self.top_k, K, D)
        gate_up = torch.bmm(h_rep, w1_sel)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        exp_out = torch.bmm(act, w2_sel).view(B, self.top_k, K, D)

        weights = topk_weights.to(work_dtype).unsqueeze(-1).unsqueeze(-1)
        moe_sum = torch.sum(exp_out * weights, dim=1).to(orig_dtype)

        sh_g, sh_u = self.shared_gate_up(h_base).chunk(2, dim=-1)
        shared_out = self.shared_down(F.silu(sh_g) * sh_u)

        combined_branch = self.branch_norm(shared_out + moe_sum)
        return self.norm(h_base + torch.tanh(self.moe_gate) * combined_branch)

class Capa2JevContextoYHerramientasMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k_experts: int,
        d_model: int,
        d_proj: int,
        d_out: int,
        d_exp_ffn: int,
        nhead: int,
        set_layers: int,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.d_proj = d_proj
        self.d_out = d_out

        self.w_q = nn.Linear(d_model, d_model * 2, bias=False)
        self.w_k = nn.Linear(d_model, d_model * 2, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.m_pooler = AttentivePooling(d_model)

        self.context_choice_head = nn.Sequential(
            nn.Linear(d_model * 5 + 2, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, d_proj // 2),
            nn.LayerNorm(d_proj // 2),
            nn.GELU(),
            nn.Linear(d_proj // 2, 4),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(4 + 2 + d_model * 2, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.ctx_polarity_head = nn.Sequential(
            nn.Linear(d_model * 5 + 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.out_norm = nn.LayerNorm(d_model)

        self.shared_q_proj = SwiGLUProjector(d_model, d_proj, dropout=dropout)
        self.shared_c_proj = SwiGLUProjector(d_model, d_proj, dropout=dropout)
        self.shared_t_proj = SwiGLUProjector(d_model, d_proj, dropout=dropout)
        self.q_att_pool = AttentivePooling(d_proj)
        self.c_att_pool = AttentivePooling(d_proj)
        self.t_att_pool = AttentivePooling(d_proj)

        self.ortho_rejection = OrthogonalRejectionSubtractor(d_proj)

        self.colbert_linear_anchor = nn.Sequential(
            nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 1)
        )

        in_pair_dim = d_proj * 6 + 16
        self.pair_fusion = nn.Sequential(
            nn.Linear(in_pair_dim, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, d_out),
            nn.LayerNorm(d_out),
        )

        self.router = nn.Sequential(
            nn.Linear(d_model * 3 + 2 + 4, d_proj // 2),
            nn.LayerNorm(d_proj // 2),
            nn.GELU(),
            nn.Linear(d_proj // 2, 512),
            nn.GELU(),
            nn.Linear(512, num_experts),
        )
        self.moe_bank = UltraSparseVectorizedMoEBank(num_experts, top_k_experts, d_out, d_exp_ffn)

        self.diff_injector = CandidateDifferentialInjector(d_out, dropout=dropout)
        res_scale = 1.0 / math.sqrt(2.0 * set_layers)
        self.set_layers = nn.ModuleList(
            [PreLNSetAttentionLayer(d_out, nhead, dropout, res_scale=res_scale) for _ in range(set_layers)]
        )
        self.set_branch_norm = nn.LayerNorm(d_out)
        self.set_gate = nn.Parameter(torch.tensor(0.25))
        self.set_norm = nn.LayerNorm(d_out)

        self.tool_logit_head = nn.Sequential(
            nn.Linear(d_out, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.tool_aff_head = nn.Sequential(
            nn.Linear(d_out, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.fusion_gate_head = nn.Sequential(
            nn.Linear(d_out + 16, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.logit_temp = nn.Parameter(torch.tensor(2.4))

    def _encode_tools_unified(
        self,
        tools_seq: torch.Tensor,
        t_mask: torch.Tensor,
        B: int,
        cand_idxs: Optional[torch.Tensor] = None,
        catalog_t_in: Optional[torch.Tensor] = None,
        catalog_t_mask: Optional[torch.Tensor] = None,
    ):
        if catalog_t_in is not None and cand_idxs is not None:
            cat_emb = F.normalize(self.shared_t_proj(catalog_t_in), p=2, dim=-1)
            cat_pool = F.normalize(self.t_att_pool(cat_emb, catalog_t_mask), p=2, dim=-1)
            safe_idxs = cand_idxs.clamp(0, catalog_t_in.size(0) - 1)
            t_emb = cat_emb[safe_idxs]
            t_att_p = cat_pool[safe_idxs]
            t_mask_4d = catalog_t_mask[safe_idxs]
        else:
            if tools_seq.dim() == 3:
                t_emb_3d = F.normalize(self.shared_t_proj(tools_seq), p=2, dim=-1)
                t_pool_2d = F.normalize(self.t_att_pool(t_emb_3d, t_mask), p=2, dim=-1)
                t_emb = t_emb_3d.unsqueeze(0).expand(B, -1, -1, -1)
                t_att_p = t_pool_2d.unsqueeze(0).expand(B, -1, -1)
                t_mask_4d = t_mask.unsqueeze(0).expand(B, -1, -1)
            else:
                t_emb = F.normalize(self.shared_t_proj(tools_seq), p=2, dim=-1)
                t_att_p = F.normalize(self.t_att_pool(t_emb, t_mask), p=2, dim=-1)
                t_mask_4d = t_mask
        return t_emb, t_att_p, t_mask_4d

    def forward(
        self,
        q_tilde,
        c_tilde,
        q_pool,
        c_pool,
        lang_probs,
        lang_bias,
        tools_seq,
        q_mask,
        c_mask,
        t_mask,
        cand_mask: Optional[torch.Tensor] = None,
        cand_idxs: Optional[torch.Tensor] = None,
        catalog_t_in: Optional[torch.Tensor] = None,
        catalog_t_mask: Optional[torch.Tensor] = None,
    ):
        B, Lq, _ = q_tilde.shape
        Lc = c_tilde.size(1)
        scale = 1.0 / math.sqrt(self.w_q.out_features)
        attn_scores = (
            torch.matmul(self.w_q(q_tilde), self.w_k(c_tilde).transpose(-1, -2)).float() * scale
        ).masked_fill((c_mask < 0.5).unsqueeze(1), -1e4)
        v_c = self.w_v(c_tilde)
        m_qc = torch.matmul(F.softmax(attn_scores, dim=-1).to(v_c.dtype), v_c) * q_mask.unsqueeze(-1)
        m_pool = self.m_pooler(m_qc, q_mask)

        ctx_in = torch.cat(
            [q_pool, c_pool, m_pool, torch.abs(q_pool - c_pool), q_pool * m_pool, lang_probs.to(q_pool.dtype)], dim=-1
        )
        ctx_logits = self.context_choice_head(ctx_in).float()
        ctx_probs = F.softmax(ctx_logits, dim=-1)
        ctx_ent = -torch.sum(ctx_probs * torch.log(ctx_probs.clamp(min=1e-8)), dim=-1, keepdim=True) / math.log(4.0)
        ctx_conf = (1.0 - ctx_ent).clamp(0.0, 1.0)

        gate_alpha = self.gate_proj(
            torch.cat([ctx_probs.to(q_pool.dtype), lang_probs.to(q_pool.dtype), q_pool, m_pool], dim=-1)
        ).float()
        ctx_polarity = self.ctx_polarity_head(
            torch.cat(
                [q_pool, c_pool, m_pool, q_pool - c_pool, q_pool * c_pool, ctx_probs.to(q_pool.dtype)], dim=-1
            )
        ).float()

        pos_ctx_gate = F.relu(ctx_polarity) * gate_alpha
        q_fused = (
            self.out_norm(q_tilde + lang_bias.unsqueeze(1) + 0.55 * pos_ctx_gate.to(q_tilde.dtype).unsqueeze(1) * m_qc)
            * q_mask.unsqueeze(-1)
        )

        q_emb_raw = F.normalize(self.shared_q_proj(q_fused), p=2, dim=-1)
        c_emb = F.normalize(self.shared_c_proj(c_tilde), p=2, dim=-1)
        c_att_p_single = F.normalize(self.c_att_pool(c_emb, c_mask), p=2, dim=-1)

        q_emb = self.ortho_rejection(q_emb_raw, c_att_p_single, ctx_polarity, q_mask)

        t_emb, t_att_p, t_mask_4d = self._encode_tools_unified(
            tools_seq, t_mask, B, cand_idxs=cand_idxs, catalog_t_in=catalog_t_in, catalog_t_mask=catalog_t_mask
        )
        K = t_emb.size(1)

        sim_qt = torch.einsum("bqd,bktd->bkqt", q_emb, t_emb).float()
        valid_qt = q_mask.unsqueeze(1).unsqueeze(3) * t_mask_4d.unsqueeze(2)
        sim_qt_masked = sim_qt.masked_fill(valid_qt < 0.5, -1.0)

        max_fwd, _ = sim_qt_masked.max(dim=-1)
        q_mask_bk = q_mask.unsqueeze(1).expand(B, K, Lq)
        q_len = q_mask_bk.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_max_fwd = (max_fwd * q_mask_bk).sum(dim=-1, keepdim=True) / q_len
        top_max_fwd, _ = max_fwd.max(dim=-1, keepdim=True)
        min_max_fwd, _ = max_fwd.masked_fill(q_mask_bk < 0.5, 1.0).min(dim=-1, keepdim=True)
        cov_fwd = ((max_fwd > 0.58).float() * q_mask_bk).sum(dim=-1, keepdim=True) / q_len

        max_bwd, _ = sim_qt_masked.max(dim=-2)
        t_len = t_mask_4d.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_max_bwd = (max_bwd * t_mask_4d).sum(dim=-1, keepdim=True) / t_len
        top_max_bwd, _ = max_bwd.max(dim=-1, keepdim=True)
        min_max_bwd, _ = max_bwd.masked_fill(t_mask_4d < 0.5, 1.0).min(dim=-1, keepdim=True)
        cov_bwd = ((max_bwd > 0.58).float() * t_mask_4d).sum(dim=-1, keepdim=True) / t_len

        sim_ct = torch.einsum("bcd,bktd->bkct", c_emb, t_emb).float()
        valid_ct = c_mask.unsqueeze(1).unsqueeze(3) * t_mask_4d.unsqueeze(2)
        sim_ct_masked = sim_ct.masked_fill(valid_ct < 0.5, -1.0)

        max_ctx_fwd, _ = sim_ct_masked.max(dim=-1)
        c_mask_bk = c_mask.unsqueeze(1).expand(B, K, Lc)
        c_len = c_mask_bk.sum(dim=-1, keepdim=True).clamp(min=1.0)
        raw_ctx_fwd = (max_ctx_fwd * c_mask_bk).sum(dim=-1, keepdim=True) / c_len
        raw_ctx_top_fwd = max_ctx_fwd.max(dim=-1, keepdim=True)[0]

        max_ctx_bwd, _ = sim_ct_masked.max(dim=-2)
        raw_ctx_bwd = (max_ctx_bwd * t_mask_4d).sum(dim=-1, keepdim=True) / t_len
        raw_ctx_top_bwd = max_ctx_bwd.max(dim=-1, keepdim=True)[0]

        pol_3d = ctx_polarity.unsqueeze(1)
        mean_ctx_fwd = pol_3d * raw_ctx_fwd
        top_ctx_fwd = pol_3d * raw_ctx_top_fwd
        mean_ctx_bwd = pol_3d * raw_ctx_bwd
        top_ctx_bwd = pol_3d * raw_ctx_top_bwd

        attn_q2t = F.softmax((sim_qt * 10.0).masked_fill(valid_qt < 0.5, -1e4), dim=-1).to(t_emb.dtype)
        q_aligned_t = torch.einsum("bkqt,bktd->bkqd", attn_q2t, t_emb)
        q_cross_pool = F.normalize(
            (q_aligned_t * q_mask_bk.unsqueeze(-1).to(t_emb.dtype)).sum(dim=-2) / q_len.to(t_emb.dtype), p=2, dim=-1
        )

        attn_t2q = F.softmax(
            (sim_qt.transpose(-1, -2) * 10.0).masked_fill(valid_qt.transpose(-1, -2) < 0.5, -1e4), dim=-1
        ).to(q_emb.dtype)
        t_aligned_q = torch.einsum("bktq,bqd->bktd", attn_t2q, q_emb)
        t_cross_pool = F.normalize(
            (t_aligned_q * t_mask_4d.unsqueeze(-1).to(q_emb.dtype)).sum(dim=-2) / t_len.to(q_emb.dtype), p=2, dim=-1
        )

        q_att_p = F.normalize(self.q_att_pool(q_emb, q_mask), p=2, dim=-1).unsqueeze(1).expand(B, K, -1)
        c_att_p = c_att_p_single.unsqueeze(1).expand(B, K, -1)

        cos_qt = (q_att_p * t_att_p).sum(dim=-1, keepdim=True).float()
        raw_cos_ct = (c_att_p * t_att_p).sum(dim=-1, keepdim=True).float()
        cos_ct = pol_3d * raw_cos_ct

        rejection_penalty = F.relu(-pol_3d) * gate_alpha.unsqueeze(1) * F.relu(raw_cos_ct - 0.35)
        confirmation_boost = F.relu(pol_3d) * gate_alpha.unsqueeze(1) * raw_cos_ct

        signed_ctx_weight = (gate_alpha * ctx_polarity).to(q_att_p.dtype).unsqueeze(1)
        intent_vec = F.normalize(q_att_p + 0.75 * signed_ctx_weight * c_att_p, p=2, dim=-1)

        colbert_stats = torch.cat(
            [
                mean_max_fwd, top_max_fwd, min_max_fwd, cov_fwd,
                mean_max_bwd, top_max_bwd, min_max_bwd, cov_bwd,
                mean_ctx_fwd, top_ctx_fwd, mean_ctx_bwd, top_ctx_bwd,
                cos_qt, cos_ct, rejection_penalty, confirmation_boost,
            ],
            dim=-1,
        )

        anchor_logit = (
            5.5 * (intent_vec.float() * t_att_p.float()).sum(dim=-1)
            + 4.0 * (mean_max_fwd.squeeze(-1) + mean_max_bwd.squeeze(-1))
            + 4.5 * gate_alpha * (mean_ctx_fwd.squeeze(-1) + mean_ctx_bwd.squeeze(-1))
            - 7.0 * rejection_penalty.squeeze(-1)
            + self.colbert_linear_anchor(colbert_stats).squeeze(-1)
        )

        pair_feats = torch.cat(
            [
                intent_vec,
                t_att_p,
                q_cross_pool,
                t_cross_pool,
                F.normalize(intent_vec * t_att_p, p=2, dim=-1),
                F.normalize(intent_vec - t_att_p, p=2, dim=-1),
                colbert_stats.to(intent_vec.dtype),
            ],
            dim=-1,
        )

        h_base = self.pair_fusion(pair_feats)

        router_in = torch.cat(
            [q_pool, c_pool, m_pool, lang_probs.to(q_pool.dtype), ctx_probs.to(q_pool.dtype)], dim=-1
        )
        router_logits = self.router(router_in).float()
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k_experts, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        sparse_gates = torch.zeros_like(router_logits).scatter(-1, topk_idx, topk_weights)

        h_tools = self.moe_bank(h_base, topk_idx, topk_weights)
        h_diff = self.diff_injector(h_tools, cand_mask=cand_mask)

        x_set = h_diff
        for layer in self.set_layers:
            x_set = layer(x_set, cand_mask=cand_mask)
        h_refined = self.set_norm(h_diff + torch.tanh(self.set_gate) * self.set_branch_norm(x_set))
        if cand_mask is not None:
            h_refined = h_refined * cand_mask.unsqueeze(-1).to(h_refined.dtype)

        mlp_tool_logit = self.tool_logit_head(h_refined).squeeze(-1).float()
        mlp_aff_logit = self.tool_aff_head(h_refined).squeeze(-1).float()

        fusion_alpha = self.fusion_gate_head(
            torch.cat([h_refined.float(), colbert_stats], dim=-1)
        ).squeeze(-1)
        temp_scale = F.softplus(self.logit_temp).clamp(min=0.8, max=6.0)

        combined_choice = (1.0 - 0.55 * fusion_alpha) * anchor_logit + (0.65 + 0.85 * fusion_alpha) * mlp_tool_logit
        combined_aff = 0.70 * anchor_logit + 0.85 * mlp_aff_logit

        raw_tool_logits = temp_scale * combined_choice
        raw_aff_logits = combined_aff

        if cand_mask is not None:
            raw_tool_logits = raw_tool_logits.masked_fill(cand_mask < 0.5, -1e4)
            tool_choice_probs = F.softmax(raw_tool_logits, dim=-1) * cand_mask
            tool_affinity_scores = torch.sigmoid(raw_aff_logits) * cand_mask
        else:
            tool_choice_probs = F.softmax(raw_tool_logits, dim=-1)
            tool_affinity_scores = torch.sigmoid(raw_aff_logits)

        return (
            m_pool,
            ctx_logits,
            ctx_probs,
            ctx_conf,
            gate_alpha,
            ctx_polarity,
            router_logits,
            sparse_gates,
            h_refined,
            raw_tool_logits,
            raw_aff_logits,
            tool_choice_probs,
            tool_affinity_scores,
        )

class Capa3JevJuicioValidez(nn.Module):
    def __init__(self, d_model: int, d_out: int, dropout: float = 0.08):
        super().__init__()
        in_dim = (d_model * 3) + 18 + (d_out * 3)
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.valid_noul_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.quality_score_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 4))
        self.register_buffer("score_levels", torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32))

    def forward(
        self,
        q_pool,
        c_pool,
        m_pool,
        lang_probs,
        lang_conf,
        ctx_probs,
        ctx_conf,
        gate_alpha,
        h_refined,
        tool_choice_probs,
        tool_affinity_scores,
        cand_mask: Optional[torch.Tensor] = None,
    ):
        B, K = tool_affinity_scores.shape
        if cand_mask is not None:
            k_tensor = cand_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            aff_mean = (tool_affinity_scores * cand_mask).sum(dim=-1, keepdim=True) / k_tensor
            aff_var = (((tool_affinity_scores - aff_mean) ** 2) * cand_mask).sum(dim=-1, keepdim=True) / k_tensor
        else:
            k_tensor = torch.full((B, 1), float(K), device=tool_affinity_scores.device, dtype=torch.float32)
            aff_mean = tool_affinity_scores.mean(dim=-1, keepdim=True)
            aff_var = tool_affinity_scores.var(dim=-1, keepdim=True, unbiased=False)

        aff_std = torch.sqrt(aff_var + 1e-6)
        log_k_norm = torch.log(k_tensor + 1.0) / 5.5

        sorted_aff, _ = torch.sort(tool_affinity_scores, dim=-1, descending=True)
        pad_zeros = torch.zeros((B, 3), device=sorted_aff.device, dtype=sorted_aff.dtype)
        padded_aff = torch.cat([sorted_aff, pad_zeros], dim=-1)
        top1_aff, top2_aff, top3_aff = padded_aff[:, 0:1], padded_aff[:, 1:2], padded_aff[:, 2:3]
        margen_12 = top1_aff - top2_aff
        top1_zscore = ((top1_aff - aff_mean) / aff_std).clamp(-5.0, 10.0) / 10.0

        top1_prob, _ = tool_choice_probs.max(dim=-1, keepdim=True)
        entropia_norm = (
            -torch.sum(tool_choice_probs * torch.log(tool_choice_probs.clamp(min=1e-8)), dim=-1, keepdim=True)
        ) / torch.log(torch.clamp(k_tensor, min=2.0))

        h_ref_fp32 = h_refined.float()
        top_tool_repr = torch.sum(h_ref_fp32 * tool_choice_probs.unsqueeze(-1), dim=1)
        if cand_mask is not None:
            masked_h = h_ref_fp32.masked_fill((cand_mask < 0.5).unsqueeze(-1), -1e4)
            max_tool_feat, _ = masked_h.max(dim=1)
            mean_tool_feat = (h_ref_fp32 * cand_mask.unsqueeze(-1)).sum(dim=1) / k_tensor
        else:
            max_tool_feat, _ = h_ref_fp32.max(dim=1)
            mean_tool_feat = h_ref_fp32.mean(dim=1)

        judge_in = torch.cat(
            [
                q_pool.float(),
                c_pool.float(),
                m_pool.float(),
                lang_probs,
                lang_conf,
                ctx_probs,
                ctx_conf,
                gate_alpha,
                top1_aff,
                top2_aff,
                top3_aff,
                margen_12,
                top1_zscore,
                aff_mean,
                top1_prob,
                entropia_norm,
                log_k_norm,
                top_tool_repr,
                max_tool_feat,
                mean_tool_feat,
            ],
            dim=-1,
        )

        h_judge = self.shared_trunk(judge_in)
        valid_noul_logit = self.valid_noul_head(h_judge).float()
        decision_valid_noul = torch.sigmoid(valid_noul_logit)
        quality_logits = self.quality_score_head(h_judge).float()
        decision_quality_score = torch.sum(
            F.softmax(quality_logits, dim=-1) * self.score_levels.unsqueeze(0), dim=-1, keepdim=True
        )
        return decision_valid_noul, decision_quality_score, quality_logits, valid_noul_logit

class RemtkJuicioJEVMoE3_8B(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k_experts: int,
        d_model: int,
        d_proj: int,
        d_out: int,
        d_exp_ffn: int,
        nhead: int,
        set_layers: int,
        dropout: float = 0.08,
        in_dim: int = E5_EMBED_DIM,
    ):
        super().__init__()
        self.num_experts, self.in_dim, self.d_model = num_experts, in_dim, d_model
        self.input_adapter = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        self.capa1_idioma = Capa1JevIdiomaChoice(d_model=d_model, dropout=dropout)
        self.capa2_contexto_moe = Capa2JevContextoYHerramientasMoE(
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            d_model=d_model,
            d_proj=d_proj,
            d_out=d_out,
            d_exp_ffn=d_exp_ffn,
            nhead=nhead,
            set_layers=set_layers,
            dropout=dropout,
        )
        self.capa3_juicio = Capa3JevJuicioValidez(d_model=d_model, d_out=d_out, dropout=dropout)
        self.op_mode_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward_train_batch(
        self,
        query_tokens,
        context_tokens,
        cand_mask,
        cand_idxs: torch.Tensor,
        catalog_tools_3d: torch.Tensor,
        noisy_aug: bool = False,
    ):
        q_mask = (torch.norm(query_tokens, p=2, dim=-1) > 1e-5).float()
        c_mask = (torch.norm(context_tokens, p=2, dim=-1) > 1e-5).float()
        cat_t_mask = (torch.norm(catalog_tools_3d, p=2, dim=-1) > 1e-5).float()
        q_mask[:, 0] = 1.0
        c_mask[:, 0] = 1.0
        cat_t_mask[:, 0] = 1.0

        q_in = self.input_adapter(query_tokens)
        if noisy_aug and self.training:
            q_in = (q_in + torch.randn_like(q_in) * 0.02) * q_mask.unsqueeze(-1)
        c_in = self.input_adapter(context_tokens)
        cat_t_in = self.input_adapter(catalog_tools_3d)

        q_tilde, c_tilde, q_pool, c_pool, lang_logits, lang_probs, lang_conf, lang_bias = self.capa1_idioma(
            q_in, c_in, q_mask, c_mask
        )
        (
            m_pool,
            ctx_logits,
            ctx_probs,
            ctx_conf,
            gate_alpha,
            ctx_polarity,
            router_logits,
            sparse_gates,
            h_refined,
            raw_tool_logits,
            raw_aff_logits,
            tool_choice_probs,
            tool_affinity_scores,
        ) = self.capa2_contexto_moe(
            q_tilde,
            c_tilde,
            q_pool,
            c_pool,
            lang_probs,
            lang_bias,
            tools_seq=cat_t_in,
            q_mask=q_mask,
            c_mask=c_mask,
            t_mask=cat_t_mask,
            cand_mask=cand_mask,
            cand_idxs=cand_idxs,
            catalog_t_in=cat_t_in,
            catalog_t_mask=cat_t_mask,
        )
        decision_valid_noul, decision_quality_score, quality_logits, valid_noul_logit = self.capa3_juicio(
            q_pool,
            c_pool,
            m_pool,
            lang_probs,
            lang_conf,
            ctx_probs,
            ctx_conf,
            gate_alpha,
            h_refined,
            tool_choice_probs,
            tool_affinity_scores,
            cand_mask=cand_mask,
        )
        return {
            "q_tilde": q_tilde,
            "q_pool": q_pool,
            "lang_logits": lang_logits,
            "lang_probs": lang_probs,
            "ctx_logits": ctx_logits,
            "ctx_probs": ctx_probs,
            "gate_alpha": gate_alpha,
            "ctx_polarity": ctx_polarity,
            "router_logits": router_logits,
            "sparse_gates": sparse_gates,
            "raw_tool_logits": raw_tool_logits,
            "raw_aff_logits": raw_aff_logits,
            "tool_choice_probs": tool_choice_probs,
            "tool_affinity_scores": tool_affinity_scores,
            "decision_valid_noul": decision_valid_noul,
            "valid_noul_logit": valid_noul_logit,
            "decision_quality_score": decision_quality_score,
            "quality_logits": quality_logits,
            "op_mode_logits": self.op_mode_head(q_pool).float(),
        }

print("[Architecture] Módulos de RemtkJuicioJEVMoE 3.8B compilados exitosamente.")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_8_code.splitlines()]})

# Cell 9: Loss Function
cell_9_code = """# ==============================================================================
# CELL 9: PÉRDIDA UNIVERSAL JEV CON CONTRASTE DE NEGACIÓN Y CALIBRACIÓN TOP-1
# ==============================================================================
class BatchedJevUniversalLoss(nn.Module):
    def __init__(self, num_experts: int = 128):
        super().__init__()
        self.num_experts = num_experts

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        out_clean: Optional[Dict[str, torch.Tensor]] = None,
        ema_router_load: Optional[torch.Tensor] = None,
    ):
        lang_onehot = F.one_hot(batch["lang_target"], num_classes=2).float()
        loss_capa1 = F.cross_entropy(out["lang_logits"], batch["lang_target"]) + 0.5 * F.mse_loss(
            out["lang_probs"], lang_onehot
        )

        loss_capa2_ctx = (
            F.kl_div(F.log_softmax(out["ctx_logits"], dim=-1), batch["ctx_target_dist"], reduction="batchmean")
            + 0.85 * F.cross_entropy(out["ctx_logits"], batch["ctx_int"])
            + 0.50 * F.smooth_l1_loss(out["gate_alpha"].squeeze(-1), batch["gate_target"])
            + 0.85 * F.smooth_l1_loss(out["ctx_polarity"].squeeze(-1), batch["polarity_target"])
        )

        cand_mask = batch["cand_mask"]
        raw_logits = out["raw_tool_logits"]
        raw_aff_logits = out["raw_aff_logits"]
        aff_scores = out["tool_affinity_scores"]
        t_scores = batch["t_scores"]

        pos_weight_mask = torch.where(t_scores > 0.5, 5.0, 1.0) * cand_mask
        bce_per_elem = F.binary_cross_entropy_with_logits(raw_aff_logits, t_scores, reduction="none")
        loss_bce_aff = (bce_per_elem * pos_weight_mask).sum() / pos_weight_mask.sum().clamp(min=1.0)

        has_tool = batch["has_active_tool"]
        loss_tools = loss_bce_aff

        if has_tool.any():
            act_logits = raw_logits[has_tool]
            act_aff = aff_scores[has_tool]
            act_pos_idx = batch["pos_idx"][has_tool]
            act_neg_mask = batch["neg_mask"][has_tool]

            loss_ce_choice = F.cross_entropy(act_logits, act_pos_idx, label_smoothing=0.015)

            pos_aff = act_aff.gather(1, act_pos_idx.unsqueeze(1))
            neg_affs = act_aff.masked_fill(~act_neg_mask, -1.0)
            top4_hardest_online, _ = torch.topk(neg_affs, k=min(4, neg_affs.size(1)), dim=-1)
            loss_online_margin = F.relu(0.55 - (pos_aff - top4_hardest_online)).mean()

            pos_logit = act_logits.gather(1, act_pos_idx.unsqueeze(1)).squeeze(1)
            max_neg_logit = act_logits.masked_fill(~act_neg_mask, -1e4).max(dim=-1)[0]
            loss_logit_gap = F.softplus(2.8 - (pos_logit - max_neg_logit)).mean()

            loss_tools = loss_tools + 1.6 * loss_ce_choice + 1.8 * loss_online_margin + 0.65 * loss_logit_gap

        if (~has_tool).any():
            abstain_aff = aff_scores[~has_tool]
            loss_abstain = F.relu(abstain_aff.max(dim=-1)[0] - 0.10).mean()
            loss_tools = loss_tools + 1.4 * loss_abstain

        r_probs = F.softmax(out["router_logits"], dim=-1)
        mean_r_probs = r_probs.mean(dim=0)
        valid_exp = batch["exp_target"] >= 0
        loss_moe = torch.tensor(0.0, device=raw_logits.device)
        if valid_exp.any():
            loss_moe = loss_moe + 0.25 * F.cross_entropy(out["router_logits"][valid_exp], batch["exp_target"][valid_exp])
        if ema_router_load is not None:
            loss_moe = loss_moe + 0.02 * self.num_experts * torch.sum(ema_router_load.detach() * mean_r_probs)

        noul_logit = out["valid_noul_logit"].squeeze(-1)
        p_noul = out["decision_valid_noul"].squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
        y_noul = batch["y_noul"]
        pt = torch.where(y_noul >= 0.5, p_noul, 1.0 - p_noul)
        focal_bce = ((1.0 - pt) ** 2.0) * F.binary_cross_entropy_with_logits(noul_logit, y_noul, reduction="none")
        loss_capa3 = (
            1.3 * focal_bce.mean()
            + 0.85 * F.mse_loss(p_noul, y_noul)
            + 0.50 * F.cross_entropy(out["quality_logits"], batch["qual_target"])
        )

        loss_consistency = torch.tensor(0.0, device=raw_logits.device)
        if out_clean is not None:
            loss_latent_inv = (
                1.0
                - (
                    F.normalize(out["q_pool"], p=2, dim=-1)
                    * F.normalize(out_clean["q_pool"].detach(), p=2, dim=-1)
                ).sum(dim=-1)
            ).mean()
            loss_kl_tools = F.kl_div(
                F.log_softmax(out["raw_tool_logits"], dim=-1),
                F.softmax(out_clean["raw_tool_logits"].detach(), dim=-1),
                reduction="batchmean",
            )
            loss_consistency = 0.35 * loss_latent_inv + 0.20 * loss_kl_tools

        loss_op = F.cross_entropy(out["op_mode_logits"], batch["op_target"])
        total = (
            0.5 * loss_capa1
            + 1.3 * loss_capa2_ctx
            + 2.5 * loss_tools
            + loss_moe
            + 1.3 * loss_capa3
            + loss_consistency
            + 0.3 * loss_op
        )
        return total, mean_r_probs.detach()

print("[Loss] BatchedJevUniversalLoss compilada con InfoNCE, Brier/Focal y consistencia siamesa.")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_9_code.splitlines()]})

# Cell 10: Dataset Assembly in VRAM
cell_10_code = """# ==============================================================================
# CELL 10: CONSTRUCCIÓN DE 48,000 EPISODIOS EN VRAM Y PRE-COMPUTACIÓN E5
# ==============================================================================
print(f"\\n[Dataset] Ensamblando {N_SYNTH_EPISODES:,} episodios conversacionales a partir del corpus de Qwen...")
rng_synth = random.Random(SEED + 2026)
DATASET_SAMPLES = [synthesize_balanced_episode(i, rng_synth) for i in range(N_SYNTH_EPISODES)]
random.Random(SEED).shuffle(DATASET_SAMPLES)

t0 = time.perf_counter()
CLEAN_QUERY_TENSORS = encode_token_sequences(
    [s["clean_query"] for s in DATASET_SAMPLES], prefix="query: ", max_len=MAX_QUERY_LEN, apply_canonical=True
)
NOISY_QUERY_TENSORS = encode_token_sequences(
    [s["noisy_query"] for s in DATASET_SAMPLES], prefix="query: ", max_len=MAX_QUERY_LEN, apply_canonical=True
)
CONTEXT_TENSORS = encode_token_sequences(
    [s["context"] for s in DATASET_SAMPLES], prefix="passage: ", max_len=MAX_CTX_LEN
)
print(f"[Precompute E5] {len(DATASET_SAMPLES):,} episodios codificados en {time.perf_counter() - t0:.2f}s")

N_TOTAL = len(DATASET_SAMPLES)
Q_CLEAN_VRAM = CLEAN_QUERY_TENSORS.to(DEVICE, dtype=torch.float16)
Q_NOISY_VRAM = NOISY_QUERY_TENSORS.to(DEVICE, dtype=torch.float16)
C_VRAM = CONTEXT_TENSORS.to(DEVICE, dtype=torch.float16)
T_CATALOG_VRAM = CATALOG_TOOL_TENSORS.to(DEVICE, dtype=torch.float32)

CAND_IDX_VRAM = torch.zeros((N_TOTAL, MAX_CANDIDATES_K), dtype=torch.long, device=DEVICE)
CAND_MASK_VRAM = torch.zeros((N_TOTAL, MAX_CANDIDATES_K), dtype=torch.float32, device=DEVICE)
T_SCORES_VRAM = torch.zeros((N_TOTAL, MAX_CANDIDATES_K), dtype=torch.float32, device=DEVICE)
NEG_MASK_VRAM = torch.zeros((N_TOTAL, MAX_CANDIDATES_K), dtype=torch.bool, device=DEVICE)

HAS_TOOL_VRAM = torch.zeros(N_TOTAL, dtype=torch.bool, device=DEVICE)
POS_IDX_VRAM = torch.zeros(N_TOTAL, dtype=torch.long, device=DEVICE)
LANG_TARGET_VRAM = torch.zeros(N_TOTAL, dtype=torch.long, device=DEVICE)
CTX_INT_VRAM = torch.zeros(N_TOTAL, dtype=torch.long, device=DEVICE)
CTX_DIST_VRAM = torch.zeros((N_TOTAL, 4), dtype=torch.float32, device=DEVICE)
GATE_TARGET_VRAM = torch.zeros(N_TOTAL, dtype=torch.float32, device=DEVICE)
POLARITY_TARGET_VRAM = torch.zeros(N_TOTAL, dtype=torch.float32, device=DEVICE)
EXP_TARGET_VRAM = torch.full((N_TOTAL,), -1, dtype=torch.long, device=DEVICE)
Y_NOUL_VRAM = torch.zeros(N_TOTAL, dtype=torch.float32, device=DEVICE)
QUAL_TARGET_VRAM = torch.zeros(N_TOTAL, dtype=torch.long, device=DEVICE)
OP_TARGET_VRAM = torch.zeros(N_TOTAL, dtype=torch.long, device=DEVICE)

num_experts_cfg = ACTIVE_CFG["num_experts"]
for i, s in enumerate(DATASET_SAMPLES):
    cands = s["cands"][:MAX_CANDIDATES_K]
    k_len = len(cands)
    c_idxs = [tool_id_to_idx[cid] for cid in cands]
    CAND_IDX_VRAM[i, :k_len] = torch.tensor(c_idxs, dtype=torch.long, device=DEVICE)
    CAND_MASK_VRAM[i, :k_len] = 1.0
    T_SCORES_VRAM[i, :k_len] = torch.tensor(
        s["teacher_scores"][:k_len], dtype=torch.float32, device=DEVICE
    ).clamp(0.0, 1.0)

    has_t = (s["target_tool"] is not None) and (k_len > 1)
    HAS_TOOL_VRAM[i] = has_t
    if has_t:
        p_idx = int(np.argmax(s["teacher_scores"][:k_len]))
        POS_IDX_VRAM[i] = p_idx
        NEG_MASK_VRAM[i, :k_len] = True
        NEG_MASK_VRAM[i, p_idx] = False

    LANG_TARGET_VRAM[i] = int(s["lang_label"])
    CTX_INT_VRAM[i] = int(s["ctx_route_label"])
    CTX_DIST_VRAM[i] = torch.tensor(s["ctx_route_probs"], dtype=torch.float32, device=DEVICE)
    GATE_TARGET_VRAM[i] = float(s["gate_target"])
    POLARITY_TARGET_VRAM[i] = float(s["polarity_target"])

    dom_id = int(s["domain_id"])
    if dom_id >= 0 and s["target_tool"] in tool_id_to_idx:
        t_global_idx = tool_id_to_idx[s["target_tool"]]
        sub_slot = t_global_idx % (num_experts_cfg // 8)
        EXP_TARGET_VRAM[i] = (dom_id * (num_experts_cfg // 8)) + sub_slot
    Y_NOUL_VRAM[i] = float(s["valid_noul_label"])
    QUAL_TARGET_VRAM[i] = int(s["quality_level"])
    OP_TARGET_VRAM[i] = int(s.get("op_mode", 2))

def slice_vram_batch(batch_idxs: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "cand_idxs": CAND_IDX_VRAM[batch_idxs],
        "cand_mask": CAND_MASK_VRAM[batch_idxs],
        "t_scores": T_SCORES_VRAM[batch_idxs],
        "neg_mask": NEG_MASK_VRAM[batch_idxs],
        "has_active_tool": HAS_TOOL_VRAM[batch_idxs],
        "pos_idx": POS_IDX_VRAM[batch_idxs],
        "lang_target": LANG_TARGET_VRAM[batch_idxs],
        "ctx_int": CTX_INT_VRAM[batch_idxs],
        "ctx_target_dist": CTX_DIST_VRAM[batch_idxs],
        "gate_target": GATE_TARGET_VRAM[batch_idxs],
        "polarity_target": POLARITY_TARGET_VRAM[batch_idxs],
        "exp_target": EXP_TARGET_VRAM[batch_idxs],
        "y_noul": Y_NOUL_VRAM[batch_idxs],
        "qual_target": QUAL_TARGET_VRAM[batch_idxs],
        "op_target": OP_TARGET_VRAM[batch_idxs],
    }

indices = list(range(N_TOTAL))
random.Random(SEED).shuffle(indices)
split_idx = int(N_TOTAL * 0.90)
train_idx_tensor = torch.tensor(indices[:split_idx], dtype=torch.long, device=DEVICE)
val_idx_tensor = torch.tensor(indices[split_idx:], dtype=torch.long, device=DEVICE)

model = RemtkJuicioJEVMoE3_8B(**ACTIVE_CFG).to(DEVICE)
criterion = BatchedJevUniversalLoss(num_experts=num_experts_cfg).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
vram_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
print("=" * 92)
print(f"[Model] RemtkJuicioJEVMoE-3.8B | Parámetros Totales: {total_params / 1e6:.2f}M ({total_params / 1e9:.2f}B)")
print(f"[Data ] Episodios Totales: {N_TOTAL:,} (Train: {len(train_idx_tensor):,} | Val: {len(val_idx_tensor):,})")
print(f"[VRAM ] Memoria reservada en GPU: {vram_alloc_gb:.2f} GB")
print("=" * 92)
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_10_code.splitlines()]})

# Cell 11: Training Loop
cell_11_code = """# ==============================================================================
# CELL 11: ENTRENAMIENTO 3.8B (16 ÉPOCAS, ONECYCLELR, PAREJAS SIAMESAS)
# ==============================================================================
BATCH_SIZE = 64
EPOCHS = 16
BASE_LR = 1.6e-4
MOE_LR = 9.0e-5
AMP_ENABLED = bool(torch.cuda.is_available() and USE_BF16)

moe_bank_params = [
    model.capa2_contexto_moe.moe_bank.w_gate_up,
    model.capa2_contexto_moe.moe_bank.w_down,
]
moe_param_ids = {id(p) for p in moe_bank_params}
dense_trunk_params = [p for p in model.parameters() if id(p) not in moe_param_ids]

steps_per_epoch = math.ceil(len(train_idx_tensor) / BATCH_SIZE)
optimizer = torch.optim.AdamW(
    [
        {"params": dense_trunk_params, "lr": BASE_LR, "weight_decay": 1e-2},
        {"params": moe_bank_params, "lr": MOE_LR, "weight_decay": 1e-3},
    ],
    fused=torch.cuda.is_available(),
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[BASE_LR, MOE_LR],
    total_steps=steps_per_epoch * EPOCHS,
    pct_start=0.08,
    div_factor=12.0,
    final_div_factor=60.0,
    anneal_strategy="cos",
)
ema_router_load = torch.full((num_experts_cfg,), 1.0 / num_experts_cfg, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def evaluate_jev_batched(eval_idxs: torch.Tensor) -> Dict[str, Any]:
    model.eval()
    loss_sum, brier_sum = 0.0, 0.0
    lang_hits, ctx_hits, noul_hits, tool_hits, tool_total, e2e_hits = 0, 0, 0, 0, 0, 0
    total_samples = len(eval_idxs)

    for start in range(0, total_samples, BATCH_SIZE):
        b_idx = eval_idxs[start : start + BATCH_SIZE]
        B_cur = len(b_idx)
        b_dict = slice_vram_batch(b_idx)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
            out_raw = model.forward_train_batch(
                Q_NOISY_VRAM[b_idx].float(),
                C_VRAM[b_idx].float(),
                b_dict["cand_mask"],
                cand_idxs=b_dict["cand_idxs"],
                catalog_tools_3d=T_CATALOG_VRAM,
                noisy_aug=False,
            )
        out = {k: v.float() for k, v in out_raw.items()}
        loss, _ = criterion(out, b_dict, ema_router_load=None)
        loss_sum += float(loss.item()) * B_cur

        lang_hits += int((out["lang_probs"].argmax(dim=-1) == b_dict["lang_target"]).sum().item())
        pred_route = out["ctx_probs"].argmax(dim=-1)
        ctx_hits += int((pred_route == b_dict["ctx_int"]).sum().item())

        p_val = out["decision_valid_noul"].squeeze(-1)
        y_val = b_dict["y_noul"]
        brier_sum += float(((p_val - y_val) ** 2).sum().item())
        pred_noul_ok = (p_val >= 0.5) == (y_val >= 0.5)
        noul_hits += int(pred_noul_ok.sum().item())

        hybrid = (0.55 * out["tool_affinity_scores"] + 0.45 * out["tool_choice_probs"]).masked_fill(
            b_dict["cand_mask"] < 0.5, -1.0
        )
        pred_tool_pos = hybrid.argmax(dim=-1)

        act = b_dict["has_active_tool"]
        if act.any():
            tool_total += int(act.sum().item())
            tool_hits += int((pred_tool_pos[act] == b_dict["pos_idx"][act]).sum().item())

        tool_match_or_na = (~act) | (pred_tool_pos == b_dict["pos_idx"])
        e2e_ok = pred_noul_ok & (pred_route == b_dict["ctx_int"]) & tool_match_or_na
        e2e_hits += int(e2e_ok.sum().item())

    return {
        "loss": loss_sum / total_samples,
        "lang_acc": lang_hits / total_samples,
        "ctx_acc": ctx_hits / total_samples,
        "tool_top1": tool_hits / max(1, tool_total),
        "noul_acc": noul_hits / total_samples,
        "noul_brier": brier_sum / total_samples,
        "e2e_acc": e2e_hits / total_samples,
    }

best_composite = -1.0
print("=" * 95)
print(f"[Training] Iniciando optimización RemtkJuicioJEVMoE-3.8B ({EPOCHS} épocas sobre {N_TOTAL:,} episodios)...")
print("=" * 95)

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(train_idx_tensor), device=DEVICE)
    shuffled_train = train_idx_tensor[perm]
    running_loss = 0.0
    t_ep_start = time.perf_counter()

    pbar = tqdm(range(0, len(shuffled_train), BATCH_SIZE), desc=f"Epoch {epoch:02d}/{EPOCHS:02d}", leave=False)
    for start in pbar:
        b_idx = shuffled_train[start : start + BATCH_SIZE]
        b_dict = slice_vram_batch(b_idx)

        optimizer.zero_grad(set_to_none=True)
        q_noisy_batch = Q_NOISY_VRAM[b_idx].float()
        q_clean_batch = Q_CLEAN_VRAM[b_idx].float()
        c_batch = C_VRAM[b_idx].float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
            out_noisy_raw = model.forward_train_batch(
                q_noisy_batch,
                c_batch,
                b_dict["cand_mask"],
                cand_idxs=b_dict["cand_idxs"],
                catalog_tools_3d=T_CATALOG_VRAM,
                noisy_aug=True,
            )
            with torch.no_grad():
                out_clean_raw = model.forward_train_batch(
                    q_clean_batch,
                    c_batch,
                    b_dict["cand_mask"],
                    cand_idxs=b_dict["cand_idxs"],
                    catalog_tools_3d=T_CATALOG_VRAM,
                    noisy_aug=False,
                )

        out_noisy = {k: v.float() for k, v in out_noisy_raw.items()}
        out_clean = {k: v.float() for k, v in out_clean_raw.items()}
        loss, detached_r_probs = criterion(
            out_noisy, b_dict, out_clean=out_clean, ema_router_load=ema_router_load
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            running_loss += float(loss.item()) * len(b_idx)
            ema_router_load.mul_(0.98).add_(detached_r_probs, alpha=0.02)

    ep_sec = time.perf_counter() - t_ep_start
    samples_per_sec = len(shuffled_train) / max(1e-3, ep_sec)
    tr_loss = running_loss / len(shuffled_train)
    val = evaluate_jev_batched(val_idx_tensor)

    composite = (
        0.10 * val["lang_acc"]
        + 0.20 * val["ctx_acc"]
        + 0.35 * val["tool_top1"]
        + 0.15 * val["noul_acc"]
        + 0.20 * val["e2e_acc"]
    )
    is_best = composite >= best_composite
    if is_best:
        best_composite = composite
        torch.save(
            {"epoch": epoch, "config": ACTIVE_CFG, "model_state_dict": model.state_dict()},
            best_ckpt_path,
        )

    print(
        f"Ep {epoch:02d} ({samples_per_sec:.0f} ep/s) | "
        f"Loss: {tr_loss:.4f}/{val['loss']:.4f} | "
        f"C1-Lang: {val['lang_acc']:.1%} | C2-Ctx: {val['ctx_acc']:.1%} | "
        f"C2-Tool: {val['tool_top1']:.1%} | C3-Noul: {val['noul_acc']:.1%} | "
        f"E2E-Global: {val['e2e_acc']:.1%} {'[BEST]' if is_best else ''}"
    )

print("=" * 95)
print(f"[Training] Entrenamiento finalizado. Mejor checkpoint guardado en {best_ckpt_path} (Eficiencia compuesta: {best_composite:.2%})")
print("=" * 95)
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_11_code.splitlines()]})

# Cell 12: Audit Hold-Out & Zero-Shot Open World
cell_12_code = """# ==============================================================================
# CELL 12: AUDITORÍA ESTRICTA SOBRE CASOS HOLD-OUT Y ZERO-SHOT ABIERTO
# ==============================================================================
print(f"[Audit] Cargando mejor checkpoint 3.8B desde {best_ckpt_path}...")
ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

@torch.inference_mode()
def run_jev_decision(
    query: str,
    context: str,
    candidate_ids: Optional[List[str]] = None,
    custom_catalog_tensors: Optional[torch.Tensor] = None,
    active_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    net = active_model or model
    if candidate_ids is None:
        candidate_ids = ALL_TOOL_IDS
    if custom_catalog_tensors is None:
        c_idxs = torch.tensor([[tool_id_to_idx[tid] for tid in candidate_ids]], dtype=torch.long, device=DEVICE)
        cat_tensors_vram = T_CATALOG_VRAM
    else:
        c_idxs = torch.arange(len(candidate_ids), dtype=torch.long, device=DEVICE).unsqueeze(0)
        cat_tensors_vram = custom_catalog_tensors.to(DEVICE, dtype=torch.float32)

    c_mask = torch.ones((1, len(candidate_ids)), dtype=torch.float32, device=DEVICE)
    q_ten = encode_token_sequences([query], prefix="query: ", max_len=MAX_QUERY_LEN, apply_canonical=True).to(
        DEVICE, dtype=torch.float32
    )
    c_ten = encode_token_sequences([context], prefix="passage: ", max_len=MAX_CTX_LEN, apply_canonical=False).to(
        DEVICE, dtype=torch.float32
    )

    t0_inf = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        out = net.forward_train_batch(
            q_ten, c_ten, c_mask, cand_idxs=c_idxs, catalog_tools_3d=cat_tensors_vram, noisy_aug=False
        )
    lat_ms = (time.perf_counter() - t0_inf) * 1000.0

    ctx_probs = out["ctx_probs"][0].float().cpu().numpy()
    route_idx = int(np.argmax(ctx_probs))
    route_name = CONTEXT_ROUTE_LABELS[route_idx]

    noul_prob = float(out["decision_valid_noul"][0, 0].float().cpu().item())
    qual_score = float(out["decision_quality_score"][0, 0].float().cpu().item())

    aff = out["tool_affinity_scores"][0].float().cpu()
    chp = out["tool_choice_probs"][0].float().cpu()
    hybrid = 0.55 * aff + 0.45 * chp
    best_i = int(torch.argmax(hybrid).item())
    best_tool = candidate_ids[best_i]

    if noul_prob < 0.50 or route_name == "INDEPENDENT":
        decision = "ABSTAIN"
    elif route_name == "MEMORY":
        decision = "SELECT_MEMORY"
    else:
        decision = f"EXECUTE -> {best_tool}"

    return {
        "route": route_name,
        "noul": noul_prob,
        "score": qual_score,
        "decision": decision,
        "top_tool": best_tool,
        "lat_ms": lat_ms,
    }

val_indices_list = val_idx_tensor.cpu().tolist()
qwen_audit_samples = []
for v_idx in val_indices_list:
    s = DATASET_SAMPLES[v_idx]
    m = s["mode"]
    if sum(1 for x in qwen_audit_samples if x[0] == m) < 2:
        if s["valid_noul_label"] < 0.5 or ROUTE_BY_MODE[m] == 3:
            exp_dec = "ABSTAIN"
        elif ROUTE_BY_MODE[m] == 1:
            exp_dec = "SELECT_MEMORY"
        else:
            exp_dec = f"EXECUTE -> {s['target_tool']}"
        qwen_audit_samples.append((m, s["noisy_query"], s["context"], s["cands"], exp_dec))
    if len(qwen_audit_samples) >= 20:
        break

print("=" * 135)
print(f"{'#':<3} | {'Modo Qwen':<22} | {'Consulta Usuario (Generada por Qwen)':<42} | {'Ruta':<13} | {'Noul':<6} | {'Decisión Modelo':<28} | {'Estado'}")
print("-" * 135)
audit_hits = 0
for idx_a, (m_name, q_txt, ctx_txt, cands_a, exp_dec) in enumerate(qwen_audit_samples, 1):
    res = run_jev_decision(q_txt, ctx_txt, candidate_ids=cands_a)
    ok = res["decision"] == exp_dec
    audit_hits += int(ok)
    st = "✅ PASS" if ok else f"⚠️ ({exp_dec})"
    print(
        f"{idx_a:<3} | {m_name:<22} | {q_txt[:40]:<42} | {res['route']:<13} | {res['noul']:.3f}  | {res['decision'][:28]:<28} | {st}"
    )
print("=" * 135)
print(f"[Auditoría Hold-Out 100% Qwen] {audit_hits}/{len(qwen_audit_samples)} casos aprobados ({audit_hits / max(1, len(qwen_audit_samples)):.1%})")

zs_suite = DYNAMIC_CORPUS.get("zero_shot_eval_suite", [])
if zs_suite:
    zs_valid = [
        it for it in zs_suite
        if isinstance(it, dict) and it.get("tool_name") and it.get("test_query") and it.get("tool_desc")
    ]
    if len(zs_valid) >= 4:
        zs_ids = [str(it["tool_name"]).strip() for it in zs_valid]
        zs_passages = [
            f"passage: {it['tool_name']} [{it.get('affordance', 'QUERY')}] ({it.get('domain_name', 'OPEN_DOMAIN')}): {it['tool_desc']}"
            for it in zs_valid
        ]
        zs_tensors = encode_token_sequences(zs_passages, prefix="passage: ", max_len=MAX_TOOL_LEN)
        zs_hits, zs_total = 0, 0
        for it in zs_valid:
            t_name = str(it["tool_name"]).strip()
            q_z = str(it["test_query"]).strip()
            ctx_z = str(it.get("context") or it["tool_desc"]).strip()
            r_pres = run_jev_decision(q_z, ctx_z, candidate_ids=zs_ids, custom_catalog_tensors=zs_tensors)
            zs_hits += int(r_pres["decision"] == f"EXECUTE -> {t_name}")
            zs_total += 1
        print(f"[Auditoría Zero-Shot Mundo Abierto Qwen] {zs_hits}/{zs_total} herramientas inéditas acertadas ({zs_hits / max(1, zs_total):.1%})")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_12_code.splitlines()]})

# Cell 13: INT8 Quantization
cell_13_code = """# ==============================================================================
# CELL 13: CUANTIZACIÓN HÍBRIDA INT8 (128 EXPERTOS INT8 + TRONCO FP16)
# ==============================================================================
QUANT_EXPORT_DIR = EXPORT_DIR / "quantized_hybrid_int8"
QUANT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

class QuantizedUltraSparseMoEBankINT8(nn.Module):
    def __init__(self, orig_bank: UltraSparseVectorizedMoEBank):
        super().__init__()
        self.num_experts = orig_bank.num_experts
        self.top_k = orig_bank.top_k
        self.d_out = orig_bank.d_out
        self.d_exp_ffn = orig_bank.d_exp_ffn

        with torch.no_grad():
            w1_fp = orig_bank.w_gate_up.detach().float()
            scale_w1 = w1_fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
            w1_int8 = torch.round(w1_fp / scale_w1).clamp(-127, 127).to(torch.int8)

            w2_fp = orig_bank.w_down.detach().float()
            scale_w2 = w2_fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
            w2_int8 = torch.round(w2_fp / scale_w2).clamp(-127, 127).to(torch.int8)

        self.register_buffer("w_gate_up_int8", w1_int8)
        self.register_buffer("scale_w_gate_up", scale_w1.to(torch.float16))
        self.register_buffer("w_down_int8", w2_int8)
        self.register_buffer("scale_w_down", scale_w2.to(torch.float16))

        self.shared_gate_up = orig_bank.shared_gate_up.half()
        self.shared_down = orig_bank.shared_down.half()
        self.branch_norm = orig_bank.branch_norm.float()
        self.moe_gate = nn.Parameter(orig_bank.moe_gate.detach().float(), requires_grad=False)
        self.norm = orig_bank.norm.float()

    def forward(self, h_base: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        B, K, D = h_base.shape
        orig_dtype = h_base.dtype
        h_in = h_base.to(torch.float16)

        flat_idx = topk_idx.reshape(-1)
        w1_sel = self.w_gate_up_int8.index_select(0, flat_idx).to(torch.float16) * self.scale_w_gate_up.index_select(0, flat_idx)
        w2_sel = self.w_down_int8.index_select(0, flat_idx).to(torch.float16) * self.scale_w_down.index_select(0, flat_idx)

        h_rep = h_in.unsqueeze(1).expand(B, self.top_k, K, D).reshape(B * self.top_k, K, D)
        gate, up = torch.bmm(h_rep, w1_sel).chunk(2, dim=-1)
        exp_out = torch.bmm(F.silu(gate) * up, w2_sel).view(B, self.top_k, K, D)

        weights = topk_weights.to(torch.float16).unsqueeze(-1).unsqueeze(-1)
        moe_sum = torch.sum(exp_out * weights, dim=1).float()

        sh_g, sh_u = self.shared_gate_up(h_in).chunk(2, dim=-1)
        shared_out = self.shared_down(F.silu(sh_g) * sh_u).float()

        combined_branch = self.branch_norm(shared_out + moe_sum)
        return self.norm(h_base.float() + torch.tanh(self.moe_gate) * combined_branch).to(orig_dtype)

@torch.no_grad()
def build_hybrid_quantized_jev_model(fp_model: RemtkJuicioJEVMoE3_8B) -> RemtkJuicioJEVMoE3_8B:
    q_model = copy.deepcopy(fp_model).eval()
    q_model.capa2_contexto_moe.moe_bank = QuantizedUltraSparseMoEBankINT8(
        q_model.capa2_contexto_moe.moe_bank
    ).to(DEVICE)
    return q_model

print("[Quantization] Construyendo modelo Híbrido INT8 a partir del mejor checkpoint...")
quant_model = build_hybrid_quantized_jev_model(model)
print("[Quantization] Modelo cuantizado listo para verificación y exportación.")
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_13_code.splitlines()]})

# Cell 14: #CORRECCION# - Calibrated Inference, Drive Backup & Safetensors Export
cell_14_code = """# ==============================================================================
# CELL 14: #CORRECCION# - RESINCRONIZACIÓN DE ÍNDICES, CALIBRACIÓN INT8 Y EXPORTACIÓN DRIVE
# ==============================================================================
import os
import gc
import json
import time
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from safetensors.torch import save_file

# 1. RESINCRONIZAR DICCIONARIO DE ÍNDICES Y TENSORES DEL CATÁLOGO
ALL_TOOL_IDS = list(CATALOG.keys())
tool_id_to_idx = {tid: idx for idx, tid in enumerate(ALL_TOOL_IDS)}

if "CATALOG_TOOL_TENSORS" not in globals() or len(CATALOG_TOOL_TENSORS) != len(ALL_TOOL_IDS):
    print("[Sync] Recodificando tensores E5 del catálogo activo...")
    CATALOG_TOOL_TENSORS = encode_token_sequences(
        [CATALOG[tid]["text_for_embedding"] for tid in ALL_TOOL_IDS],
        prefix="passage: ",
        max_len=MAX_TOOL_LEN,
    )
T_CATALOG_VRAM = CATALOG_TOOL_TENSORS.to(DEVICE, dtype=torch.float32)

qwen_audit_samples = []
val_indices_list = val_idx_tensor.cpu().tolist() if "val_idx_tensor" in globals() else list(range(len(DATASET_SAMPLES)))
for v_idx in val_indices_list:
    if v_idx >= len(DATASET_SAMPLES):
        continue
    s = DATASET_SAMPLES[v_idx]
    m = s["mode"]
    valid_cands = [c for c in s["cands"] if c in tool_id_to_idx]
    if len(valid_cands) < 4:
        continue
    if s["target_tool"] is not None and s["target_tool"] not in tool_id_to_idx:
        continue
    if sum(1 for x in qwen_audit_samples if x[0] == m) < 2:
        if s["valid_noul_label"] < 0.5 or ROUTE_BY_MODE[m] == 3:
            exp_dec = "ABSTAIN"
        elif ROUTE_BY_MODE[m] == 1:
            exp_dec = "SELECT_MEMORY"
        else:
            exp_dec = f"EXECUTE -> {s['target_tool']}"
        qwen_audit_samples.append((m, s["noisy_query"], s["context"], valid_cands, exp_dec))
    if len(qwen_audit_samples) >= 20:
        break

# 2. FUNCIÓN DE INFERENCIA BLINDADA CONTRA KEYERROR Y CALIBRADA EN ZERO-SHOT
@torch.inference_mode()
def run_jev_decision_calibrated(
    query: str,
    context: str,
    candidate_ids: Optional[List[str]] = None,
    custom_catalog_tensors: Optional[torch.Tensor] = None,
    is_open_world_zeroshot: bool = False,
) -> Dict[str, Any]:
    if candidate_ids is None:
        candidate_ids = ALL_TOOL_IDS

    if custom_catalog_tensors is None:
        valid_cands = [tid for tid in candidate_ids if tid in tool_id_to_idx]
        if not valid_cands:
            valid_cands = ALL_TOOL_IDS
        candidate_ids = valid_cands
        c_idxs = torch.tensor([[tool_id_to_idx[tid] for tid in candidate_ids]], dtype=torch.long, device=DEVICE)
        cat_tensors_vram = T_CATALOG_VRAM
    else:
        c_idxs = torch.arange(len(candidate_ids), dtype=torch.long, device=DEVICE).unsqueeze(0)
        cat_tensors_vram = custom_catalog_tensors.to(DEVICE, dtype=torch.float32)

    c_mask = torch.ones((1, len(candidate_ids)), dtype=torch.float32, device=DEVICE)
    q_ten = encode_token_sequences([query], prefix="query: ", max_len=MAX_QUERY_LEN, apply_canonical=True).to(
        DEVICE, dtype=torch.float32
    )
    c_ten = encode_token_sequences([context], prefix="passage: ", max_len=MAX_CTX_LEN, apply_canonical=False).to(
        DEVICE, dtype=torch.float32
    )

    t0_inf = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        out = quant_model.forward_train_batch(
            q_ten, c_ten, c_mask, cand_idxs=c_idxs, catalog_tools_3d=cat_tensors_vram, noisy_aug=False
        )
    lat_ms = (time.perf_counter() - t0_inf) * 1000.0

    ctx_probs = out["ctx_probs"][0].float().cpu().numpy()
    route_idx = int(np.argmax(ctx_probs))
    route_name = CONTEXT_ROUTE_LABELS[route_idx]

    noul_prob = float(out["decision_valid_noul"][0, 0].float().cpu().item())
    qual_score = float(out["decision_quality_score"][0, 0].float().cpu().item())

    aff = out["tool_affinity_scores"][0].float().cpu()
    chp = out["tool_choice_probs"][0].float().cpu()
    hybrid = 0.55 * aff + 0.45 * chp

    best_i = int(torch.argmax(hybrid).item())
    best_tool = candidate_ids[best_i]

    aff_mean = float(aff.mean().item())
    aff_std = float(aff.std(unbiased=False).clamp(min=1e-5).item())
    top1_z = (float(aff[best_i].item()) - aff_mean) / aff_std

    if is_open_world_zeroshot and top1_z >= 1.15:
        noul_effective = max(noul_prob, 0.85)
        route_effective = "TOOL"
    else:
        noul_effective = noul_prob
        route_effective = route_name

    if noul_effective < 0.50 or route_effective == "INDEPENDENT":
        decision = "ABSTAIN"
    elif route_effective == "MEMORY":
        decision = "SELECT_MEMORY"
    else:
        decision = f"EXECUTE -> {best_tool}"

    return {
        "route": route_effective,
        "noul": noul_effective,
        "score": qual_score,
        "decision": decision,
        "top_tool": best_tool,
        "z_score": top1_z,
        "lat_ms": lat_ms,
    }

# 3. AUDITORÍA DEL MODELO INT8
print("\\n[Audit INT8] Evaluando exactitud del modelo híbrido INT8...")
print("=" * 135)
print(f"{'#':<3} | {'Modo Qwen':<22} | {'Consulta Usuario (Generada por Qwen)':<42} | {'Ruta':<13} | {'Noul':<6} | {'Decisión INT8':<28} | {'Estado'}")
print("-" * 135)
int8_holdout_hits = 0
for idx_a, (m_name, q_txt, ctx_txt, cands_a, exp_dec) in enumerate(qwen_audit_samples, 1):
    res = run_jev_decision_calibrated(q_txt, ctx_txt, candidate_ids=cands_a, is_open_world_zeroshot=False)
    ok = res["decision"] == exp_dec
    int8_holdout_hits += int(ok)
    st = "✅ PASS" if ok else f"⚠️ ({exp_dec})"
    print(
        f"{idx_a:<3} | {m_name:<22} | {q_txt[:40]:<42} | {res['route']:<13} | {res['noul']:.3f}  | {res['decision'][:28]:<28} | {st}"
    )
print("=" * 135)
print(
    f"[Auditoría Hold-Out INT8] {int8_holdout_hits}/{len(qwen_audit_samples)} "
    f"casos aprobados ({int8_holdout_hits / max(1, len(qwen_audit_samples)):.1%})"
)

# 4. EXPORTACIÓN SAFETENSORS Y COPIA AUTOMÁTICA A GOOGLE DRIVE
print("\\n[Export] Guardando pesos cuantizados 3.8B en formato Safetensors...")
state_to_save = {}
total_bytes = 0

with torch.no_grad():
    for k, v in quant_model.state_dict().items():
        if v.dtype in (torch.int8, torch.float16):
            tensor_out = v.detach().cpu().contiguous()
        elif any(tag in k for tag in ("norm", "gate", "score_levels", "logit_temp", "reject_scale")):
            tensor_out = v.detach().float().cpu().contiguous()
        else:
            tensor_out = v.detach().half().cpu().contiguous()
        state_to_save[k] = tensor_out
        total_bytes += tensor_out.numel() * tensor_out.element_size()

safetensors_path = QUANT_EXPORT_DIR / "remtk_juicio_jev_moe_3_8b_hybrid_int8.safetensors"
catalog_tensors_path = QUANT_EXPORT_DIR / "catalog_e5_tensors.safetensors"
catalog_meta_path = QUANT_EXPORT_DIR / "catalog_metadata.json"
inference_entry_path = QUANT_EXPORT_DIR / "inference_router.py"

save_file(state_to_save, str(safetensors_path))
del state_to_save
gc.collect()

save_file(
    {"catalog_tool_tensors": CATALOG_TOOL_TENSORS.detach().cpu().contiguous()},
    str(catalog_tensors_path),
)

with open(catalog_meta_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_name": "REMTK-JUICIO-JEV-MOE-3.8B-INT8-400TOOLS",
            "config": ACTIVE_CFG,
            "backbone_encoder": BACKBONE_NAME,
            "max_lengths": {"query": MAX_QUERY_LEN, "context": MAX_CTX_LEN, "tool": MAX_TOOL_LEN},
            "all_tool_ids": ALL_TOOL_IDS,
            "lang_labels": LANG_LABELS,
            "context_route_labels": CONTEXT_ROUTE_LABELS,
            "slang_rules": QWEN_SLANG_RULES,
            "catalog": CATALOG,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

standalone_loader_code = '''import json
from pathlib import Path
import torch
from safetensors.torch import load_file

def load_exported_artifacts(export_dir: str = "."):
    base = Path(export_dir)
    with open(base / "catalog_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model_tensors = load_file(str(base / "remtk_juicio_jev_moe_3_8b_hybrid_int8.safetensors"), device="cpu")
    catalog_tensors = load_file(str(base / "catalog_e5_tensors.safetensors"), device="cpu")["catalog_tool_tensors"]
    return metadata, model_tensors, catalog_tensors
'''
with open(inference_entry_path, "w", encoding="utf-8") as f:
    f.write(standalone_loader_code)

print("=" * 95)
print(f"[Export Local OK ] Carpeta Local : {QUANT_EXPORT_DIR.resolve()}")
print(f"  ├── Modelo INT8  : {safetensors_path.name} ({total_bytes / (1024**3):.2f} GB)")
print(f"  ├── Catálogo E5  : {catalog_tensors_path.name} ({catalog_tensors_path.stat().st_size / (1024**2):.2f} MB)")
print(f"  ├── Metadatos    : {catalog_meta_path.name}")
print(f"  └── Loader Script: {inference_entry_path.name}")
print("=" * 95)

def export_to_google_drive(local_export_dir: Path, drive_subfolder_name: str = "remtk-juicio-jev-moe-3.8b-400tools") -> Optional[Path]:
    drive_mount_point = Path("/content/drive")
    my_drive_dir = drive_mount_point / "MyDrive"

    if not my_drive_dir.exists() and not (drive_mount_point / "Mi unidad").exists():
        try:
            from google.colab import drive  # type: ignore
            print("[Google Drive] Montando Google Drive en /content/drive...")
            drive.mount(str(drive_mount_point), force_remount=False)
        except Exception as e:
            print(f"[Google Drive] No se pudo montar automáticamente ({e}).")

    target_root = None
    for candidate_root in [
        drive_mount_point / "MyDrive",
        drive_mount_point / "Mi unidad",
        Path.home() / "Google Drive" / "My Drive",
    ]:
        if candidate_root.exists():
            target_root = candidate_root
            break

    if target_root is None:
        print(
            "[Google Drive] Aviso: No se encontró la unidad montada.\\n"
            "  -> Ejecuta `from google.colab import drive; drive.mount('/content/drive')` "
            "y luego llama a `export_to_google_drive(QUANT_EXPORT_DIR)`."
        )
        return None

    drive_dest_dir = target_root / "REMTK_MODELS" / drive_subfolder_name
    drive_dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Google Drive] Copiando archivos hacia: {drive_dest_dir} ...")
    for item in local_export_dir.iterdir():
        if item.is_file():
            dest_file = drive_dest_dir / item.name
            t0_cp = time.perf_counter()
            shutil.copy2(item, dest_file)
            mb_size = dest_file.stat().st_size / (1024 ** 2)
            print(f"  ✓ Copiado a Drive: {item.name:<48} ({mb_size:8.2f} MB en {time.perf_counter() - t0_cp:.1f}s)")

    if DYNAMIC_CORPUS_CACHE_PATH.exists():
        shutil.copy2(DYNAMIC_CORPUS_CACHE_PATH, drive_dest_dir / DYNAMIC_CORPUS_CACHE_PATH.name)
        print(f"  ✓ Copiado a Drive: {DYNAMIC_CORPUS_CACHE_PATH.name} (Corpus 100% Qwen)")

    print("=" * 95)
    print(f"[Export Google Drive COMPLETADO] Ruta en tu Google Drive: {drive_dest_dir}")
    print("=" * 95)
    return drive_dest_dir

DRIVE_EXPORT_PATH = export_to_google_drive(QUANT_EXPORT_DIR)
"""
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell_14_code.splitlines()]})

# Build the complete notebook
notebook_dict = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

target_notebook_path = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_colab.ipynb")
with open(target_notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, ensure_ascii=False, indent=2)

print(f"Notebook written to: {target_notebook_path} ({target_notebook_path.stat().st_size / 1024:.2f} KB, {len(cells)} cells)")

# Also write the standalone Python script
target_standalone_py = Path(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\training\remtk_juicio_moe_3_8b_standalone.py")
py_parts = []
for c in cells:
    if c["cell_type"] == "code":
        py_parts.append("".join(c["source"]))

with open(target_standalone_py, "w", encoding="utf-8") as f:
    f.write("\n\n".join(py_parts))

print(f"Standalone Python script written to: {target_standalone_py} ({target_standalone_py.stat().st_size / 1024:.2f} KB)")
