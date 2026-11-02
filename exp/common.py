"""공통: 모델 로드, 하이퍼파라미터, dtype 호환 패치."""
import importlib.util
import os
import sys
from pathlib import Path

# CUDA 가 초기화되기 전에 순서를 바꿔야 한다. 예: EXP_CUDA_ORDER="1,0" → 4070 Ti SUPER 가 cuda:0
if os.environ.get("EXP_CUDA_ORDER"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["EXP_CUDA_ORDER"]

import torch

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / "code"
sys.path.insert(0, str(CODE))

MODEL = os.environ.get("EXP_MODEL", str(ROOT / "models" / "Llama-3.1-8B-Instruct"))
LAYERS = [int(x) for x in os.environ.get("EXP_LAYERS", "13,14,15").split(",")]
DTYPE = os.environ.get("EXP_DTYPE", "bf16")
SHARD = os.environ.get("EXP_SHARD", "1") == "1"
NAMES = [f"model.layers.{l}.mlp.down_proj.weight" for l in LAYERS]


def device_map(n_layers):
    """13-15층을 GPU0에 두고 뒤쪽만 GPU1로. tied weight 충돌 방지로 embed/head/norm 은 GPU0."""
    m = {"model.embed_tokens": "cuda:0", "lm_head": "cuda:0",
         "model.norm": "cuda:0", "model.rotary_emb": "cuda:0"}
    split = min(21, max(1, n_layers // 2 + 5))
    for i in range(split):
        m[f"model.layers.{i}"] = "cuda:0"
    for i in range(split, n_layers):
        m[f"model.layers.{i}"] = "cuda:1"
    return m


def load_model():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    dm = device_map(cfg.num_hidden_layers) if SHARD else {"": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if DTYPE == "bf16" else torch.float32,
        attn_implementation="sdpa",
        device_map=dm,
        local_files_only=True,
    ).eval()
    return tok, model


def make_hparams():
    spec = importlib.util.spec_from_file_location(
        "AlphaEdit_hparams", str(CODE / "AlphaEdit" / "AlphaEdit_hparams.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hp = mod.AlphaEditHyperParams.from_json(str(CODE / "config" / "alphaedit-hparams.json"))
    hp.layers = LAYERS
    hp.L2 = 0.1
    hp.v_num_grad_steps = 50
    return hp


def identity_projector(device, dim):
    return [torch.eye(dim, dtype=torch.float32, device=device) for _ in LAYERS]


def prepare():
    """cwd 고정 + dtype 패치 + 모델/hp 로드."""
    os.chdir(CODE)
    import patches
    patches.install()
    tok, model = load_model()
    hp = make_hparams()
    hp.v_loss_layer = model.config.num_hidden_layers - 1
    projector = identity_projector(next(model.parameters()).device, model.config.intermediate_size)
    return tok, model, hp, projector
