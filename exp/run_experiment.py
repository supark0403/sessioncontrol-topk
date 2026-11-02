"""B파트 라우팅 실측: 8세션 뱅크에서 1단(전수 에너지) vs 2단(키 Top-N + 에너지).

원본과 다른 점(명시):
- dtype bf16 (원본 float32): 16GB x2 샤딩에 맞추기 위함
- 투영행렬 = 단위행렬 (원본은 LongMemEval 코퍼스로 만든 13-15층 투영 사용)
- 뱅크/문항 = 우리가 만든 8세션 (원본 64+448 뱅크 아님)
따라서 여기 수치는 원본 810ms/3458ms와 비교 대상이 아니다.
"""
import json
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sessions import SESSIONS, PROMPT

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("EXP_MODEL", str(ROOT / "models" / "Llama-3.1-8B-Instruct"))
LAYERS = [int(x) for x in os.environ.get("EXP_LAYERS", "13,14,15").split(",")]
DTYPE = os.environ.get("EXP_DTYPE", "bf16")
SHARD = os.environ.get("EXP_SHARD", "1") == "1"
NAMES = [f"model.layers.{l}.mlp.down_proj.weight" for l in LAYERS]


def device_map(model_cfg_layers):
    """13-15층이 GPU0에 오도록 앞쪽 절반을 GPU0, 뒤쪽을 GPU1에 배치.
    embed/lm_head/norm 은 같은 카드에 둔다(가중치 공유 모델에서 충돌 방지)."""
    m = {"model.embed_tokens": "cuda:0", "lm_head": "cuda:0", "model.norm": "cuda:0"}
    split = min(21, max(1, model_cfg_layers // 2 + 5))
    for i in range(split):
        m[f"model.layers.{i}"] = "cuda:0"
    for i in range(split, model_cfg_layers):
        m[f"model.layers.{i}"] = "cuda:1"
    return m


def load():
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.pad_token = tok.eos_token
    kw = {}
    if SHARD:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
        kw["device_map"] = device_map(cfg.num_hidden_layers)
    else:
        kw["device_map"] = {"": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if DTYPE == "bf16" else torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
        **kw,
    ).eval()
    return tok, model


def make_hparams(n_layers):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AlphaEdit_hparams", str(CODE / "AlphaEdit" / "AlphaEdit_hparams.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hp = mod.AlphaEditHyperParams.from_json(str(CODE / "config" / "alphaedit-hparams.json"))
    hp.layers = LAYERS
    hp.L2 = 0.1
    hp.v_num_grad_steps = 50
    hp.v_loss_layer = n_layers - 1
    return hp


def identity_projector(device, dim):
    import torch as t
    return [t.eye(dim, dtype=t.float32, device=device) for _ in LAYERS]


def build_requests():
    """세션별 grounded write 요청."""
    from session_write import write_request
    reqs = []
    for s in SESSIONS:
        r = write_request(s, case_id=s["sid"])
        reqs.append(r)
    return reqs


def energy_scores(payloads, keys):
    """세션별 에너지 점수 (원본 factor_scores와 동일 식)."""
    from compact_session_memory import factor_scores
    return {sid: factor_scores(p, keys)[0] for sid, p in payloads.items()}


def cosine_rank(qkey, sess_keys):
    """질문키 vs 세션 대표키 코사인 상위 정렬."""
    q = torch.nn.functional.normalize(qkey.float(), dim=-1)
    out = []
    for sid, k in sess_keys.items():
        kk = torch.nn.functional.normalize(k.float(), dim=-1)
        out.append((sid, float((q * kk).sum())))
    return sorted(out, key=lambda x: (-x[1], x[0]))


def main():
    torch.manual_seed(0)
    os.chdir(CODE)  # util/globals.py 가 cwd 기준으로 globals.yml 을 연다
    import shutil
    shutil.rmtree(ROOT / "factors", ignore_errors=True)
    tok, model = load()
    print("model loaded", flush=True)
    hp = make_hparams(model.config.num_hidden_layers)
    from compact_session_memory import CompactSessionMemory, capture_keys
    import patches
    patches.install()

    store = CompactSessionMemory(model, NAMES, ROOT / "factors")
    projector = identity_projector(next(model.parameters()).device, model.config.intermediate_size)
    reqs = build_requests()

    # ---- 쓰기 ----
    write_times = {}
    for s, req in zip(SESSIONS, reqs):
        t0 = time.monotonic()
        store.write(s["sid"], tok, req, hp, projector)
        write_times[s["sid"]] = time.monotonic() - t0
        print(f"write {s['sid']} {write_times[s['sid']]:.1f}s", flush=True)

    # ---- 대표키 캡처 (원문 질문키) ----
    store.restore()
    qkeys = {s["sid"]: capture_keys(model, tok, NAMES, [s["question"]]) for s in SESSIONS}
    sess_keys = {}
    for s in SESSIONS:
        k = qkeys[s["sid"]]
        sess_keys[s["sid"]] = torch.stack([k[n][0] for n in NAMES]).mean(0)

    # ---- 라우팅 평가 ----
    payloads = {s["sid"]: store.load(s["sid"]) for s in SESSIONS}
    results = []
    for s in SESSIONS:
        qk = capture_keys(model, tok, NAMES, [s["question"]])

        t0 = time.monotonic()
        scores = energy_scores(payloads, qk)
        route_1 = time.monotonic() - t0
        pick_1 = max(scores, key=lambda k: (scores[k], k))

        t0 = time.monotonic()
        ranked = cosine_rank(torch.stack([qk[n][0] for n in NAMES]).mean(0), sess_keys)
        for n in (4, 6):
            top = [sid for sid, _ in ranked[:n]]
            sub = {sid: payloads[sid] for sid in top}
            s2 = energy_scores(sub, qk)
            pick_2 = max(s2, key=lambda k: (s2[k], k))
            if n == 6:
                route_2 = time.monotonic() - t0
        results.append(dict(
            qid=s["sid"], question=s["question"], truth=s["sid"],
            pick_1stage=pick_1, ok_1stage=(pick_1 == s["sid"]),
            pick_2stage_top6=pick_2, ok_2stage=(pick_2 == s["sid"]),
            truth_rank_by_key=[sid for sid, _ in ranked].index(s["sid"]) + 1,
            route_1stage_s=route_1, route_2stage_s=route_2,
        ))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    # ---- 선택 세션으로 답 생성 ----
    from answer_only_generation import generate_answer_batch
    gen = []
    for r, s in zip(results, SESSIONS):
        t0 = time.monotonic()
        out = store.run(r["pick_1stage"], lambda m: generate_answer_batch(m, tok, [s["question"]]))[0]
        gen.append(dict(qid=s["sid"], answer=s["answer"], got=out["text"],
                        exact=(out["text"].strip() == s["answer"].strip()),
                        seconds=time.monotonic() - t0))
        print(json.dumps(gen[-1], ensure_ascii=False), flush=True)

    summary = dict(
        n_sessions=len(SESSIONS), dtype="bfloat16", layers=LAYERS,
        projector="identity", model="Llama-3.1-8B-Instruct",
        write_seconds=write_times,
        route_1stage_correct=sum(r["ok_1stage"] for r in results),
        route_2stage_correct=sum(r["ok_2stage"] for r in results),
        key_top1_correct=sum(1 for r in results if r["truth_rank_by_key"] == 1),
        exact=sum(g["exact"] for g in gen),
        mean_route_1stage_s=sum(r["route_1stage_s"] for r in results) / len(results),
        mean_route_2stage_s=sum(r["route_2stage_s"] for r in results) / len(results),
    )
    (ROOT / "results.json").write_text(
        json.dumps(dict(summary=summary, routing=results, generation=gen), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
