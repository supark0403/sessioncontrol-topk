"""KV 전체 prefix 재측정: 레이어를 16/16으로 균형 배치해 32GiB를 실제로 쓰는 실험.

기존 실험은 13–15층을 GPU0에 고정(21/11 분할)해서 GPU0에만 KV가 몰렸다.
KV 경로는 write가 필요 없으므로 13–15층 고정 조건을 풀고 균형 배치로 측정한다.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import common
import patches

ROOT = Path(__file__).resolve().parent


def balanced_map(n_layers):
    m = {"model.embed_tokens": "cuda:0", "lm_head": "cuda:0",
         "model.norm": "cuda:0", "model.rotary_emb": "cuda:0"}
    half = n_layers // 2
    for i in range(half):
        m[f"model.layers.{i}"] = "cuda:0"
    for i in range(half, n_layers):
        m[f"model.layers.{i}"] = "cuda:1"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "local-512"))
    ap.add_argument("--sizes", default="8,16,32")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "runs" / "kv_balanced.json"))
    a = ap.parse_args()

    patches.install()
    import os
    os.chdir(common.CODE)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from memory_comparison_core import bank_prefix, query_suffix, prefill, decode_cached, kv_bytes_per_token
    from experiment_utils import document, evaluated, metrics

    run = Path(a.run)
    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    panel_file = ROOT / "data" / "prepared" / "panel8.json"
    ids = json.loads(panel_file.read_text(encoding="utf-8"))
    by_qid = {r["qid"]: r for r in items}
    panel = [by_qid[q] for q in ids]
    # 본 실험과 동일하게 패널을 뱅크 앞으로 (N=8/16에도 정답 세션이 포함되도록)
    pid = {r["qid"] for r in panel}
    items = panel + [r for r in items if r["qid"] not in pid]

    tok = AutoTokenizer.from_pretrained(common.MODEL, local_files_only=True)
    tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(common.MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=balanced_map(cfg.num_hidden_layers), local_files_only=True).eval()
    print("balanced model loaded", flush=True)

    bank = [dict(record_id="record-" + str(i), sid=r["sid"], stored_question=r["question"],
                 stored_answer=r["answer"],
                 conversation="\n".join(t["role"] + ": " + t["content"] for t in r["context"]))
            for i, r in enumerate(items)]

    out = dict(config=dict(dtype="bf16", shard="balanced 16/16"), results={})
    for n in [int(x) for x in a.sizes.split(",")]:
        folder = run / ("bank-" + str(n))
        folder.mkdir(parents=True, exist_ok=True)
        pids = tok.encode(bank_prefix(tok, [document(r) for r in bank[:n]]), add_special_tokens=False)
        suffixes = {q["qid"]: tok.encode(query_suffix(tok, q["question"]), add_special_tokens=False)
                    for q in panel}
        required = len(pids) + max(map(len, suffixes.values())) + 64
        plan = dict(prefix_tokens=len(pids), required_tokens=required,
                    limit=cfg.max_position_embeddings,
                    bytes_per_token=kv_bytes_per_token(cfg))
        if required > cfg.max_position_embeddings:
            out["results"][str(n)] = dict(available=False, reason="context_limit", plan=plan)
            print(n, "context_limit", required, flush=True)
            continue
        t = time.monotonic()
        try:
            cache = prefill(model, pids, chunk_size=a.chunk)
        except torch.OutOfMemoryError as exc:
            out["results"][str(n)] = dict(available=False, reason="gpu_limit", error=str(exc), plan=plan)
            torch.cuda.empty_cache()
            print(n, "OOM", flush=True)
            continue
        setup_s = time.monotonic() - t
        trials = []
        for rep in range(2):
            for q in panel:
                t0 = time.monotonic()
                final, cache = decode_cached(model, tok, cache, suffixes[q["qid"]])
                secs = time.monotonic() - t0
                tr = evaluated(q, final, secs)
                tr.update(repeat=rep, context_tokens=len(pids))
                trials.append(tr)
        summary = metrics([x for x in trials if x["repeat"] == 0])
        out["results"][str(n)] = dict(available=True, plan=plan, setup_s=round(setup_s, 1),
                                      summary=summary, trials=trials)
        print(n, "exact", summary["returned_exact"]["count"], "/8 F1",
              round(summary["mean_token_f1"], 4), "setup", round(setup_s, 1), flush=True)
        cache = None
        torch.cuda.empty_cache()

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
