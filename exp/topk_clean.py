"""top-k 재측정: 생성 캐시 없이 k마다 실제 답 생성까지 포함한 시간.

기존 experiment_local.py의 top-k 평가는 (qid, pick)별 생성 시간을 한 번만 재서
모든 k와 warm 반복에 같은 상수로 더했다. 이 스크립트는 그 캐시를 쓰지 않고
매 (질문, k, 반복)마다 실제로 생성한다. 라우팅 타이머도 코사인·정렬까지 포함한다.

N=512는 k 전체 스윕, N=64/128/256은 k=8만 잰다(시간 스케일 확인용).
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

import common
import patches

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "local-512"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--sizes", default="64,128,256,512")
    a = ap.parse_args()

    patches.install()
    import os
    os.chdir(common.CODE)

    from compact_session_memory import CompactSessionMemory, capture_keys, factor_scores
    from answer_only_generation import generate_answer_batch
    from memory_comparison_core import token_f1
    from question_key import normalize

    run = Path(a.run)
    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    ids = json.loads((ROOT / "data" / "prepared" / "panel8.json").read_text(encoding="utf-8"))
    by_qid = {r["qid"]: r for r in items}
    panel = [by_qid[q] for q in ids]
    pid = {r["qid"] for r in panel}
    items = panel + [r for r in items if r["qid"] not in pid]

    tok, model = common.load_model()
    store = CompactSessionMemory(model, common.NAMES, run / "factors")

    keys = {}
    for r in items:
        p = run / "keys" / (r["sid"] + ".pt")
        keys[r["sid"]] = torch.nn.functional.normalize(
            torch.load(p, map_location="cpu", weights_only=True), dim=-1)

    sizes = [int(x) for x in a.sizes.split(",")]
    ks_all = [int(x) for x in a.ks.split(",")]
    trials = []

    for n in sizes:
        subset = items[:n]
        sids = [r["sid"] for r in subset]
        mat = torch.stack([keys[s] for s in sids])
        ks = ks_all if n == 512 else [8]
        for rep in range(a.reps):
            order = list(panel)
            random.Random(100 + rep).shuffle(order)
            for q in order:
                t0 = time.monotonic()
                qk_raw = capture_keys(model, tok, common.NAMES, [q["question"]])
                qv = torch.cat([qk_raw[x][0].float().cpu() for x in common.NAMES])
                qv = torch.nn.functional.normalize(qv, dim=-1)
                cos = (mat @ qv).tolist()
                ranking = sorted(range(len(sids)), key=lambda i: (-cos[i], sids[i]))
                key_s = time.monotonic() - t0
                rank_true = ranking.index(sids.index(q["sid"])) + 1

                for k in ks:
                    t1 = time.monotonic()
                    cand = [sids[i] for i in ranking[:k]]
                    sc = {s: factor_scores(store.load(s), qk_raw)[0] for s in cand}
                    pick = max(sc, key=lambda s: (sc[s], s))
                    route_s = key_s + (time.monotonic() - t1)

                    payload = store.load(pick)
                    store.apply(payload)
                    try:
                        g = generate_answer_batch(model, tok, [q["question"]])[0]
                    finally:
                        store.restore()
                    total_s = key_s + (time.monotonic() - t1)
                    trials.append(dict(
                        n=n, k=k, repeat=rep, qid=q["qid"], pick=pick,
                        key_rank=rank_true, key_hit=(rank_true <= k),
                        retrieval_at1=(pick == q["sid"]),
                        exact=(normalize(g["text"]) == normalize(q["answer"])),
                        f1=token_f1(g["text"], q["answer"]),
                        route_ms=round(route_s * 1000, 1),
                        total_ms=round(total_s * 1000, 1),
                    ))
                print(n, rep, q["qid"], "rank", rank_true, flush=True)

    summary = {}
    for n in sizes:
        ks = ks_all if n == 512 else [8]
        summary[str(n)] = {}
        for k in ks:
            warm = [t for t in trials if t["n"] == n and t["k"] == k and t["repeat"] > 0]
            first = [t for t in trials if t["n"] == n and t["k"] == k and t["repeat"] == 0]
            summary[str(n)][str(k)] = dict(
                n=len(warm),
                total_ms=round(statistics.median([t["total_ms"] for t in warm]), 1),
                route_ms=round(statistics.median([t["route_ms"] for t in warm]), 1),
                recall=sum(t["key_hit"] for t in first) / len(first),
                exact=sum(t["exact"] for t in first),
                f1=sum(t["f1"] for t in first) / len(first),
            )

    out = dict(config=dict(reps=a.reps, sizes=sizes, ks=ks_all,
                           dtype=common.DTYPE, shard=common.SHARD,
                           generation="real per (q,k,rep), no gen_cache"),
               summary=summary, trials=trials)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
