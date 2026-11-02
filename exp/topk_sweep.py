"""2단 라우팅(top-k) 스윕: 키 코사인으로 k개 추린 뒤 그 k개만 에너지 정밀 점수.

검증하려는 것
1) 키 필터의 recall@k — 정답 세션이 상위 k 안에 남는가 (키 저장이 유효한가)
2) k를 줄였을 때 정확도 손실과 라우팅 시간 감소의 교환
3) 1단(전수 에너지) 대비 이득이 어디서 발생하는가

원본 대비 추가: 세션별 대표키를 저장(keys/<sid>.pt)하고 질문키와 코사인 비교.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import common
import patches

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "local-64"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "topk"))
    ap.add_argument("--ks", default="1,2,4,8,16,24,32,48,64")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    patches.install()
    run = Path(a.run)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in a.ks.split(",") if int(x) > 0]

    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    panel = items[:8]
    n_bank = len(items)

    tok, model = common.load_model()

    from compact_session_memory import CompactSessionMemory, capture_keys, factor_scores

    store = CompactSessionMemory(model, common.NAMES, run / "factors")
    payloads = {r["sid"]: store.load(r["sid"]) for r in items}

    # ---- 세션 대표키 저장 ----
    keys_dir = run / "keys"
    keys_dir.mkdir(exist_ok=True)
    sess = {}
    t0 = time.monotonic()
    for r in items:
        p = keys_dir / (r["sid"] + ".pt")
        if p.exists():
            k = torch.load(p, map_location="cpu", weights_only=True)
        else:
            kk = capture_keys(model, tok, common.NAMES, [r["question"]])
            k = torch.cat([kk[n][0].float().cpu() for n in common.NAMES])
            torch.save(k, p)
        sess[r["sid"]] = torch.nn.functional.normalize(k, dim=-1)
    key_build_s = time.monotonic() - t0
    key_bytes = sum((keys_dir / (r["sid"] + ".pt")).stat().st_size for r in items)
    factor_bytes = sum((run / "factors" / (r["sid"] + ".pt")).stat().st_size for r in items)
    print(f"session keys built in {key_build_s:.1f}s, {key_bytes/2**20:.2f}MB "
          f"(factors {factor_bytes/2**20:.2f}MB)", flush=True)

    from answer_only_generation import generate_answer_batch
    from memory_comparison_core import token_f1

    sid_list = [r["sid"] for r in items]
    sess_mat = torch.stack([sess[s] for s in sid_list])

    gen_cache = {}
    trials = []
    for q in panel:
        for rep in range(a.repeats):
            t0 = time.monotonic()
            qk_raw = capture_keys(model, tok, common.NAMES, [q["question"]])
            qk = torch.cat([qk_raw[n][0].float().cpu() for n in common.NAMES])
            qk = torch.nn.functional.normalize(qk, dim=-1)
            cos = (sess_mat @ qk).tolist()
            order = sorted(range(len(sid_list)), key=lambda i: (-cos[i], sid_list[i]))
            key_s = time.monotonic() - t0
            rank_true = order.index(sid_list.index(q["sid"])) + 1

            for k in ks:
                kk = min(k, n_bank)
                cand = [sid_list[i] for i in order[:kk]]
                t1 = time.monotonic()
                sub = {s: payloads[s] for s in cand}
                scores = {s: factor_scores(sub[s], qk_raw)[0] for s in cand}
                pick = max(scores, key=lambda s: (scores[s], s))
                route_s = key_s + (time.monotonic() - t1)

                ck = (q["qid"], pick)
                if ck not in gen_cache:
                    store.apply(payloads[pick])
                    try:
                        o = generate_answer_batch(model, tok, [q["question"]])[0]
                    finally:
                        store.restore()
                    gen_cache[ck] = dict(
                        exact=(o["text"].strip() == q["answer"].strip()),
                        f1=token_f1(o["text"], q["answer"]), text=o["text"])
                g = gen_cache[ck]

                trials.append(dict(qid=q["qid"], repeat=rep, k=kk, pick=pick,
                                   retrieval_at1=(pick == q["sid"]),
                                   key_recall=(rank_true <= kk), key_rank=rank_true,
                                   exact=g["exact"], f1=g["f1"],
                                   route_s=route_s, key_s=key_s))
            print(f"{q['qid']} rep{rep} rank={rank_true} done", flush=True)

    summary = {}
    first = [t for t in trials if t["repeat"] == 0]
    for k in ks:
        rows = [t for t in first if t["k"] == min(k, n_bank)]
        if not rows:
            continue
        n = len(rows)
        summary[str(k)] = dict(
            n=n,
            key_recall=sum(r["key_recall"] for r in rows) / n,
            route_at1=sum(r["retrieval_at1"] for r in rows) / n,
            exact=sum(r["exact"] for r in rows) / n,
            mean_f1=sum(r["f1"] for r in rows) / n,
            mean_route_s=sum(r["route_s"] for r in trials if r["k"] == min(k, n_bank))
            / len([t for t in trials if t["k"] == min(k, n_bank)]),
        )

    result = dict(
        bank=run.name, n_sessions=n_bank, panel=len(panel), repeats=a.repeats,
        layers=common.LAYERS, session_key="concat(13,14,15) then L2-normalize",
        key_bytes=key_bytes, factor_bytes=factor_bytes,
        key_build_s=key_build_s,
        summary=summary, trials=trials,
    )
    (out / "results_topk.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    for k, v in summary.items():
        print(f"k={k:>3} recall@k {v['key_recall']:.2f} route {v['route_at1']:.2f} "
              f"exact {v['exact']:.2f} f1 {v['mean_f1']:.3f} route_s {v['mean_route_s']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
