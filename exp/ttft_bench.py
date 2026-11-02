"""3방식 TTFT 측정: 델타 / RAG / top-k.

환경 왜곡을 줄이기 위해
- 세션 파일은 1회 프리로드 (디스크 I/O 제거)
- 모델은 단일 GPU (PCIe 왕복 제거)
를 기본으로 한다. EXP_SHARD=1 이면 샤딩 조건으로도 측정된다.

TTFT 정의: 질문 도착 → 첫 non-empty 답 텍스트.
델타/top-k는 라우팅 시간을 포함하고, RAG는 임베딩+검색+prefill을 포함한다.
"""
import json
import statistics
import time
from pathlib import Path

import torch

import common
import patches

ROOT = Path(__file__).resolve().parent


def median(xs):
    return round(statistics.median(xs), 1)


def main():
    patches.install()
    import os
    os.chdir(common.CODE)

    from transformers import AutoModel, AutoTokenizer
    from compact_session_memory import (CompactSessionMemory, capture_keys, factor_scores)
    from answer_only_generation import generate_answer_batch
    from memory_comparison_core import (bank_prefix, query_suffix, prefill,
                                        embed, rank_vectors)
    from ttft_timing import FirstAnswerClock, observe_generate, decode_cached_observed
    from experiment_utils import document
    from question_key import normalize

    run = ROOT / "runs" / "local-512"
    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    ids = json.loads((ROOT / "data" / "prepared" / "panel8.json").read_text(encoding="utf-8"))
    by_qid = {r["qid"]: r for r in items}
    panel = [by_qid[q] for q in ids]
    # 본 실험과 동일하게 패널을 뱅크 앞으로 (record-id 번호까지 일치시킨다)
    pid = {r["qid"] for r in panel}
    items = panel + [r for r in items if r["qid"] not in pid]

    tok, model = common.load_model()
    store = CompactSessionMemory(model, common.NAMES, run / "factors")

    t0 = time.monotonic()
    payload = {r["sid"]: store.load(r["sid"]) for r in items[:512]}
    preload_s = time.monotonic() - t0

    sess_key = {}
    for r in items[:512]:
        k = torch.load(run / "keys" / (r["sid"] + ".pt"), map_location="cpu", weights_only=True)
        sess_key[r["sid"]] = torch.nn.functional.normalize(k, dim=-1)
    sids = [r["sid"] for r in items[:512]]
    mat = torch.stack([sess_key[s] for s in sids])

    emb = str(ROOT / "models" / "all-MiniLM-L6-v2")
    etok = AutoTokenizer.from_pretrained(emb, local_files_only=True)
    enc = AutoModel.from_pretrained(emb, local_files_only=True).to("cuda:0").float().eval()
    bank = [dict(record_id="record-" + str(i), sid=r["sid"], stored_question=r["question"],
                 stored_answer=r["answer"],
                 conversation="\n".join(t["role"] + ": " + t["content"] for t in r["context"]))
            for i, r in enumerate(items[:512])]
    vectors = embed(enc, etok, [b["stored_question"] for b in bank])

    res = {"delta": [], "rag": [], "topk": []}
    k = 8
    reps = int(os.environ.get("EXP_TTFT_REPS", "5"))
    for rep in range(reps):
        for q in panel:
            # ---- delta ----
            t0 = time.monotonic()
            clock = FirstAnswerClock(tok, t0)
            keys = capture_keys(model, tok, common.NAMES, [q["question"]])
            sc = {s: factor_scores(payload[s], keys)[0] for s in sids}
            pick = max(sc, key=lambda s: (sc[s], s))
            store.apply(payload[pick])
            try:
                with observe_generate(model, clock):
                    gen = generate_answer_batch(model, tok, [q["question"]])[0]
            finally:
                store.restore()
            total = (time.monotonic() - t0) * 1000
            ttft = clock.result()["ttft_seconds"]
            res["delta"].append(dict(rep=rep, qid=q["qid"], pick=pick,
                                     ttft_ms=None if ttft is None else round(ttft * 1000, 1),
                                     total_ms=round(total, 1),
                                     exact=(normalize(gen["text"]) == normalize(q["answer"]))))

            # ---- top-k ----
            t0 = time.monotonic()
            clock = FirstAnswerClock(tok, t0)
            keys = capture_keys(model, tok, common.NAMES, [q["question"]])
            qv = torch.cat([keys[n][0].float().cpu() for n in common.NAMES])
            qv = torch.nn.functional.normalize(qv, dim=-1)
            cos = (mat @ qv).tolist()
            order = sorted(range(len(sids)), key=lambda i: (-cos[i], sids[i]))[:k]
            sc = {sids[i]: factor_scores(payload[sids[i]], keys)[0] for i in order}
            pick = max(sc, key=lambda s: (sc[s], s))
            store.apply(payload[pick])
            try:
                with observe_generate(model, clock):
                    gen = generate_answer_batch(model, tok, [q["question"]])[0]
            finally:
                store.restore()
            total = (time.monotonic() - t0) * 1000
            ttft = clock.result()["ttft_seconds"]
            res["topk"].append(dict(rep=rep, qid=q["qid"], pick=pick, k=k,
                                    ttft_ms=None if ttft is None else round(ttft * 1000, 1),
                                    total_ms=round(total, 1),
                                    exact=(normalize(gen["text"]) == normalize(q["answer"]))))

            # ---- RAG ----
            t0 = time.monotonic()
            clock = FirstAnswerClock(tok, t0)
            qv = embed(enc, etok, [q["question"]])[0]
            ranked = rank_vectors(qv, vectors, 1)
            doc = document(bank[ranked[0][0]])
            pids = tok.encode(bank_prefix(tok, [doc]), add_special_tokens=False)
            cache = prefill(model, pids, chunk_size=256)
            suffix = tok.encode(query_suffix(tok, q["question"]), add_special_tokens=False)
            final, _ = decode_cached_observed(model, tok, cache, suffix, clock)
            total = (time.monotonic() - t0) * 1000
            ttft = clock.result()["ttft_seconds"]
            res["rag"].append(dict(rep=rep, qid=q["qid"],
                                   pick=bank[ranked[0][0]]["sid"],
                                   ttft_ms=None if ttft is None else round(ttft * 1000, 1),
                                   total_ms=round(total, 1),
                                   exact=(normalize(final["text"]) == normalize(q["answer"]))))
            cache = None
            torch.cuda.empty_cache()
            print(rep, q["qid"], "done", flush=True)

    summary = {}
    for m, rows in res.items():
        warm = [r for r in rows if r["rep"] > 0]
        summary[m] = dict(
            n=len(warm),
            ttft_ms=median([r["ttft_ms"] for r in warm]),
            total_ms=median([r["total_ms"] for r in warm]),
            exact=sum(r["exact"] for r in rows if r["rep"] == 0),
        )

    out = dict(config=dict(shard=common.SHARD, dtype=common.DTYPE,
                           preload_seconds=round(preload_s, 1),
                           panel=len(panel), k=k),
               summary=summary, trials=res)
    (ROOT / "runs" / "ttft3.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
