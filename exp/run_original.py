"""원본 리포 방식 재현: write → 전 세션 에너지 전수 라우팅 → 선택 세션으로 생성.
+ 원본 비교군 Matched RAG.

원본과 동일: 절차, 층(13-15), L2=0.1, 50스텝 grounded write, greedy 64토큰, EOS/턴 정지.
원본과 다름: dtype(bf16), 투영행렬(단위), 뱅크(우리 8세션), 모델샤딩. → 시간은 원본과 비교 불가.
"""
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
import common  # noqa: E402  (cwd 고정 포함)

from sessions import SESSIONS  # noqa: E402


def build_documents(tok):
    """RAG가 읽을 문서 문자열 (원본 document()와 같은 구성)."""
    docs = []
    for s in SESSIONS:
        conv = "\n".join(f"{t['role']}: {t['content']}" for t in s["context"])
        docs.append(
            f"record-{s['sid']}\nConversation:\n{conv}\n"
            f"Stored question: {s['question']}\nStored answer: {s['answer']}"
        )
    return docs


def main():
    torch.manual_seed(0)
    import shutil
    shutil.rmtree(ROOT / "factors", ignore_errors=True)

    tok, model, hp, projector = common.prepare()
    print("model loaded", flush=True)

    from compact_session_memory import CompactSessionMemory, capture_keys, factor_scores
    from session_write import write_request
    from answer_only_generation import generate_answer_batch
    from memory_comparison_core import (
        bank_prefix, query_suffix, prefill, token_f1,
    )
    from ttft_timing import FirstAnswerClock, observe_generate, decode_cached_observed

    store = CompactSessionMemory(model, common.NAMES, ROOT / "factors")

    # ---------- 1) write ----------
    writes = {}
    for s in SESSIONS:
        req = write_request(s, case_id=s["sid"])
        t0 = time.monotonic()
        store.write(s["sid"], tok, req, hp, projector)
        writes[s["sid"]] = round(time.monotonic() - t0, 2)
        print(f"write {s['sid']} {writes[s['sid']]}s", flush=True)
    store.restore()

    # ---------- 2) 원본 방식 라우팅: 전 세션 에너지 전수 → argmax ----------
    payloads = {s["sid"]: store.load(s["sid"]) for s in SESSIONS}
    rows = []
    for s in SESSIONS:
        t0 = time.monotonic()
        keys = capture_keys(model, tok, common.NAMES, [s["question"]])
        scores = {sid: factor_scores(p, keys)[0] for sid, p in payloads.items()}
        pick = max(scores, key=lambda k: (scores[k], k))
        route_s = time.monotonic() - t0

        clock = FirstAnswerClock(tok, t0)
        store.apply(payloads[pick])
        try:
            with observe_generate(model, clock):
                out = generate_answer_batch(model, tok, [s["question"]])[0]
        finally:
            store.restore()
        total_s = time.monotonic() - t0
        ttft = clock.result()["ttft_seconds"]

        rows.append(dict(
            qid=s["sid"], truth=s["sid"], pick=pick,
            retrieval_at1=(pick == s["sid"]),
            exact=(out["text"].strip() == s["answer"].strip()),
            f1=round(token_f1(out["text"], s["answer"]), 4),
            got=out["text"],
            route_s=round(route_s, 3), ttft_s=None if ttft is None else round(ttft, 3),
            total_s=round(total_s, 3),
        ))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    # ---------- 3) Matched RAG ----------
    from transformers import AutoModel, AutoTokenizer
    from memory_comparison_core import embed, rank_vectors

    emb_path = str(ROOT / "models" / "all-MiniLM-L6-v2")
    etok = AutoTokenizer.from_pretrained(emb_path, local_files_only=True)
    enc = AutoModel.from_pretrained(emb_path, local_files_only=True).to("cuda:0").float().eval()
    docs = build_documents(tok)
    vectors = embed(enc, etok, [s["question"] for s in SESSIONS])  # 저장 질문 임베딩

    rag = []
    for s in SESSIONS:
        t0 = time.monotonic()
        qv = embed(enc, etok, [s["question"]])[0]
        ranked, _ = rank_vectors(qv, vectors, 1)
        doc = docs[ranked[0][0]]
        pids = tok.encode(bank_prefix(tok, [doc]), add_special_tokens=False)
        cache = prefill(model, pids, chunk_size=256)
        suffix = tok.encode(query_suffix(tok, s["question"]), add_special_tokens=False)
        clock = FirstAnswerClock(tok, t0)
        final, _ = decode_cached_observed(model, tok, cache, suffix, clock)
        total_s = time.monotonic() - t0
        ttft = clock.result()["ttft_seconds"]
        rag.append(dict(
            qid=s["sid"], pick_sid=SESSIONS[ranked[0][0]]["sid"],
            retrieval_at1=(SESSIONS[ranked[0][0]]["sid"] == s["sid"]),
            exact=(final["text"].strip() == s["answer"].strip()),
            f1=round(token_f1(final["text"], s["answer"]), 4),
            got=final["text"],
            ttft_s=None if ttft is None else round(ttft, 3),
            total_s=round(total_s, 3),
        ))
        print(json.dumps(rag[-1], ensure_ascii=False), flush=True)

    def summ(rs, key="retrieval_at1"):
        n = len(rs)
        return dict(
            n=n,
            retrieval_at1=sum(r[key] for r in rs),
            exact=sum(r["exact"] for r in rs),
            mean_f1=round(sum(r["f1"] for r in rs) / n, 4),
            mean_ttft_s=round(sum((r["ttft_s"] or 0) for r in rs) / n, 3),
            mean_total_s=round(sum(r["total_s"] for r in rs) / n, 3),
        )

    out = dict(
        config=dict(model=common.MODEL, dtype=common.DTYPE, layers=common.LAYERS,
                    shard=common.SHARD, n_sessions=len(SESSIONS),
                    projector="identity", note="원본 절차 재현, dtype/투영/뱅크는 다름"),
        write_seconds=writes,
        original=dict(summary=summ(rows), rows=rows),
        rag=dict(summary=summ(rag), rows=rag),
    )
    (ROOT / "results_original.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(original=out["original"]["summary"], rag=out["rag"]["summary"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
