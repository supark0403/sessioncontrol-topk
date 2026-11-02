"""원본 experiment_real512_compact.py의 448 extractive 세션 생성 규칙을 그대로 옮긴 것.

원본과 동일: reserved 제외, 32000자 초과 스킵, user 문장 9~18단어, I/My/We 시작,
물음표 제외, 앞 6단어 접두가 세션 내 유일, 질문 문구, sid/qid 해시, sha256(sid) 정렬, 448개.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(records, existing_questions):
    sessions = {}
    for rec in records:
        for sid, ctx in zip(rec["haystack_session_ids"], rec["haystack_sessions"]):
            if sid in sessions and sessions[sid] != ctx:
                raise ValueError("inconsistent session")
            sessions[sid] = ctx

    reserved = {sid for rec in records for sid in rec["answer_session_ids"]}
    seen = set(existing_questions)
    added = []

    for sid in sorted(sessions, key=lambda s: hashlib.sha256(s.encode()).hexdigest()):
        if sid in reserved:
            continue
        ctx = sessions[sid]
        if sum(len(t["content"]) for t in ctx) > 32000:
            continue
        sentences = [
            s.strip()
            for t in ctx if t["role"] == "user"
            for s in re.split(r"(?<=[.!?])\s+|\n+", t["content"])
        ]
        for sentence in sentences:
            words = sentence.split()
            if not 9 <= len(words) <= 18 or not re.match(r"^(I |My |We )", sentence) or "?" in sentence:
                continue
            prefix = " ".join(words[:6])
            if sum(s.startswith(prefix) for s in sentences) != 1:
                continue
            question = "What was my complete statement beginning with " + json.dumps(prefix, ensure_ascii=False) + "?"
            if question in seen:
                continue
            turn = next(i for i, t in enumerate(ctx) if t["role"] == "user" and sentence in t["content"])
            suffix = hashlib.sha256(sid.encode()).hexdigest()[:16]
            added.append(dict(
                qid="extractive-" + suffix, sid="real-" + suffix,
                source_session_id=sid, question=question, answer=sentence,
                context=ctx, answer_session_ids=[sid], history="new_extractive_distractor",
                annotation=dict(kind="unique_user_quote_completion", user_turn=turn,
                                quote_prefix=prefix, answer_is_verbatim_user_span=True,
                                context_sha256=hashlib.sha256(
                                    json.dumps(ctx, ensure_ascii=False, sort_keys=True).encode()
                                ).hexdigest()),
            ))
            seen.add(question)
            break
        if len(added) == 448:
            break

    return added, len(sessions), len(reserved)


def main():
    prepared = ROOT / "data" / "prepared"
    items64 = json.loads((prepared / "items.json").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / "models" / "longmemeval" / "longmemeval_s").read_text(encoding="utf-8"))
    added, n_sess, n_res = build(raw, {r["question"] for r in items64})
    print(f"sessions={n_sess} reserved={n_res} added={len(added)}")
    if len(added) != 448:
        raise ValueError("expected 448 extractive sessions")
    if len({r["source_session_id"] for r in items64 + added}) != 512:
        raise ValueError("duplicate source sessions")
    items = items64 + added
    out = prepared / "items_512.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, len(items))


if __name__ == "__main__":
    main()
