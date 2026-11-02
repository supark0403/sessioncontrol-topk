"""원본 scripts/prepare_experiment.py 를 로컬 데이터로 돌리기 위한 사본.

바꾼 것: 데이터셋 파일 sha256 게이트만 완화(공개 배포본은 직렬화가 달라 해시가 다름).
그대로 둔 것: 매니페스트 검증, 64세션 재구성, distractor 코퍼스 재구성과
CORPUS_SHA256 검증(내용이 원본과 같으면 이 해시는 반드시 일치한다).
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

DATASET_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
CORPUS_SHA256 = "e23dff571defe17a41c5a1fbe936efefea65029db2bcdd8a78cd063e70134525"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_records(path):
    path = Path(path)
    if path.suffix == ".json" or path.read_bytes()[:1] in (b"[", b"{"):
        return json.loads(path.read_text(encoding="utf-8"))
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return table.to_pylist()


def reconstruct_items(records, manifest):
    entries = manifest if isinstance(manifest, list) else manifest["sessions"]
    by_question = {str(row["question_id"]): row for row in records}
    if len(by_question) != len(records):
        raise ValueError("Dataset has duplicate question identifiers")
    items, questions, sessions = [], set(), set()
    for entry in entries:
        qid = str(entry["qid"])
        if qid in questions:
            raise ValueError("Manifest repeats a question")
        row = by_question[qid]
        answer_sessions = row.get("answer_session_ids") or []
        if row.get("question_type") != "single-session-user" or len(answer_sessions) != 1 or qid.endswith("_abs"):
            raise ValueError("Manifest must use non-abstention single-session-user items")
        source = answer_sessions[0]
        if source != entry["source_session_id"] or source in sessions:
            raise ValueError("Source session mismatch or repeated source session")
        expected_sid = "qa-" + hashlib.sha256(qid.encode()).hexdigest()[:16]
        if entry["sid"] != expected_sid:
            raise ValueError("Delta identifier does not match question")
        conversation = dict(zip(row["haystack_session_ids"], row["haystack_sessions"]))[source]
        if not conversation or not isinstance(row["answer"], str) or not row["answer"].strip():
            raise ValueError("Missing complete answer session or answer")
        items.append(dict(qid=qid, sid=expected_sid, source_session_id=source,
                          answer_session_ids=answer_sessions, history=entry["history"],
                          question=row["question"], answer=row["answer"], context=conversation))
        questions.add(qid)
        sessions.add(source)
    return items


def distractor_records(records, count=200, seed=0):
    excluded = {sid for row in records for sid in row.get("answer_session_ids", [])}
    pool, seen = [], set()
    for row in records:
        for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            if sid in excluded:
                continue
            for turn_index, turn in enumerate(session):
                text = turn.get("content", "")
                if turn.get("role") == "user" and 10 <= len(text.split()) <= 40 and text not in seen:
                    seen.add(text)
                    pool.append(dict(text=text, source_session_id=sid, turn=turn_index))
    if len(pool) < count:
        raise ValueError("Insufficient eligible distractor turns")
    random.Random(seed).shuffle(pool)
    return pool[:count], len(pool), excluded


def prepare(data, manifest_path, out):
    data, manifest_path, out = Path(data).resolve(), Path(manifest_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new preparation directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_records(data)
    print(f"records: {len(records)}")
    items = reconstruct_items(records, manifest)
    if len(items) != 64 or sum(r["history"] in {"previous_test", "writer_development"} for r in items) != 40:
        raise ValueError("Expected the published 64-session cohort and 40 bootstrap writers")
    if any(r["history"] != "previous_test" for r in items[:24]):
        raise ValueError("First 24 manifest entries must be the original test anchors")
    corpus, pool_size, excluded = distractor_records(records)
    corpus_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus).encode("utf-8")
    corpus_ok = hashlib.sha256(corpus_bytes).hexdigest() == CORPUS_SHA256
    print("corpus sha match:", corpus_ok)
    if not corpus_ok:
        raise ValueError("Rebuilt distractor corpus does not reproduce the original bytes")
    out.mkdir(parents=True)
    write_json(out / "items.json", items)
    (out / "distractors.jsonl").write_bytes(corpus_bytes)
    write_json(out / "preparation.json", dict(
        stage="prepared_only", experiment_completed=False,
        dataset=str(data), dataset_sha256=sha256(data),
        dataset_sha256_pinned=DATASET_SHA256,
        dataset_note="공개 배포본은 직렬화가 달라 파일 해시가 다름. corpus 해시로 내용 동일성을 검증.",
        manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
        items_sha256=sha256(out / "items.json"), corpus_sha256=CORPUS_SHA256,
        distractor_pool_size=pool_size, distractor_turns=200,
        excluded_answer_sessions=len(excluded), source="Literal local official dataset",
    ))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print("Prepared", prepare(a.data, a.manifest, a.out))


if __name__ == "__main__":
    main()
