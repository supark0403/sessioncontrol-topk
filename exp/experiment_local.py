"""원본 experiment_real512_compact.py 를 로컬 8세션 세팅용으로 복사·개조한 파일.

원본에서 바꾼 것 (그 외 절차/함수/지표는 동일):
- prepare(): 64세션 LongMemEval 소스뱅크 요구 → 로컬 8세션 뱅크 생성
- run(): GPU1 강제/해시 게이트/legacy dense 검증/448 추가쓰기 제거
- 모델: bf16 + 2GPU 샤딩 (원본 float32 단일 GPU)
- 투영행렬: 단위행렬 (원본은 코퍼스 기반 null-space)
- SIZES: [8] (원본 8..512)

원본과 같은 것: grounded write(50스텝, L2=0.1, 13-15층), 에너지 전수 라우팅,
선택 델타 적용→greedy 64토큰 생성→복원, Matched RAG 비교군, KV 전체 prefix 비교군,
자원/지연 측정 구조.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import time

import common
import patches

from experiment_utils import read, save, sha256
from fixed_query_scaling_core import latency_summary
from experiment_utils import document, evaluated, metrics
from memory_comparison_core import (bank_prefix, query_suffix, kv_bytes_per_token,
    cache_payload_bytes, prefill, decode_cached, embed, rank_vectors)
from answer_only_generation import generate_answer_batch
from question_key import prompt

SIZES = tuple(int(x) for x in os.environ.get(
    "EXP_SIZES", "8,16,32,64,128,256,512").split(","))
TOPK_SIZES = tuple(int(x) for x in os.environ.get(
    "EXP_TOPK_SIZES", "64,128,256,512").split(","))
TOPK_KS = tuple(int(x) for x in os.environ.get(
    "EXP_TOPK_KS", "1,2,4,8,16,32,64,128").split(","))
ROOT = Path(__file__).resolve().parent


def prepare(out):
    """원본 64세션 + 448 extractive = 512 뱅크."""
    p512 = ROOT / "data" / "prepared" / "items_512.json"
    src = p512 if p512.exists() else (ROOT / "data" / "prepared" / "items.json")
    items = json.loads(src.read_text(encoding="utf-8"))
    design = dict(model=common.MODEL, layers=common.LAYERS, dtype=common.DTYPE,
                  shard=common.SHARD, sizes=list(SIZES), repeats=3, projector="empirical",
                  panel_note="원본 8문항 패널 파일은 비공개 → 매니페스트 순서 첫 8개로 대체",
                  scope=("원본 64세션 + 원본 규칙으로 생성한 448 extractive = 512. "
                         "dtype bf16·2GPU 샤딩이라 원본 시간/정확도와 직접 비교 불가."))
    out.mkdir(parents=True, exist_ok=True)
    save(out, "items.json", items)
    save(out, "design.json", design)
    save(out, "completion.json", dict(completed=False, stage="prepared"))
    print(f"PREPARED {len(items)} sessions", flush=True)


def records(items):
    return [dict(record_id="record-" + str(i), sid=r["sid"], stored_question=r["question"],
                 stored_answer=r["answer"],
                 conversation="\n".join(t["role"] + ": " + t["content"] for t in r["context"]))
            for i, r in enumerate(items)]


def run(out, stage="all"):
    patches.install()
    os.chdir(common.CODE)

    import torch
    from transformers import AutoModel, AutoTokenizer

    from compact_session_memory import CompactSessionMemory
    from session_write import write_request

    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    design = read(out / "design.json")
    design["repeats"] = int(os.environ.get("EXP_REPEATS", design["repeats"]))
    items = read(out / "items.json")
    tok, model = common.load_model()
    hp = common.make_hparams()
    hp.v_loss_layer = model.config.num_hidden_layers - 1
    proj_path = ROOT / "projection" / "alphaedit-13-15.pt"
    if proj_path.exists():
        projector = [p.to(next(model.parameters()).device) for p in torch.load(
            proj_path, map_location="cpu", weights_only=True)]
        print("projector: empirical", flush=True)
    else:
        projector = common.identity_projector(next(model.parameters()).device,
                                              model.config.intermediate_size)
        print("projector: identity (fallback)", flush=True)
    names = common.NAMES
    compact = CompactSessionMemory(model, names, out / "factors")
    base_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    begin = time.monotonic()
    measurements = {}
    results = {}

    def sync():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    def status(stage, **extra):
        save(out, "status.json", dict(stage=stage, elapsed_seconds=time.monotonic() - begin, **extra))

    def start(stage):
        gc.collect()
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
        sync(); status(stage)
        return time.monotonic()

    def end(stage, clock, **extra):
        sync()
        ndev = max(1, torch.cuda.device_count())
        r = dict(elapsed_seconds=time.monotonic() - clock,
                 peak_allocated_bytes=max(torch.cuda.max_memory_allocated(i) for i in range(ndev)),
                 peak_reserved_bytes=max(torch.cuda.max_memory_reserved(i) for i in range(ndev)),
                 end_allocated_bytes=sum(torch.cuda.memory_allocated(i) for i in range(ndev)), **extra)
        measurements[stage] = r
        save(out, "measurements.json", measurements)
        return r

    def timed(function):
        sync(); t = time.monotonic(); value = function(); sync()
        return value, time.monotonic() - t

    save(out, "completion.json", dict(completed=False, stage="running"))

    # ---- 1) write: 원본과 동일한 grounded writer (이미 쓴 세션은 건너뜀) ----
    prev = {r["sid"]: r for r in (read(out / "writes.json") if (out / "writes.json").exists() else [])}
    t = start("writes")
    writes = []
    for index, r in enumerate(items):
        status("writes", done=index, total=len(items), qid=r["qid"])
        path = out / "factors" / (r["sid"] + ".pt")
        if path.exists():
            row = prev.get(r["sid"])
            if row is None:
                row = dict(sid=r["sid"], returned_exact=None, query_seconds=None,
                           sha256=sha256(path), bytes=path.stat().st_size, resumed=True)
            writes.append(row)
            continue
        request = write_request(r, index)
        request["target_new"]["str"] = r["answer"] + tok.eos_token
        w = time.monotonic()
        compact.write(r["sid"], tok, request, hp, projector)
        final = compact.run(r["sid"], lambda m: generate_answer_batch(m, tok, [r["question"]]))[0]
        row = evaluated(r, final, time.monotonic() - w, selected_sid=r["sid"])
        row.update(sid=r["sid"], sha256=sha256(path), bytes=path.stat().st_size)
        writes.append(row)
        save(out, "writes.json", writes)
        print("WRITE", index + 1, "/", len(items), "exact", row["returned_exact"], flush=True)
    end("writes", t, count=len(writes),
        actual_factor_payload_bytes=sum(r["bytes"] for r in writes))
    compact.restore()
    projector = None
    gc.collect(); torch.cuda.empty_cache()

    if stage == "writes":
        save(out, "completion.json", dict(completed=False, stage="writes_done",
                                          written=len(writes)))
        print("WRITES DONE", len(writes), flush=True)
        return

    # ---- 2) 원본 라우팅: 질문키 1개 → 전 세션 에너지 전수 → argmax → 생성 ----
    pfile = ROOT / "data" / "prepared" / "panel8.json"
    if pfile.exists():
        ids = json.loads(pfile.read_text(encoding="utf-8"))
        by_qid = {r["qid"]: r for r in items}
        panel = [by_qid[q] for q in ids]
        print("panel: reconstructed 8 questions", flush=True)
    else:
        panel = items[:8]
    # 원본 nested_bank와 동일하게 패널을 뱅크 앞쪽에 둔다(작은 N에서 패널이 포함되도록).
    panel_ids = {r["qid"] for r in panel}
    items = panel + [r for r in items if r["qid"] not in panel_ids]
    bank = records(items)
    save(out, "bank.json", bank)
    from compact_session_memory import capture_keys, factor_scores
    for n in SIZES:
        (out / ("bank-" + str(n))).mkdir(parents=True, exist_ok=True)
        t = start("our_delta_" + str(n))
        trials = []
        for repeat in range(design["repeats"]):
            order = list(panel); random.Random(100 + repeat).shuffle(order)
            for q in order:
                status("our_delta", n=n, repeat=repeat, qid=q["qid"])

                def route():
                    keys = capture_keys(model, tok, names, [q["question"]])
                    scores = {r["sid"]: factor_scores(compact.load(r["sid"]), keys)[0]
                              for r in items[:n]}
                    return sorted(scores, key=lambda sid: (-scores[sid], sid)), scores

                (ranked, scores), route_seconds = timed(route)
                final, answer_seconds = timed(
                    lambda: compact.run(ranked[0], lambda m: generate_answer_batch(m, tok, [q["question"]]))[0])
                final["max_query_cache_payload_bytes"] = (
                    len(tok.encode(prompt(q["question"]))) + final["tokens"] - 1) * kv_bytes_per_token(model.config)
                final["cache_measurement"] = "Configuration-derived estimate for generate API transient cache"
                trial = evaluated(q, final, route_seconds + answer_seconds, selected_sid=ranked[0])
                trial.update(repeat=repeat, rank=ranked.index(q["sid"]) + 1,
                             route_seconds=route_seconds, answer_seconds=answer_seconds,
                             activation_scores=scores)
                trials.append(trial)
                save(out / ("bank-" + str(n)), "our_delta.json", dict(trials=trials))
        resource = end("our_delta_" + str(n), t,
                       persistent_payload_bytes=sum((out / "factors" / (r["sid"] + ".pt")).stat().st_size
                                                    for r in items[:n]),
                       persistent_tier="Disk compact factors; streamed per query",
                       additional_model_bytes=0)
        first = [r for r in trials if r["repeat"] == 0]
        summary = metrics(first)
        summary.update(mean_rank=sum(r["rank"] for r in first) / len(first),
                       mrr=sum(1 / r["rank"] for r in first) / len(first))
        results[str(n)] = {"our_delta": dict(available=True, trials=trials,
                                             summary=summary, latency=latency_summary(trials),
                                             resources=resource)}
        save(out, "evaluation_progress.json", dict(results=results))
        print("OUR", n, "route", summary["retrieval_at1"]["count"],
              "exact", summary["returned_exact"]["count"], flush=True)
    compact.restore()

    # ---- 2b) 우리 방식: 키 코사인 top-k → 그 k개만 에너지 → 생성 ----
    from compact_session_memory import capture_keys, factor_scores
    keys_dir = out / "keys"
    keys_dir.mkdir(exist_ok=True)
    for n in [s for s in TOPK_SIZES if s <= len(items)]:
        subset = items[:n]
        sess = {}
        for r in subset:
            p = keys_dir / (r["sid"] + ".pt")
            if p.exists():
                k = torch.load(p, map_location="cpu", weights_only=True)
            else:
                kk = capture_keys(model, tok, names, [r["question"]])
                k = torch.cat([kk[x][0].float().cpu() for x in names])
                torch.save(k, p)
            sess[r["sid"]] = torch.nn.functional.normalize(k, dim=-1)
        sid_list = [r["sid"] for r in subset]
        mat = torch.stack([sess[s] for s in sid_list])
        ks = [k for k in TOPK_KS if k <= n]
        trials_by_k = {k: [] for k in ks}
        for repeat in range(design["repeats"]):
            order = list(panel); random.Random(100 + repeat).shuffle(order)
            for q in order:
                t0 = time.monotonic()
                qk_raw = capture_keys(model, tok, names, [q["question"]])
                qk = torch.cat([qk_raw[x][0].float().cpu() for x in names])
                qk = torch.nn.functional.normalize(qk, dim=-1)
                cos = (mat @ qk).tolist()
                ranking = sorted(range(len(sid_list)), key=lambda i: (-cos[i], sid_list[i]))
                key_s = time.monotonic() - t0
                rank_true = ranking.index(sid_list.index(q["sid"])) + 1
                for k in ks:
                    cand = [sid_list[i] for i in ranking[:k]]
                    t1 = time.monotonic()
                    sc = {s: factor_scores(compact.load(s), qk_raw)[0] for s in cand}
                    pick = max(sc, key=lambda s: (sc[s], s))
                    energy_s = time.monotonic() - t1
                    final, gen_s = timed(lambda: compact.run(
                        pick, lambda m: generate_answer_batch(m, tok, [q["question"]]))[0])
                    tr = evaluated(q, final, key_s + energy_s + gen_s, selected_sid=pick)
                    tr.update(repeat=repeat, k=k, key_rank=rank_true,
                              key_hit=(rank_true <= k), route_seconds=key_s + energy_s)
                    trials_by_k[k].append(tr)
                status("topk", n=n, repeat=repeat, qid=q["qid"])
        for k, trs in trials_by_k.items():
            first = [x for x in trs if x["repeat"] == 0]
            results[str(n)].setdefault("topk", {})[str(k)] = dict(
                trials=trs, summary=metrics(first), latency=latency_summary(trs),
                key_recall=sum(x["key_hit"] for x in first) / len(first))
        save(out, "evaluation_progress.json", dict(results=results))
        print("TOPK", n,
              {k: (v["summary"]["retrieval_at1"]["count"],
                   round(v["latency"]["warm_median_seconds"] * 1000))
               for k, v in results[str(n)]["topk"].items()}, flush=True)

    # ---- 3) Matched RAG ----
    t = start("rag_index_setup")
    emb_path = str(Path(design["model"]).parent / "all-MiniLM-L6-v2")
    etok = AutoTokenizer.from_pretrained(emb_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(emb_path, local_files_only=True).to("cuda:0").float().eval()
    embbytes = sum(p.numel() * p.element_size() for p in encoder.parameters())
    vectors = embed(encoder, etok, [r["stored_question"] for r in bank])
    torch.save(vectors, out / "qa_embeddings.pt")
    end("rag_index_setup", t, embedding_parameter_bytes=embbytes,
        vector_payload_bytes=vectors.numel() * vectors.element_size())

    for n in SIZES:
        folder = out / ("bank-" + str(n))
        save(folder, "bank.json", bank[:n])
        torch.save(vectors[:n].clone(), folder / "qa_embeddings.pt")
        t = start("rag_qa_" + str(n))
        trials = []
        for repeat in range(design["repeats"]):
            order = list(panel); random.Random(100 + repeat).shuffle(order)
            for q in order:
                status("rag_qa", n=n, repeat=repeat, qid=q["qid"])
                query, embedding_seconds = timed(lambda: embed(encoder, etok, [q["question"]])[0])
                ranked, search_seconds = timed(lambda: rank_vectors(query, vectors[:n], 1))
                selected = bank[ranked[0][0]]

                def tokens():
                    return (tok.encode(bank_prefix(tok, [document(selected)]), add_special_tokens=False),
                            tok.encode(query_suffix(tok, q["question"]), add_special_tokens=False))

                (pids, suffix), tokenization_seconds = timed(tokens)
                if len(pids) + len(suffix) + 64 > model.config.max_position_embeddings:
                    raise ValueError("RAG context limit")
                cache, prefill_seconds = timed(lambda: prefill(model, pids, chunk_size=256))
                (final, cache), decode_seconds = timed(lambda: decode_cached(model, tok, cache, suffix))
                seconds = (embedding_seconds + search_seconds + tokenization_seconds
                           + prefill_seconds + decode_seconds)
                trial = evaluated(q, final, seconds, sources=[selected["sid"]])
                trial.update(repeat=repeat, embedding_seconds=embedding_seconds,
                             search_seconds=search_seconds, tokenization_seconds=tokenization_seconds,
                             prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                             context_tokens=len(pids), retrieval_scores=[ranked[0][1]])
                trials.append(trial)
                cache = None
                save(folder, "rag_qa.json", dict(trials=trials))
        resource = end("rag_qa_" + str(n), t,
                       persistent_payload_bytes=sum((folder / f).stat().st_size
                                                    for f in ("bank.json", "qa_embeddings.pt")),
                       persistent_tier="CPU/disk text and vector index",
                       additional_model_bytes=embbytes)
        results[str(n)]["rag_qa"] = dict(available=True, trials=trials,
                                         summary=metrics([r for r in trials if r["repeat"] == 0]),
                                         latency=latency_summary(trials), resources=resource)
        save(out, "evaluation_progress.json", dict(results=results))
        print("RAG", n, "exact", results[str(n)]["rag_qa"]["summary"]["returned_exact"]["count"], flush=True)
    encoder = None; etok = None; vectors = None
    gc.collect(); torch.cuda.empty_cache()

    # ---- 4) KV 전체 prefix (원본과 같은 방식, 문맥 초과 시 불가 기록) ----
    if os.environ.get("EXP_SKIP_KV") == "1":
        print("KV skipped by EXP_SKIP_KV", flush=True)
        SIZES_KV = ()
    else:
        SIZES_KV = SIZES
    for n in SIZES_KV:
        folder = out / ("bank-" + str(n))
        pids = tok.encode(bank_prefix(tok, [document(r) for r in bank[:n]]), add_special_tokens=False)
        suffixes = {q["qid"]: tok.encode(query_suffix(tok, q["question"]), add_special_tokens=False)
                    for q in panel}
        required = len(pids) + max(map(len, suffixes.values())) + 64
        plan = dict(prefix_tokens=len(pids), required_tokens=required,
                    model_context_limit=model.config.max_position_embeddings,
                    bytes_per_token=kv_bytes_per_token(model.config),
                    calculated_prefix_payload_bytes=len(pids) * kv_bytes_per_token(model.config))
        save(folder, "kv_capacity.json", plan)
        if required > model.config.max_position_embeddings:
            results[str(n)]["kv_full_qa"] = dict(available=False, reason="context_limit",
                                                 plan=plan, trials=[], summary=None)
            continue
        t = start("kv_setup_" + str(n))
        try:
            cache = prefill(model, pids, chunk_size=256,
                            progress=lambda done, total: status("kv_setup", n=n, done=done, total=total))
        except torch.OutOfMemoryError as exc:
            resource = end("kv_failed_setup_" + str(n), t)
            results[str(n)]["kv_full_qa"] = dict(available=False, reason="gpu_limit", error=str(exc),
                                                 plan=plan, trials=[], summary=None, resources=resource)
            cache = None; gc.collect(); torch.cuda.empty_cache()
            continue
        actual = cache_payload_bytes(cache)
        setup = end("kv_setup_" + str(n), t, persistent_payload_bytes=actual)
        t = start("kv_full_qa_" + str(n))
        trials = []
        for repeat in range(design["repeats"]):
            order = list(panel); random.Random(100 + repeat).shuffle(order)
            for q in order:
                status("kv_full_qa", n=n, repeat=repeat, qid=q["qid"])
                (final, cache), seconds = timed(lambda: decode_cached(model, tok, cache, suffixes[q["qid"]]))
                trial = evaluated(q, final, seconds); trial.update(repeat=repeat, context_tokens=len(pids))
                trials.append(trial)
                save(folder, "kv_full_qa.json", dict(trials=trials))
        resource = end("kv_full_qa_" + str(n), t, persistent_payload_bytes=actual,
                       persistent_tier="GPU full-bank prefix cache",
                       setup_seconds=setup["elapsed_seconds"],
                       setup_peak_allocated_bytes=setup["peak_allocated_bytes"],
                       setup_peak_reserved_bytes=setup["peak_reserved_bytes"],
                       additional_model_bytes=0)
        results[str(n)]["kv_full_qa"] = dict(available=True, plan=plan, trials=trials,
                                             summary=metrics([r for r in trials if r["repeat"] == 0]),
                                             latency=latency_summary(trials), resources=resource)
        cache = None; gc.collect(); torch.cuda.empty_cache()
        save(out, "evaluation_progress.json", dict(results=results))
        print("KV", n, "exact", results[str(n)]["kv_full_qa"]["summary"]["returned_exact"]["count"], flush=True)

    save(out, "evaluation.json", dict(results=results, sizes=list(SIZES),
                                      panel_ids=[q["qid"] for q in panel],
                                      scope=design["scope"]))
    save(out, "resource_summary.json", dict(
        elapsed_seconds=time.monotonic() - begin,
        base_model_parameter_bytes=base_bytes,
        embedding_parameter_bytes=embbytes,
        measurements=measurements,
        compact_actual_bytes=sum((out / "factors" / (r["sid"] + ".pt")).stat().st_size for r in items),
        cpu_base_backup_bytes=sum(v.numel() * v.element_size() for v in compact.base.values())))
    save(out, "completion.json", dict(completed=True, model=design["model"], dtype=design["dtype"],
                                      layers=design["layers"], real_sessions=len(items),
                                      sizes=list(SIZES), base_weights_restored=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=["all", "writes", "eval"], default="all")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    out = args.out.resolve()
    if not out.exists():
        prepare(out)
    if not args.prepare_only:
        run(out, args.stage)


if __name__ == "__main__":
    main()
