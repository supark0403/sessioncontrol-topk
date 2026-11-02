"""A영역(쓰기) 키 저장 오버헤드 측정 + 손익분기 질의 수 계산."""
import json
import statistics
import time
from pathlib import Path

import torch

import common
import patches

ROOT = Path(__file__).resolve().parent


def main():
    patches.install()
    run = ROOT / "runs" / "local-512"
    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    tok, model = common.load_model()
    from compact_session_memory import capture_keys

    tmp = run / "_keybench"
    tmp.mkdir(exist_ok=True)

    cap_ms, save_ms, size = [], [], []
    for r in items[:10]:
        t = time.monotonic()
        k = capture_keys(model, tok, common.NAMES, [r["question"]])
        cap_ms.append((time.monotonic() - t) * 1000)
        vec = torch.cat([k[n][0].float().cpu() for n in common.NAMES])
        p = tmp / (r["sid"] + ".pt")
        t = time.monotonic()
        torch.save(vec, p)
        save_ms.append((time.monotonic() - t) * 1000)
        size.append(p.stat().st_size)

    for p in tmp.glob("*.pt"):
        p.unlink()
    tmp.rmdir()

    n = len(items)
    cap = statistics.median(cap_ms)
    sav = statistics.median(save_ms)
    per = statistics.median(size)

    writes = json.loads((run / "writes.json").read_text(encoding="utf-8"))
    write_s = statistics.median([w["query_seconds"] for w in writes if w.get("query_seconds")])

    bench = json.loads((ROOT / "runs" / "route_sharded.json").read_text(encoding="utf-8"))
    full = bench["route"]["512"]["route_all_ms"]
    topk8 = bench["topk"]["512/8"]
    saving = full - topk8

    total_key_s = n * (cap + sav) / 1000
    breakeven = total_key_s / (saving / 1000)

    out = dict(
        n_sessions=n,
        capture_ms=round(cap, 1), save_ms=round(sav, 1),
        key_kb=round(per / 1024, 1),
        write_median_s=round(write_s, 2),
        key_overhead_vs_write_pct=round((cap + sav) / (write_s * 1000) * 100, 3),
        total_key_build_s=round(total_key_s, 1),
        route_full_ms=full, route_topk8_ms=topk8, saving_per_query_ms=saving,
        breakeven_queries=round(breakeven, 1),
        note="route 값은 프리로드+2GPU 샤딩 기준(디스크 I/O 제거).",
    )
    (ROOT / "runs" / "key_overhead.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
