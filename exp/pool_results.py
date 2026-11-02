"""2회 반복 측정 통합: 정확도 일치 확인 + 시간 표본 합산."""
import json
import statistics
from pathlib import Path

RUN = Path("runs/local-512")
a = json.loads((RUN / "evaluation_run1.json").read_text(encoding="utf-8"))
b = json.loads((RUN / "evaluation_run2.json").read_text(encoding="utf-8"))


def g(s, k):
    v = (s or {}).get(k)
    return v["count"] if isinstance(v, dict) else v


def warm_med(trials, key="query_seconds"):
    xs = [t[key] for t in trials if t["repeat"] > 0 and t.get(key) is not None]
    return statistics.median(xs) * 1000 if xs else None


out = {}
print("== 정확도 일치 확인 ==")
for n in a["sizes"]:
    n = str(n)
    for m in ("our_delta", "rag_qa"):
        x, y = a["results"][n][m]["summary"], b["results"][n][m]["summary"]
        same = (g(x, "returned_exact") == g(y, "returned_exact")
                and abs(x["mean_token_f1"] - y["mean_token_f1"]) < 1e-12)
        print(n, m, g(x, "returned_exact"), "/8", round(x["mean_token_f1"], 4), "same" if same else "DIFF")
        if not same:
            raise SystemExit("accuracy mismatch between runs")

print("\n== 시간 표본 합산 (ms, warm) ==")
for n in a["sizes"]:
    n = str(n)
    row = {"n": int(n)}
    for m in ("our_delta", "rag_qa"):
        t1 = a["results"][n][m]["trials"]
        t2 = b["results"][n][m]["trials"]
        r1, r2 = warm_med(t1), warm_med(t2)
        pool = warm_med(t1 + t2)
        row[m] = dict(run1=round(r1, 1), run2=round(r2, 1), pooled=round(pool, 1),
                      diff_pct=round(abs(r1 - r2) / ((r1 + r2) / 2) * 100, 1),
                      samples=len([t for t in t1 + t2 if t["repeat"] > 0]))
    out[n] = row
    print(n, row)

print("\n== top-k 통합 (N=512) ==")
clean = RUN.parent / "topk_clean_pooled.json"
if clean.exists():
    ck = json.loads(clean.read_text(encoding="utf-8"))
    for n, cells in ck.items():
        out.setdefault(n, {})["topk"] = {
            k: dict(pooled=v["total_ms"], route_ms=v["route_ms"],
                    recall=v["recall"], exact=v["exact"], samples=v["samples"])
            for k, v in cells.items()
        }
        print(n, {k: v["pooled"] for k, v in out[n]["topk"].items()})
else:
    print("WARNING: topk_clean_pooled.json 없음 — 구버전 top-k로 대체")
    for n in ("64", "128", "256", "512"):
        tk1 = a["results"][n].get("topk", {})
        tk2 = b["results"][n].get("topk", {})
        for k in sorted(tk1, key=int):
            t1 = tk1[k]["trials"]
            t2 = tk2.get(k, {}).get("trials", [])
            pool = warm_med(t1 + t2)
            out.setdefault(n, {}).setdefault("topk", {})[k] = dict(
                pooled=round(pool, 1),
                recall=tk1[k]["key_recall"],
                exact=g(tk1[k]["summary"], "returned_exact"),
                samples=len([t for t in t1 + t2 if t["repeat"] > 0]))
        print(n, {k: v["pooled"] for k, v in out[n]["topk"].items()})

(RUN / "pooled.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved", RUN / "pooled.json")
