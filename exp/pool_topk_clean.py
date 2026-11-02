"""topk_clean 2회 실행을 합쳐 pooled.json의 topk 시간을 갱신."""
import json
import statistics
from pathlib import Path

RUN = Path(__file__).resolve().parent / "runs"
a = json.loads((RUN / "topk_clean_run1.json").read_text(encoding="utf-8"))
b = json.loads((RUN / "topk_clean_run2.json").read_text(encoding="utf-8"))
pooled_path = RUN / "local-512" / "pooled.json"
pooled = json.loads(pooled_path.read_text(encoding="utf-8"))

out = {}
for n, cells in a["summary"].items():
    out[n] = {}
    pooled[str(n)]["topk"] = {}
    for k, cell in cells.items():
        rows = [t for src in (a, b) for t in src["trials"]
                if t["n"] == int(n) and t["k"] == int(k) and t["repeat"] > 0]
        total = round(statistics.median([t["total_ms"] for t in rows]), 1)
        route = round(statistics.median([t["route_ms"] for t in rows]), 1)
        out[n][k] = dict(total_ms=total, route_ms=route, samples=len(rows),
                         recall=cell["recall"], exact=cell["exact"], f1=cell["f1"])
        pooled[str(n)]["topk"][str(k)] = dict(
            pooled=total, route_ms=route, recall=cell["recall"],
            exact=cell["exact"], samples=len(rows))

(RUN / "topk_clean_pooled.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
pooled_path.write_text(json.dumps(pooled, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
