import json

d = json.load(open("runs/local-512/evaluation.json", encoding="utf-8"))


def g(s, k):
    v = (s or {}).get(k)
    return v["count"] if isinstance(v, dict) else v


print("sizes", d["sizes"])
for n in map(str, d["sizes"]):
    r = d["results"][n]
    for m in ("our_delta", "rag_qa"):
        sm = r[m]["summary"]
        lat = r[m]["latency"]
        print("%3s %-9s route %s/8 exact %s/8 F1 %.4f ms %.0f" % (
            n, m, g(sm, "retrieval_at1"), g(sm, "returned_exact"),
            sm["mean_token_f1"], lat["warm_median_seconds"] * 1000))
print()
for n in map(str, [64, 128, 256, 512]):
    tk = d["results"][n].get("topk", {})
    for k in sorted(map(int, tk)):
        e = tk[str(k)]
        sm = e["summary"]
        print("N=%3s k=%3d recall %.2f route %s/8 exact %s/8 F1 %.4f ms %.0f" % (
            n, k, e["key_recall"], g(sm, "retrieval_at1"), g(sm, "returned_exact"),
            sm["mean_token_f1"], e["latency"]["warm_median_seconds"] * 1000))
    print()
print("KV:")
for n in map(str, d["sizes"]):
    kv = d["results"][n].get("kv_full_qa") or {}
    s = kv.get("summary")
    print(" ", n, kv.get("available"), kv.get("reason"),
          (g(s, "returned_exact") if s else None))
