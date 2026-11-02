import json

ours = json.load(open("runs/local-512/evaluation.json", encoding="utf-8"))
hist = json.load(open("hist_memory_scaling.json", encoding="utf-8-sig"))


def g(s, k):
    v = (s or {}).get(k)
    return v["count"] if isinstance(v, dict) else v


print("=== 512 (ours vs original published) ===")
for m, orig in (("our_delta", dict(route=8, exact=7, f1=0.9326923076923077)),
                ("rag_qa", dict(route=8, exact=7, f1=0.9886363636363636))):
    sm = ours["results"]["512"][m]["summary"]
    lat = ours["results"]["512"][m]["latency"]
    print(f"{m:<9} ours route {g(sm,'retrieval_at1')} exact {g(sm,'returned_exact')} "
          f"F1 {sm['mean_token_f1']:.4f} ms {lat['warm_median_seconds']*1000:.0f} | "
          f"orig route {orig['route']} exact {orig['exact']} F1 {orig['f1']:.4f} "
          f"ttft 809.9 total 903.2" if m == "our_delta" else
          f"{m:<9} ours route {g(sm,'retrieval_at1')} exact {g(sm,'returned_exact')} "
          f"F1 {sm['mean_token_f1']:.4f} ms {lat['warm_median_seconds']*1000:.0f} | "
          f"orig route {orig['route']} exact {orig['exact']} F1 {orig['f1']:.4f} "
          f"ttft 3457.6 total 3570.2")

print("\n=== RAG 문서 길이 (context tokens) ===")
for n in ("8", "64", "512"):
    t = ours["results"][n]["rag_qa"]["trials"]
    print(" ours", n, sorted({x["context_tokens"] for x in t})[:5])

for n in ("64",):
    r = hist["results"][n]["rag_qa"]["trials"]
    print(" hist", n, "tokens", sorted({x.get("context_tokens") for x in r})[:8])

print("\n=== KV ===")
for n in ("8", "16", "32"):
    kv = ours["results"][n].get("kv_full_qa") or {}
    s = kv.get("summary")
    print(" ours", n, kv.get("available"), kv.get("reason"),
          "exact", g(s, "returned_exact") if s else None,
          "F1", round(s["mean_token_f1"], 4) if s else None)
print(" orig: 8 -> exact 6 F1 0.8616 | 16 -> exact 3 F1 0.5999 | 32 -> exact 5 F1 0.7751")
