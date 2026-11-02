import json

d = json.load(open("runs/local-512/evaluation.json", encoding="utf-8"))

for n in ("8", "64", "512"):
    t = d["results"][n]["our_delta"]["trials"]
    warm = [x for x in t if x["repeat"] > 0]
    route = sum(x["route_seconds"] for x in warm) / len(warm) * 1000
    ans = sum(x["answer_seconds"] for x in warm) / len(warm) * 1000
    print(f"delta N={n:>3} route {route:7.1f}ms  answer {ans:7.1f}ms  total {route+ans:7.1f}ms")

print()
for n in ("8", "64", "512"):
    t = d["results"][n]["rag_qa"]["trials"]
    warm = [x for x in t if x["repeat"] > 0]
    def m(k):
        return sum(x[k] for x in warm) / len(warm) * 1000
    print(f"rag   N={n:>3} embed {m('embedding_seconds'):6.1f} search {m('search_seconds'):5.1f} "
          f"tok {m('tokenization_seconds'):5.1f} prefill {m('prefill_seconds'):7.1f} "
          f"decode {m('decode_seconds'):7.1f} total {m('query_seconds'):7.1f}")
    print(f"           context tokens={warm[0]['context_tokens']}, output tokens~{warm[0]['final']['tokens']}")
