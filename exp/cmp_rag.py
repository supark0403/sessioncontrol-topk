import json

a = json.load(open("runs/local-512/evaluation.json", encoding="utf-8"))
b = json.load(open("runs/ttft3.json", encoding="utf-8"))

main = {t["qid"]: t for t in a["results"]["512"]["rag_qa"]["trials"] if t["repeat"] == 0}
tt = {}
for t in b["trials"]["rag"]:
    if t["rep"] == 0:
        tt[t["qid"]] = t

for q in main:
    m, x = main[q], tt.get(q)
    print(q, "| main pick", m["retrieved_sids"][0][:8], "exact", m["returned_exact"],
          "| ttft pick", (x["pick"][:8] if x else None), "exact", (x["exact"] if x else None))
    if x and m["final"]["text"].strip() != "":
        pass

print()
print("main text sample:", main[list(main)[0]]["final"]["text"][:80])
