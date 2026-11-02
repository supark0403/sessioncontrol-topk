import json

a = json.load(open("runs/local-512/evaluation.json", encoding="utf-8"))
b = json.load(open("runs/ttft3.json", encoding="utf-8"))
main = {t["qid"]: t for t in a["results"]["512"]["rag_qa"]["trials"] if t["repeat"] == 0}
tt = {t["qid"]: t for t in b["trials"]["rag"] if t["rep"] == 0}

for q in ("94f70d80", "dccbc061", "3b6f954b"):
    m, x = main[q], tt[q]
    print("QID", q)
    print("  ref  :", repr(m["answer"]))
    print("  main :", repr(m["final"]["text"]), "tokens", m["final"]["tokens"],
          "reason", m["final"]["finish_reason"])
    print("  ttft :", repr(x.get("text")), "tokens", x.get("tokens"),
          "reason", x.get("finish_reason"))
