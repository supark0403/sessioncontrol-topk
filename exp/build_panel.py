"""복원 패널 생성: 원본 512 집계와 산술적으로 일치하는 8문항.

근거
- 델타 F1 0.9326923 = (7*1.0 + 0.4615385)/8
- RAG  F1 0.9886364 = (7*1.0 + 0.9090909)/8
- 64문항 실측에서 두 값을 동시에 만족하는 문항은 3d86fd0a 하나뿐
- 나머지 7개는 두 방식 모두 F1 1.0·exact True인 문항(상호 교체 가능)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
items = json.loads((ROOT / "data" / "prepared" / "items_512.json").read_text(encoding="utf-8"))
hist = json.loads((ROOT / "hist_memory_scaling.json").read_text(encoding="utf-8-sig"))
res = hist["results"]["64"]
D = {t["qid"]: t for t in res["our_delta"]["trials"]}
R = {t["qid"]: t for t in res["rag_qa"]["trials"]}

HARD = "3d86fd0a"
assert abs(D[HARD]["token_f1"] - 0.4615384615384616) < 1e-6
assert abs(R[HARD]["token_f1"] - 0.9090909090909091) < 1e-6

perfect = [r["qid"] for r in items
           if r["qid"] in D and r["qid"] in R and r["qid"] != HARD
           and D[r["qid"]]["token_f1"] == 1.0 and R[r["qid"]]["token_f1"] == 1.0
           and D[r["qid"]]["returned_exact"] and R[r["qid"]]["returned_exact"]]
print("perfect candidates:", len(perfect))

panel = [HARD] + perfect[:7]
d_f1 = sum(D[q]["token_f1"] for q in panel) / 8
r_f1 = sum(R[q]["token_f1"] for q in panel) / 8
d_ex = sum(D[q]["returned_exact"] for q in panel)
r_ex = sum(R[q]["returned_exact"] for q in panel)
print(f"reconstructed: delta exact {d_ex}/8 F1 {d_f1:.7f} | rag exact {r_ex}/8 F1 {r_f1:.7f}")
assert d_ex == 7 and r_ex == 7
assert abs(d_f1 - 0.9326923076923077) < 1e-9
assert abs(r_f1 - 0.9886363636363636) < 1e-9

out = ROOT / "data" / "prepared" / "panel8.json"
out.write_text(json.dumps(panel, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
for q in panel:
    print(" ", q, "|", next(r["question"] for r in items if r["qid"] == q)[:70])
