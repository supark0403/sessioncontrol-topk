"""원본 512 결과(F1 0.9326923077 / exact 7)와 일치하는 8문항 부분집합 탐색."""
import itertools
import json

D_F1 = 0.9326923076923077
D_EX = 7
R_F1 = 0.9886363636363636
R_EX = 7

d = json.load(open("exp/hist_memory_scaling.json", encoding="utf-8-sig"))
res = d["results"]["64"]
delta = res["our_delta"]["trials"]
rag = res["rag_qa"]["trials"]
assert len(delta) == len(rag) == 24

df = [t["token_f1"] for t in delta]
de = [t["returned_exact"] for t in delta]
rf = [t["token_f1"] for t in rag]
re_ = [t["returned_exact"] for t in rag]

cands = []
for idx in itertools.combinations(range(24), 8):
    if sum(de[i] for i in idx) != D_EX:
        continue
    if abs(sum(df[i] for i in idx) / 8 - D_F1) > 1e-9:
        continue
    if sum(re_[i] for i in idx) != R_EX:
        continue
    if abs(sum(rf[i] for i in idx) / 8 - R_F1) > 1e-9:
        continue
    cands.append(idx)

print("candidates:", len(cands))
for idx in cands[:10]:
    print([delta[i]["qid"] for i in idx])
