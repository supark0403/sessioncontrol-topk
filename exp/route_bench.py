"""라우팅 비용만 분리 측정.

비교
A) 디스크 스트리밍 + 2GPU 샤딩 (기존 프로토콜)
B) 메모리 프리로드 + 2GPU 샤딩        → 디스크 I/O 제거
C) 메모리 프리로드 + 단일 GPU          → PCIe 왕복 제거 (가능하면)

생성은 측정하지 않는다. 순수 라우팅(키 추출 + 코사인 + 에너지)만.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import common
import patches

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "local-512"))
    ap.add_argument("--sizes", default="64,128,256,512")
    ap.add_argument("--ks", default="1,8,32,128")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    patches.install()
    run = Path(a.run)
    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    panel_file = ROOT / "data" / "prepared" / "panel8.json"
    ids = json.loads(panel_file.read_text(encoding="utf-8"))
    by_qid = {r["qid"]: r for r in items}
    panel = [by_qid[q] for q in ids]
    pid = {r["qid"] for r in panel}
    items = panel + [r for r in items if r["qid"] not in pid]

    tok, model = common.load_model()
    from compact_session_memory import capture_keys, factor_scores
    from compact_session_memory import CompactSessionMemory

    store = CompactSessionMemory(model, common.NAMES, run / "factors")
    sizes = [int(x) for x in a.sizes.split(",")]

    # 프리로드: 세션 파일을 한 번만 읽어 메모리에 둔다
    payload = {}
    t0 = time.monotonic()
    for r in items:
        payload[r["sid"]] = store.load(r["sid"])
    preload_s = time.monotonic() - t0

    keys_cache = {}
    for r in items:
        p = run / "keys" / (r["sid"] + ".pt")
        keys_cache[r["sid"]] = torch.nn.functional.normalize(
            torch.load(p, map_location="cpu", weights_only=True), dim=-1)

    out = {}
    for n in sizes:
        subset = items[:n]
        sids = [r["sid"] for r in subset]
        mat = torch.stack([keys_cache[s] for s in sids])
        route_ms, key_ms = [], []
        for q in panel:
            for rep in range(a.repeats + 1):
                t = time.monotonic()
                qk = capture_keys(model, tok, common.NAMES, [q["question"]])
                k_ms = (time.monotonic() - t) * 1000
                qv = torch.cat([qk[x][0].float().cpu() for x in common.NAMES])
                qv = torch.nn.functional.normalize(qv, dim=-1)
                cos = (mat @ qv).tolist()
                order = sorted(range(n), key=lambda i: (-cos[i], sids[i]))
                # 전수 에너지 (k=전체): 코사인/정렬은 전수 방식에 필요 없으므로 제외
                t_energy = time.monotonic()
                sc = {s: factor_scores(payload[s], qk)[0] for s in sids}
                max(sc, key=lambda s: (sc[s], s))
                energy_ms = (time.monotonic() - t_energy) * 1000
                if rep > 0:
                    route_ms.append(k_ms + energy_ms)
                    key_ms.append(k_ms)
        out[str(n)] = dict(
            key_ms=round(statistics.median(key_ms), 1),
            route_all_ms=round(statistics.median(route_ms), 1),
            n=len(panel) * a.repeats,
        )
        print(n, out[str(n)], flush=True)

    # top-k 변형: k개만 에너지
    topk = {}
    for n in sizes:
        subset = items[:n]
        sids = [r["sid"] for r in subset]
        mat = torch.stack([keys_cache[s] for s in sids])
        for k in [int(x) for x in a.ks.split(",")]:
            k = min(k, n)
            vals = []
            for q in panel:
                for rep in range(a.repeats + 1):
                    t = time.monotonic()
                    qk = capture_keys(model, tok, common.NAMES, [q["question"]])
                    qv = torch.cat([qk[x][0].float().cpu() for x in common.NAMES])
                    qv = torch.nn.functional.normalize(qv, dim=-1)
                    cos = (mat @ qv).tolist()
                    order = sorted(range(n), key=lambda i: (-cos[i], sids[i]))[:k]
                    sc = {sids[i]: factor_scores(payload[sids[i]], qk)[0] for i in order}
                    max(sc, key=lambda s: (sc[s], s))
                    if rep > 0:
                        vals.append((time.monotonic() - t) * 1000)
            topk[f"{n}/{k}"] = round(statistics.median(vals), 1)
        print("topk", n, {kk: v for kk, v in topk.items() if kk.startswith(f"{n}/")}, flush=True)

    result = dict(config=dict(shard=common.SHARD, dtype=common.DTYPE,
                              model=common.MODEL, preload_seconds=round(preload_s, 1),
                              repeats=a.repeats),
                  route=out, topk=topk)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
