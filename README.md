# sessioncontrol-topk

Key-scored **top-k routing** for independent session memories of the kind used in
[SessionControl](https://github.com/juliemasters/SessionControl).

The original method scores **every** stored session at query time
(`Σ‖ΔW·k‖² / Σ‖ΔW‖²` over the whole bank). This repository adds one small
activation key per session, so routing compares keys first and only scores the
top-k candidates.

## Results (512-session bank, 2 runs pooled)

| Arm | Exact | Token F1 | Total time |
| --- | --- | --- | --- |
| Full scan (original method) | 7/8 | 0.9327 | 1332 ms |
| Matched RAG | 7/8 | 0.9886 | 1186 ms |
| **top-k, k=1** | 7/8 | 0.9327 | **314 ms** |
| **top-k, k=8** | 7/8 | 0.9327 | **337 ms** |
| top-k, k=128 | 7/8 | 0.9327 | 566 ms |

Routing cost alone (preloaded, 512 sessions): **536 ms → 51 ms** at k=8.
TTFT with preloaded factors: full scan 505 ms, RAG 1000 ms, top-k k=8 **117 ms**.

Fitting the sweep gives `T(k) ≈ 322 + 1.92k ms` (R² = 0.997): the intercept is
answer generation plus the key scan, and k only adds the per-candidate score.

## What the key is

Per session, the last-token activation at layers 13/14/15 `mlp.down_proj` input,
43,008-dim, fp32, 169.6 KiB. 512 keys = 84.8 MiB. It is the same activation the
original folds into ΔW via `one_key_factors`.

Cost: 44.7 ms to capture, 0.5 ms to save, **0.33 %** write overhead, 23.1 s to
build all 512. Break-even is **48 queries** against the bank.

## Scope and caveats

- **Routing uses the literal write-time question**, the same protocol as the
  original. `recall@1 = 1.00` is therefore by construction, not a
  generalization result. Paraphrase robustness is untested.
- Local reproduction, not the original machine: **bf16** instead of fp32,
  **2×16 GB layer-sharded** instead of a single 48 GB card.
- The 8-question panel is **reconstructed** from published aggregates; only the
  aggregates match, 7 question identities are unverified.
- Totals include disk streaming. Do not line 1332 ms up against the original's
  903 ms — the I/O conditions differ.
- Legacy dense deltas are not distributed upstream, so compact-vs-dense
  equivalence could not be verified.
- KV full-prefix comparison is below the original: 2/8 exact at N=8 (original
  6/8) under bf16, 0/8 at N=16, OOM at N=32.

## Layout

```
exp/        our scripts (the original code/ is never modified)
results/    JSON behind every number above
```

See `exp/README.md` for the file-by-file table (Korean) and the exact run order.

## Setup

The upstream repository is **not vendored here**; clone it next to `exp/`:

```bash
git clone https://github.com/juliemasters/SessionControl.git code
git -C code checkout e377e42
cd exp
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r ../code/requirements.txt
```

Model weights are not redistributed. Obtain
`meta-llama/Llama-3.1-8B-Instruct` under its provider's terms and place it under
`exp/models/`. `download.py` does this for you (gated repo, so `HF_TOKEN` is
required):

```bash
HF_TOKEN=... python download.py
```

The benchmark file is not redistributed either. The pin is the **original**
`longmemeval_s`, not `longmemeval-cleaned` — the cleaned release changes the
bytes and breaks reproduction:

```bash
huggingface-cli download xiaowu0162/longmemeval longmemeval_s \
  --repo-type dataset --local-dir exp/models/longmemeval
# sha256 08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894
```

## Run

From `exp/` (the `cd` above):

```bash
python prepare_local.py --data models/longmemeval/longmemeval_s \
  --manifest ../code/config/session_manifest.json --out data/prepared
python build_extractive.py
python build_panel.py
python build_projection.py --corpus data/prepared/distractors.jsonl \
  --out projection/alphaedit-13-15.pt --layers 13,14,15 --threshold 0.02
```

Then, with `EXP_MODEL`, `EXP_LAYERS=13,14,15`, `EXP_DTYPE=bf16`, `EXP_SHARD=1`,
`EXP_CUDA_ORDER=1,0`:

```bash
python experiment_local.py --out runs/local-512 --stage writes
python experiment_local.py --out runs/local-512 --stage eval
python topk_clean.py --out runs/topk_clean_run1.json
python ttft_bench.py
python key_overhead.py
python route_bench.py --out runs/route_sharded.json
python kv_bench.py --sizes 8,16 --out runs/kv_balanced.json
```

`EXP_CUDA_ORDER=1,0` maps cuda:0 to the RTX 4070 Ti SUPER and cuda:1 to the
RTX 5070 Ti. The layer split is 21/11.

`route_bench.py` is the source of the 536 → 51 ms routing numbers and
`key_overhead.py` of the 0.33 % / 48-query figures. `kv_bench.py` defaults to
`--sizes 8,16,32`, but N=32 OOMs on 16 GB cards, so the reported run used
`8,16`.

Both cards are needed for the reported timings. On a single card set
`EXP_SHARD=0`; 8B in bf16 leaves little activation headroom there, which is why
the single-GPU routing numbers were not trusted.

## Attribution

See [NOTICE.md](NOTICE.md). Upstream third-party licenses (AlphaEdit, ROME,
MEMIT, LongMemEval) apply to their respective material.
