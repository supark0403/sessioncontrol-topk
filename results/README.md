# Results

Every number in the root README comes from one of these files. All timings are
two-run pooled medians.

## `local-512/`

| File | What it holds |
| --- | --- |
| `design.json` | run configuration: model path, layers, dtype, shard split, bank size |
| `writes.json` | 512 write records (per-session timing, retained bytes, self-query check) |
| `measurements.json` | write-stage timings and totals |
| `evaluation.json` | per-question results for the three arms: retrieval@1, exact, token F1, time |
| `pooled.json` | pooled medians quoted in the README (full scan / RAG / top-k) |
| `resource_summary.json` | peak allocated GPU memory per arm |
| `completion.json`, `status.json` | run completion markers |

## Top level

| File | What it holds |
| --- | --- |
| `topk_clean_pooled.json` | k sweep (k=1/8/32/128), pooled — the 314/337/387/566 ms row |
| `topk_clean_run1.json`, `topk_clean_run2.json` | raw per-k runs behind the pooled file |
| `route_sharded.json` | routing-only benchmark, 512 sessions, preloaded: **536 → 51 ms** |
| `route_sharded_full.json` | same benchmark across N=64/128/256/512 (coverage check) |
| `ttft_run1.json`, `ttft_run2.json` | TTFT benchmark, preloaded factors + 2-GPU sharding |
| `key_overhead.json` | key capture/save cost and break-even (0.33 %, 48 queries) |
| `kv_balanced_fixed.json` | KV full-prefix comparison, N=8 and N=16, 16/16 balanced placement |

`local-512/evaluation.json` is large (~5 MB) because it stores full generated
text per question per run; that is the audit trail for the F1 values.
