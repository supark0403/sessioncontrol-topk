# Attribution and license scope

This repository is an independent reproduction and extension of
[SessionControl](https://github.com/juliemasters/SessionControl). The upstream
source is **not redistributed** here: it is cloned separately into `code/` (see
README), and the pin used for the reported numbers is commit `e377e42`.

Several files in `exp/` are adapted copies of upstream scripts:

| This repo | Derived from |
| --- | --- |
| `exp/experiment_local.py` | `experiment_real512_compact.py` |
| `exp/prepare_local.py` | `scripts/prepare_experiment.py` |
| `exp/build_projection.py` | `prepare_deep_projection_gpu1.py` |
| `exp/build_extractive.py` | extractive generation rule in `experiment_real512_compact.py` |

Third-party material carries its own license, including modified copies:

| Material | Upstream | License |
| --- | --- | --- |
| `AlphaEdit/` and copied AlphaEdit hyperparameters | [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit), copyright 2025 jianghc | MIT |
| `rome/`, `util/` | [ROME](https://github.com/kmeng01/rome) / [MEMIT](https://github.com/kmeng01/memit), copyright 2022 Kevin Meng | MIT |
| Benchmark-derived aggregate results | [LongMemEval](https://github.com/xiaowu0162/LongMemEval), copyright 2024 Di Wu | MIT |

Model checkpoints are not redistributed. Obtain Llama-3.1-8B-Instruct under its
provider's terms and all-MiniLM-L6-v2 from its official model repository.
Third-party source licenses do not replace model terms.

Following the upstream project's position, **no new distribution license is
assigned to this repository's code** at this time.
