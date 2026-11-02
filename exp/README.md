# SessionControl B파트 로컬 재현 (Llama-3.1-8B, 2×16GB)

`code/`(원본 리포)는 수정하지 않고, 복사·개조한 실행 파일만 `exp/`에 둔다.

## 최종 구성

| 파일 | 역할 |
|---|---|
| `prepare_local.py` | 원본 `scripts/prepare_experiment.py` 사본. 공개 LongMemEval + `config/session_manifest.json`으로 **원본 64세션 코호트** 복원 |
| `build_extractive.py` | 원본 `experiment_real512_compact.py`의 448 extractive 생성 규칙 포팅 |
| `build_panel.py` | 원본 512 집계와 일치하는 8문항 패널 복원 |
| `build_projection.py` | 원본 `prepare_deep_projection_gpu1.py` 수식 그대로, bf16·2GPU용 |
| `experiment_local.py` | 원본 `experiment_real512_compact.py` 사본. write → 전수 에너지 → RAG/KV |
| `topk_clean.py` | top-k 재측정. k마다 실제 생성(캐시 없음), 라우팅 타이머에 코사인·정렬 포함 |
| `route_bench.py` | 라우팅 비용만 분리(디스크/프리로드/샤딩) |
| `ttft_bench.py` | 3방식 TTFT 측정(프리로드+2GPU 샤딩) |
| `key_overhead.py` | 키 생성·저장 비용과 손익분기 |
| `kv_bench.py` | KV 전체 prefix 비교(16/16 균형 배치) |
| `common.py` | 모델 로드(샤딩), 하이퍼파라미터 |
| `patches.py` | bf16 ↔ float32 factor dtype 래퍼 (수식 불변) |

## 실행

```powershell
.\.venv\Scripts\python.exe prepare_local.py --data models/longmemeval/longmemeval_s `
  --manifest ..\code\config\session_manifest.json --out data/prepared
.\.venv\Scripts\python.exe build_extractive.py
.\.venv\Scripts\python.exe build_panel.py
.\.venv\Scripts\python.exe build_projection.py --corpus data/prepared/distractors.jsonl `
  --out projection/alphaedit-13-15.pt --layers 13,14,15 --threshold 0.02

# GPU 순서: 4070 Ti SUPER가 cuda:0, 5070 Ti가 cuda:1
$env:EXP_MODEL='...\models\Llama-3.1-8B-Instruct'
$env:EXP_LAYERS='13,14,15'; $env:EXP_DTYPE='bf16'
$env:EXP_CUDA_ORDER='1,0'; $env:EXP_SHARD='1'

.\.venv\Scripts\python.exe experiment_local.py --out runs/local-512 --stage writes
.\.venv\Scripts\python.exe experiment_local.py --out runs/local-512 --stage eval
.\.venv\Scripts\python.exe topk_clean.py --out runs/topk_clean_run1.json
.\.venv\Scripts\python.exe ttft_bench.py
```

## 원본과 같은 것

- 64세션 본문·질문·답: **corpus sha256 일치 확인** (`e23dff57...`)
- 448 extractive: 원본 규칙(9~18단어, I/My/We 시작, 앞 6단어 유일) 그대로
- 총 512세션: 중복 세션 0, 중복 질문 0
- 모델: Llama-3.1-8B-Instruct, 층 13·14·15
- write: grounded 50스텝, L2=0.1, clamp 0.75, kl 0.0625, v_loss_layer=31
- 라우팅: 질문키 1개 → 전 세션 에너지 `Σ||ΔW·k||²/Σ||ΔW||²` → argmax
- 생성: greedy 64토큰, EOS/다음턴 정지
- 비교군: Matched RAG(MiniLM+prefix prefill), KV 전체 prefix
- 측정: retrieval@1, exact, token F1

## 원본과 다른 것

| 항목 | 원본 | 여기 | 이유 |
|---|---|---|---|
| dtype | float32 | **bf16** | float32 가중치 32.1GB. 2×16GB에 활성값 여지 없음 |
| GPU | 단일 GPU | **2GPU 샤딩(21/11)** | 16GB 한 장 불가. cuda:0=4070 Ti SUPER, cuda:1=5070 Ti |
| 뱅크 | 512 (64+448) | **512** | 448 extractive 생성 완료 |
| 8문항 패널 | 비공개 pilot | **집계 일치 복원 패널** | 원문 비공개, F1/exact 집계로 역추적 |
| legacy dense 델타 | 있음(검증용) | 없음 | 미배포 → compact-dense 동등성 검증 불가 |
| TTFT | 전용 리플레이 | **프리로드+2GPU 별도 측정** | 단일 GPU는 RAG prefill OOM |
| KV | 원본 float32 | **bf16 + 16/16 균형** | N=32는 메모리 OOM |

## 최종 결과 요약 (2회 반복 통합)

| 방식 | exact | token F1 | 전체 시간 |
|---|---|---|---|
| 기존 델타(전수) | 7/8 | 0.9327 | **1332ms** |
| RAG | 7/8 | 0.9886 | **1186ms** |
| top-k k=1 | 7/8 | 0.9327 | **314ms** |
| top-k k=8 | 7/8 | 0.9327 | **337ms** |
| top-k k=128 | 7/8 | 0.9327 | **566ms** |

TTFT (프리로드+2GPU, 2회 통합 64표본):

| 방식 | TTFT | 전체 |
|---|---|---|
| 기존 델타 | 505ms | 663ms |
| RAG | 1000ms | 1202ms |
| top-k k=8 | 117ms | 271ms |

KV 전체 prefix (16/16 균형):

| N | exact | F1 | 비고 |
|---|---|---|---|
| 8 | 2/8 | 0.663 | 원본 6/8, 0.862 |
| 16 | 0/8 | 0.513 | setup 46.8초 |
| 32 | — | — | OOM |

## 해석 규칙

- 시간·정확도를 **원본 810ms(TTFT)/903ms(전체)와 직접 비교하지 않는다.** dtype·샤딩·패널이 다르다.
- top-k 표의 시간은 **답 생성 완료까지** 포함한다. 라우팅만은 `route_bench.py` 기준(프리로드+2GPU 51ms).
- 키 비용: write 오버헤드 0.33%, 손익분기 48회(키 생성 44.7ms, 512개 23.1초).
- top-k k=8 전체 시간(337ms)은 디스크에서 k개 factor를 읽는 조건이고, TTFT 표(271ms)는 프리로드 조건이다.
- `runs/route_sharded_full.json`은 N=64·128·256·512 커버리지용이다. 본문에 인용한 536→51ms는 동일 프로토콜의 512 전용 `route_sharded.json` 값이고, 전체 런은 재실행 변동으로 501.5→55.5ms를 보였다.
- KV는 패널 8문항을 뱅크 앞에 둔 뒤 재측정했다. N=8은 본 실험과 일치(2/8)하지만 원본보다 낮다.
- dense-compact 동등성과 말바꿈 질문 일반화는 아직 검증하지 않았다.
