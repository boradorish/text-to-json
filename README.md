# text-to-json

엑셀/스프레드시트 데이터를 JSON으로 추출하는 LLM 파인튜닝 파이프라인.

> **목표**: 원격 서버에서 학습 데이터를 불러오고, 모델을 파인튜닝한 뒤 평가까지 실행한다.

---

## 빠른 시작 (원격 서버)

```bash
# 1. 레포 클론
git clone https://github.com/boradorish/text-to-json.git
cd text-to-json

# 2. 패키지 설치 (서버 전용 환경이면 가상환경 없이 시스템 파이썬 사용 권장)
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env   # 없으면 아래 내용으로 직접 생성

# 4. 베이스 모델 다운로드 (HuggingFace에서 자동 다운로드)
# 학습 실행 시 model_name_or_path에 지정한 HuggingFace repo ID로 자동 다운로드됩니다.
# 예: boradorish/qwen3-4b-finetuned
# 수동으로 미리 받으려면:
huggingface-cli download boradorish/qwen3-4b-finetuned --local-dir models/qwen3-4b-finetuned
```

`.env` 파일:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4.1-mini
```

> **GPU 주의**: `Max CUDA 12.x` 이하인 서버를 사용해야 합니다. RTX 5060 Ti / RTX 4080S 등 Blackwell 아키텍처(CUDA 13.0)는 PyTorch 안정 버전과 호환되지 않습니다. RTX 4090, A100, H100 권장.

---

## 디렉토리 구조

```
text-to-json/
├── prompt/                           # 프롬프트 템플릿
├── src/
│   ├── process_data.py               # xlsx → report/JSON/Schema 생성 (API)
│   ├── generate_user_prompts.py      # user_prompt 텍스트 생성 (API)
│   ├── prepare_jsonschemabench.py    # JSONSchemaBench → GRPO 학습 데이터 준비
│   ├── grpo_train.py                 # GRPO 강화학습 (스키마 준수 reward)
│   ├── generate_rft_data.py          # Rejection Sampling SFT 데이터 생성
│   ├── generate_dpo_data.py          # DPO 데이터 생성 (chosen/rejected 쌍)
│   ├── get_model.py                  # HuggingFace에서 모델 다운로드
│   ├── finetune_unsloth.py           # Unsloth LoRA 파인튜닝 (경량 대안)
│   ├── test/
│   │   ├── make_test_split.py        # data/test_stems.txt 생성 (seed=42, 10% hold-out)
│   │   ├── infer.py                  # 로컬 모델로 JSON 추론 (배치, --test-only 지원)
│   │   └── evaluate.py               # 추론 결과 평가 (exact/schema/value match)
│   ├── train/
│   │   ├── prepare_dataset.ipynb     # 학습 데이터셋 구성 (90/10 split)
│   │   ├── qwen3_0.6B_full_guide.yaml
│   │   ├── qwen3_1.7B_full_guide.yaml
│   │   ├── qwen3_8B_full_guide.yaml
│   │   ├── qwen3_8B_rft.yaml         # Rejection Sampling SFT 학습 설정
│   │   ├── qwen3_8B_dpo.yaml         # DPO 학습 설정
│   │   ├── qwen3_8B_grpo_sft.yaml    # GRPO 이후 SFT 학습 설정
│   │   └── extra_install.sh          # 서버 추가 패키지 설치
│   └── utils/
│       ├── parsing.py                # xlsx → 마크다운 변환
│       ├── parsing_answer.py         # LLM 응답에서 JSON/Schema 추출
│       ├── prompt_loader.py          # 프롬프트 파일 로드
│       ├── validate_json_with_schema.py  # JSON 스키마 검증 + 불량 파일 삭제
│       ├── crawling.py               # data.go.kr xlsx 크롤링
│       ├── crawling_google.py        # Google 검색으로 xlsx 수집
│       ├── clean_sparse_xlsx.py      # 희소한 xlsx 자동 삭제
│       ├── rename_xlsx.py            # xlsx 파일 순차 이름 변경
│       ├── rename_files_sequentially.py  # json/schema/report 번호 통일
│       └── split_excel_sheets.py     # 시트별 xlsx 분리
├── data/
│   ├── user_prompt/                  # 사용자 요청문 (stem별 .txt)
│   ├── user_prompt_question/         # 요청문 (질문 형태)
│   ├── report/                       # 분석 보고서 (stem별 .txt)
│   ├── json/                         # 정답 JSON (stem별 .json)
│   ├── json_schema/                  # 정답 JSON Schema (stem별 .json)
│   ├── json_infer/                   # 모델 추론 결과
│   ├── grpo/                         # GRPO 학습 데이터
│   │   └── jsonschemabench.jsonl
│   ├── rft/                          # Rejection Sampling SFT 학습 데이터
│   │   └── sunny_rft.jsonl
│   ├── dpo/                          # DPO 학습 데이터
│   │   └── sunny_dpo.jsonl
│   └── test_stems.txt                # 테스트셋 stem 목록 (prepare_dataset 생성)
├── download_from_hf.py               # HuggingFace 데이터셋 다운로드
├── upload_model_to_hf.py             # 학습된 모델 HuggingFace 업로드
├── models/                           # 다운로드된 모델
├── requirements.txt
└── pyproject.toml
```

---

## 전체 파이프라인

두 가지 파이프라인을 선택할 수 있습니다.

### A. 기본 파이프라인 (SFT → RFT/DPO)

```
[데이터 준비]  HuggingFace 데이터셋 다운로드
      ↓
[학습 데이터]  prepare_dataset.ipynb → 90/10 split
      ↓
[1단계 학습]   LLaMA-Factory SFT (qwen3_*_full_guide.yaml)
      ↓
[스키마 강화]  RFT / DPO 데이터 생성 → 2단계 학습 (선택)
      ↓
[모델 업로드]  upload_model_to_hf.py
      ↓
[테스트 split] make_test_split.py → data/test_stems.txt
      ↓
[추론]         src/test/infer.py --test-only 로 test set 추론
      ↓
[평가]         src/test/evaluate.py 로 메트릭 산출
```

### B. GRPO → SFT 파이프라인 (스키마 준수 사전 학습)

베이스 모델에 JSONSchemaBench로 GRPO 강화학습을 먼저 수행해 스키마 준수 능력을 키운 뒤 SFT를 진행합니다.

```
[RL 데이터]    prepare_jsonschemabench.py (JSONSchemaBench 다운로드)
      ↓
[GRPO 학습]    grpo_train.py — 스키마 → 유효 JSON 생성 reward
      ↓
[SFT 데이터]   generate_rft_data.py --model <grpo 출력>
      ↓
[SFT 학습]     LLaMA-Factory (qwen3_8B_grpo_sft.yaml)
      ↓
[추론 / 평가]  src/test/infer.py → src/test/evaluate.py
```

---

## 학습 세팅 (원격 서버)

### 1. 학습 데이터 다운로드

학습 데이터는 HuggingFace([boradorish/text-to-json-data](https://huggingface.co/datasets/boradorish/text-to-json-data))에서 관리합니다.

```bash
python download_from_hf.py
```

private 레포인 경우 HF 토큰 필요:

```bash
HF_TOKEN=hf_xxx python download_from_hf.py
```

실행하면 `data/` 하위에 다음 디렉토리가 생성됩니다:

```
data/
├── user_prompt/           # 사용자 요청문
├── user_prompt_question/  # 요청문 (질문 형태)
├── report/                # 분석 보고서
├── json/                  # 정답 JSON
└── json_schema/           # 정답 JSON Schema
```

### 2. 학습 데이터셋 구성

`data/user_prompt/`, `data/report/`, `data/json/`을 읽어 sharegpt 포맷으로 변환하고 **90/10 split** (seed=42)을 수행합니다.

```bash
# 프로젝트 루트에서 실행
jupyter nbconvert --to notebook --execute src/train/prepare_dataset.ipynb \
  --allow-root --output src/train/prepare_dataset.ipynb
```

출력:

- `/LLaMA-Factory/data/custom-reasoning.json` — 학습용 (90%)
- `data/test_stems.txt` — 테스트셋 stem 목록 (10%)

### 3. LLaMA-Factory 설치 및 학습

```bash
# LLaMA-Factory 설치 (서버에 없는 경우)
git clone https://github.com/hiyouga/LLaMA-Factory.git /LLaMA-Factory
cd /LLaMA-Factory && pip install -e ".[torch,metrics]"

# 추가 패키지 설치 (wandb, liger-kernel)
bash /workspace/text-to-json/src/train/extra_install.sh

# wandb 로그인 (최초 1회, https://wandb.ai/authorize 에서 발급)
wandb login

# 학습 실행 (DeepSpeed는 FORCE_TORCHRUN=1 필수)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 llamafactory-cli train /workspace/text-to-json/src/train/qwen3_0.6B_full_guide.yaml
```

체크포인트에서 재시작:

```bash
# yaml의 resume_from_checkpoint에 경로 지정 후 실행
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 llamafactory-cli train /workspace/text-to-json/src/train/qwen3_0.6B_full_guide.yaml
```

모델별 yaml 및 권장 설정:

| yaml                         | 모델       | deepspeed | GPU 권장 |
| ---------------------------- | ---------- | --------- | -------- |
| `qwen3_0.6B_full_guide.yaml` | Qwen3-0.6B | z0        | 16GB+    |
| `qwen3_1.7B_full_guide.yaml` | Qwen3-1.7B | z0        | 24GB+    |
| `qwen3_8B_full_guide.yaml`   | Qwen3-8B   | z2        | 48GB+    |

### 4. 모델 업로드

```bash
HF_TOKEN=hf_xxx python upload_model_to_hf.py
```

---

## GRPO → SFT 파이프라인 (스키마 준수 사전 학습)

베이스 모델에 JSONSchemaBench (epfl-dlab/JSONSchemaBench, 5754개 스키마)로 GRPO 강화학습을 먼저 수행해 스키마 준수 능력을 키운 뒤 SFT를 진행합니다.

> **reward 설계**: 1.0 (유효 JSON + 스키마 통과) / 0.3 (유효 JSON, 스키마 불통과) / 0.0 (파싱 불가)

### 1. RL 학습 데이터 준비

```bash
python src/prepare_jsonschemabench.py
```

옵션:

| 옵션                | 기본값                            | 설명                                           |
| ------------------- | --------------------------------- | ---------------------------------------------- |
| `--split`           | `train`                           | 사용할 데이터 split (`train` / `val` / `test`) |
| `--tokenizer`       | `Qwen/Qwen3-8B`                   | 토큰 수 필터링에 사용할 토크나이저             |
| `--max-tokens`      | `1024`                            | 프롬프트 최대 토큰 수 (초과 스키마 제외)       |
| `--max-samples`     | 전체                              | 샘플 수 제한 (빠른 실험용)                     |
| `--no-token-filter` | —                                 | 토크나이저 없이 전체 저장                      |
| `--output`          | `data/grpo/jsonschemabench.jsonl` | 출력 경로                                      |

출력: `data/grpo/jsonschemabench.jsonl`

### 2. GRPO 학습

TRL의 `GRPOTrainer`를 사용하며 단일/멀티 GPU 모두 지원합니다.

```bash
# 단일 GPU
python src/grpo_train.py --model Qwen/Qwen3-8B

# 멀티 GPU (accelerate)
accelerate launch src/grpo_train.py --model Qwen/Qwen3-8B

# DeepSpeed ZeRO-3 (대용량 모델)
accelerate launch --config_file /path/to/zero3.yaml \
    src/grpo_train.py \
    --model Qwen/Qwen3-8B \
    --per-device-batch-size 1 \
    --grad-accum 8
```

주요 옵션:

| 옵션                      | 기본값                                | 설명                             |
| ------------------------- | ------------------------------------- | -------------------------------- |
| `--model`                 | `Qwen/Qwen3-8B`                       | 베이스 모델 경로 또는 HF repo ID |
| `--data`                  | `data/grpo/jsonschemabench.jsonl`     | RL 학습 데이터 경로              |
| `--output-dir`            | `saves/qwen3-8b/grpo/jsonschemabench` | 모델 저장 경로                   |
| `--num-generations`       | `4`                                   | GRPO 그룹 크기 G                 |
| `--beta`                  | `0.01`                                | KL 페널티 계수                   |
| `--learning-rate`         | `1e-6`                                | 학습률                           |
| `--max-prompt-length`     | `1024`                                | 프롬프트 최대 토큰 수            |
| `--max-completion-length` | `2048`                                | 생성 최대 토큰 수                |
| `--max-samples`           | 전체                                  | 샘플 수 제한 (디버그용)          |
| `--deepspeed`             | —                                     | DeepSpeed config 경로            |

출력: `saves/qwen3-8b/grpo/jsonschemabench/`

### 3. SFT 데이터 생성

GRPO 학습된 모델로 기존 user_prompt에 대해 샘플을 생성하고, 스키마 통과한 것만 추출합니다.

```bash
python src/generate_rft_data.py \
    --model saves/qwen3-8b/grpo/jsonschemabench \
    --num-samples 4 \
    --batch-size 4
```

출력: `data/rft/sunny_rft.jsonl`

### 4. SFT 학습

생성된 JSONL을 LLaMA-Factory에 등록한 뒤 학습합니다.

```bash
# 1. JSONL 복사
cp data/rft/sunny_rft.jsonl /LLaMA-Factory/data/

# 2. dataset_info.json 업데이트 (generate_rft_data.py 실행 시 출력된 항목 참고)

# 3. SFT 학습 실행
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 llamafactory-cli train src/train/qwen3_8B_grpo_sft.yaml
```

> `qwen3_8B_grpo_sft.yaml`의 `model_name_or_path`가 GRPO 출력 경로(`saves/qwen3-8b/grpo/jsonschemabench`)로 설정되어 있는지 확인하세요.

출력: `saves/qwen3-8b/full/grpo_then_sft/`

---

## 스키마 준수 강화: RFT / DPO (선택)

SFT 모델이 JSON Schema를 잘 따르지 않는 경우, Rejection Sampling SFT 또는 DPO로 추가 학습할 수 있습니다.

### Rejection Sampling SFT (RFT)

학습된 모델로 여러 샘플을 생성하고, 스키마 검증을 통과한 것만 새 SFT 데이터로 활용합니다.

```bash
# 1. 데이터 생성 (스키마 통과한 샘플만 추출)
python src/generate_rft_data.py \
  --model saves/qwen3-8b/full/sft \
  --num-samples 4 \
  --batch-size 4 \
  --max-prompts 5000

# 2. 생성된 JSONL을 LLaMA-Factory에 등록 후 학습
FORCE_TORCHRUN=1 llamafactory-cli train src/train/qwen3_8B_rft.yaml
```

출력: `data/rft/sunny_rft.jsonl`

### DPO

Gold 데이터를 chosen으로, 모델이 생성했지만 스키마 불통과한 출력을 rejected로 사용합니다.

```bash
# 1. chosen/rejected 쌍 생성
python src/generate_dpo_data.py \
  --model saves/qwen3-8b/full/sft \
  --num-samples 8 \
  --batch-size 2 \
  --max-prompts 5000

# 2. 생성된 JSONL을 LLaMA-Factory에 등록 후 학습
#    qwen3_8B_dpo.yaml의 model_name_or_path를 SFT 체크포인트 경로로 수정
FORCE_TORCHRUN=1 llamafactory-cli train src/train/qwen3_8B_dpo.yaml
```

출력: `data/dpo/sunny_dpo.jsonl`

### LLaMA-Factory 데이터셋 등록

두 스크립트 모두 실행 종료 시 `dataset_info.json`에 추가할 항목을 출력합니다. `/LLaMA-Factory/data/dataset_info.json`에 붙여넣고 JSONL 파일을 `/LLaMA-Factory/data/`에 복사하세요.

### 파라미터 요약

| 스크립트               | 주요 옵션                | 기본값 |
| ---------------------- | ------------------------ | ------ |
| `generate_rft_data.py` | `--num-samples`          | 4      |
| `generate_rft_data.py` | `--temperature`          | 0.8    |
| `generate_dpo_data.py` | `--num-samples`          | 8      |
| `generate_dpo_data.py` | `--temperature`          | 0.9    |
| `generate_dpo_data.py` | `--max-pairs-per-prompt` | 3      |

---

## 추론 및 평가 (원격 서버)

### 1. 파인튜닝된 모델 다운로드

```bash
python src/get_model.py
```

기본으로 `boradorish/qwen3-0.6b-finetuned`를 내려받습니다. 다른 모델을 받으려면 `get_model.py`의 `repo_id`를 수정하세요.

### 2. 테스트 split 생성

`prepare_dataset.ipynb`과 동일한 `random.seed(42)`로 전체 데이터를 섞은 뒤, 하위 10%를 테스트셋으로 분리합니다.

```bash
python src/test/make_test_split.py
```

출력: `data/test_stems.txt` (테스트셋 stem 목록)

> `prepare_dataset.ipynb`을 이미 실행해 `test_stems.txt`가 있다면 이 단계는 건너뛰어도 됩니다.

### 3. 추론 실행

```bash
# test set만 추론 (data/test_stems.txt 기준)
python src/test/infer.py --test-only

# 전체 처리
python src/test/infer.py

# 모델/출력 경로 직접 지정
python src/test/infer.py \
  --model model/qwen3-4b-finetuned \
  --output data/infer_results.xlsx \
  --batch-size 16 \
  --test-only
```

주요 옵션:

| 옵션               | 기본값                        | 설명                                   |
| ------------------ | ----------------------------- | -------------------------------------- |
| `--model`          | `models/qwen3-0.6b-finetuned` | 모델 경로 또는 HF repo ID              |
| `--output`         | `data/infer_results.xlsx`     | Excel 출력 경로                        |
| `--batch-size`     | `32`                          | 배치 크기                              |
| `--max-new-tokens` | `4096`                        | 최대 생성 토큰 수                      |
| `--test-only`      | —                             | `data/test_stems.txt` 기준 파일만 처리 |

출력: `data/infer_results.xlsx` — 아래 컬럼을 포함한 단일 Excel 파일

| 컬럼          | 내용                             |
| ------------- | -------------------------------- |
| `stem`        | 파일 이름 (확장자 제외)          |
| `user_prompt` | 사용자 입력 텍스트               |
| `gold_json`   | 정답 JSON (GT)                   |
| `json_schema` | JSON Schema                      |
| `raw_output`  | 모델 raw 출력                    |
| `pred_json`   | 파싱된 JSON (파싱 실패 시 빈 칸) |

---

### 4. 평가

`infer.py`가 생성한 Excel을 읽어 메트릭을 계산하고 결과를 같은 파일에 덮어씁니다.
**JSON 파싱 실패(`pred_json` 비어 있음)는 모든 메트릭 0으로 집계**됩니다.

```bash
# 기본 실행
python src/test/evaluate.py

# 입력/출력 경로 지정
python src/test/evaluate.py \
  --input data/infer_results.xlsx \
  --output data/eval_results.xlsx

# LLM 기반 평가 포함 (OPENAI_API_KEY 필요)
python src/test/evaluate.py --llm --llm-model gpt-4o-mini
```

주요 옵션:

| 옵션          | 기본값                    | 설명                           |
| ------------- | ------------------------- | ------------------------------ |
| `--input`     | `data/infer_results.xlsx` | infer.py 출력 Excel            |
| `--output`    | (input과 동일)            | 결과 저장 경로                 |
| `--llm`       | —                         | GPT 기반 value_match 추가 평가 |
| `--llm-model` | `gpt-4o-mini`             | 사용할 OpenAI 모델             |

추가되는 메트릭 컬럼 (Excel에 덧붙여 저장):

| 컬럼           | 설명                                           |
| -------------- | ---------------------------------------------- |
| `no_output`    | JSON 파싱 실패 여부                            |
| `exact_match`  | 정답과 완전히 동일한지 여부                    |
| `schema_valid` | JSON Schema 검증 통과 여부                     |
| `noise_ratio`  | Schema에 없는 여분 key 비율 (schema 불통과 시) |
| `value_match`  | gold leaf value 중 정확 매칭 비율              |
| `llm_score`    | GPT 채점 결과 0–1 정규화 (`--llm` 시에만)      |

---

## 학습 데이터 포맷

`/LLaMA-Factory/data/custom-reasoning.json` — sharegpt 포맷:

```json
{
  "messages": [
    { "role": "system", "content": "<json_SYSTEM_prompt>" },
    { "role": "user", "content": "<user_prompt 텍스트>" },
    { "role": "assistant", "content": "<think>\n{report}\n</think>\n{json}" }
  ]
}
```

---

## HuggingFace 리소스

| 종류        | 경로                                                                                         |
| ----------- | -------------------------------------------------------------------------------------------- |
| 데이터셋    | [boradorish/text-to-json-data](https://huggingface.co/datasets/boradorish/text-to-json-data) |
| 모델 (0.6B) | [boradorish/qwen3-0.6b-finetuned](https://huggingface.co/boradorish/qwen3-0.6b-finetuned)    |
| 모델 (4B)   | [boradorish/qwen3-4b-finetuned](https://huggingface.co/boradorish/qwen3-4b-finetuned)        |
