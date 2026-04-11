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
│   ├── get_model.py                  # HuggingFace에서 모델 다운로드
│   ├── infer.py                      # 로컬 모델로 JSON 추론 (배치)
│   ├── finetune_unsloth.py           # Unsloth LoRA 파인튜닝 (경량 대안)
│   ├── train/
│   │   ├── prepare_dataset.ipynb     # 학습 데이터셋 구성 (80/20 split)
│   │   ├── qwen3_0.6B_full_guide.yaml
│   │   ├── qwen3_1.7B_full_guide.yaml
│   │   ├── qwen3_8B_full_guide.yaml
│   │   └── extra_install.sh          # 서버 추가 패키지 설치
│   └── utils/
│       ├── parsing.py                # xlsx → 마크다운 변환
│       ├── parsing_answer.py         # LLM 응답에서 JSON/Schema 추출
│       ├── prompt_loader.py          # 프롬프트 파일 로드
│       ├── evaluate.py               # 추론 결과 평가
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
│   └── test_stems.txt                # 테스트셋 stem 목록 (prepare_dataset 생성)
├── download_from_hf.py               # HuggingFace 데이터셋 다운로드
├── upload_model_to_hf.py             # 학습된 모델 HuggingFace 업로드
├── models/                           # 다운로드된 모델
├── requirements.txt
└── pyproject.toml
```

---

## 전체 파이프라인

```
[데이터 준비]  HuggingFace 데이터셋 다운로드
      ↓
[학습 데이터]  prepare_dataset.ipynb → 80/20 split
      ↓
[학습]         LLaMA-Factory (llamafactory-cli train)
      ↓
[모델 업로드]  upload_model_to_hf.py
      ↓
[추론]         infer.py --test-only 로 test set 추론
      ↓
[평가]         evaluate.py 로 메트릭 산출
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

`data/user_prompt/`, `data/report/`, `data/json/`을 읽어 sharegpt 포맷으로 변환하고 **80/20 split** (seed=42)을 수행합니다.

```bash
# 프로젝트 루트에서 실행
jupyter nbconvert --to notebook --execute src/train/prepare_dataset.ipynb \
  --allow-root --output src/train/prepare_dataset.ipynb
```

출력:
- `/LLaMA-Factory/data/custom-reasoning.json` — 학습용 (80%)
- `data/test_stems.txt` — 테스트셋 stem 목록 (20%)

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

| yaml | 모델 | deepspeed | GPU 권장 |
|------|------|-----------|----------|
| `qwen3_0.6B_full_guide.yaml` | Qwen3-0.6B | z0 | 16GB+ |
| `qwen3_1.7B_full_guide.yaml` | Qwen3-1.7B | z0 | 24GB+ |
| `qwen3_8B_full_guide.yaml`   | Qwen3-8B   | z2 | 48GB+ |

### 4. 모델 업로드

```bash
HF_TOKEN=hf_xxx python upload_model_to_hf.py
```

---

## 추론 세팅 (원격 서버)

### 1. 파인튜닝된 모델 다운로드

```bash
python src/get_model.py
```

기본으로 `boradorish/qwen3-0.6b-finetuned`를 내려받습니다. 다른 모델을 받으려면 `get_model.py`의 `repo_id`를 수정하세요.

### 2. 추론 실행

```bash
# test set만 추론 (data/test_stems.txt 기준)
python src/infer.py --test-only

# 전체 처리
python src/infer.py

# 모델/출력 경로 직접 지정
python src/infer.py \
  --model models/qwen3-0.6b-finetuned \
  --output data/json_infer \
  --batch-size 16 \
  --test-only
```

출력:

- `data/json_infer/{stem}.json` — JSON 파싱 성공
- `data/json_infer_raw/{stem}.txt` — 파싱 실패 시 raw 텍스트

---

## 평가

```bash
# 기본 실행
python src/utils/evaluate.py

# 경로 직접 지정
python src/utils/evaluate.py \
  --pred data/json_infer \
  --gold data/json \
  --schema data/json_schema \
  --output data/eval_result.json

# LLM 기반 평가 포함
python src/utils/evaluate.py --llm --llm-model gpt-4o-mini
```

출력 메트릭:

| 메트릭 | 설명 |
|--------|------|
| `no_output_rate` | JSON 파싱 실패 비율 |
| `exact_match_rate` | 정답과 완전히 동일한 비율 |
| `schema_match_rate` | JSON Schema 검증 통과 비율 |
| `mean_noise_ratio` | Schema에 없는 여분 key 비율 |
| `mean_value_match_rule` | leaf value 정확 매칭 비율 평균 |
| `mean_value_match_llm` | LLM 채점 결과 (0–1 정규화) |

결과는 `data/eval_result.json`에 저장됩니다.

---

## 학습 데이터 포맷

`/LLaMA-Factory/data/custom-reasoning.json` — sharegpt 포맷:

```json
{
  "messages": [
    { "role": "system",    "content": "<json_SYSTEM_prompt>" },
    { "role": "user",      "content": "<user_prompt 텍스트>" },
    { "role": "assistant", "content": "<think>\n{report}\n</think>\n{json}" }
  ]
}
```

---

## HuggingFace 리소스

| 종류 | 경로 |
|------|------|
| 데이터셋 | [boradorish/text-to-json-data](https://huggingface.co/datasets/boradorish/text-to-json-data) |
| 모델 (0.6B) | [boradorish/qwen3-0.6b-finetuned](https://huggingface.co/boradorish/qwen3-0.6b-finetuned) |
| 모델 (4B) | [boradorish/qwen3-4b-finetuned](https://huggingface.co/boradorish/qwen3-4b-finetuned) |
