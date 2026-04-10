# text-to-json

엑셀/스프레드시트 데이터를 JSON으로 추출하는 LLM 파인튜닝 파이프라인.

> **목표**: 원격 서버에서 학습 데이터를 불러오고, 모델을 파인튜닝한 뒤 평가까지 실행한다.

---

## 빠른 시작 (원격 서버)

```bash
# 1. 레포 클론
git clone https://github.com/boradorish/text-to-json.git
cd text-to-json

# 2. 가상환경 생성 및 패키지 설치
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env   # 없으면 아래 내용으로 직접 생성
```

`.env` 파일:
```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4.1-mini
```

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
│   │   ├── prepare_dataset.ipynb     # 학습 데이터셋 구성
│   │   ├── qwen3_4B_full_guide.yaml  # LLaMA-Factory 학습 설정
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
│   ├── user_prompt/                  # 학습용 사용자 요청문 (stem별 .txt)
│   ├── report/                       # 분석 보고서 (stem별 .txt)
│   ├── json/                         # 정답 JSON (stem별 .json)
│   ├── json_schema/                  # 정답 JSON Schema (stem별 .json)
│   └── json_infer/                   # 모델 추론 결과
├── models/                           # 다운로드된 모델
├── requirements.txt
└── pyproject.toml
```

---

## 전체 파이프라인

```
[데이터 준비]  HuggingFace 데이터셋 다운로드
      ↓
[학습 데이터]  prepare_dataset.ipynb 으로 sharegpt 포맷 구성
      ↓
[학습]         LLaMA-Factory (llamafactory-cli train)
      ↓
[추론]         infer.py 로 배치 추론
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

`data/user_prompt/`, `data/report/`, `data/json/` 을 읽어 sharegpt 포맷으로 변환합니다.

```bash
jupyter notebook src/train/prepare_dataset.ipynb
```

출력: `data/custom-reasoning.json`

### 3. LLaMA-Factory 설치 및 학습

```bash
# LLaMA-Factory 설치 (서버에 없는 경우)
git clone https://github.com/hiyouga/LLaMA-Factory.git /LLaMA-Factory
cd /LLaMA-Factory && pip install -e ".[torch,metrics]"

# 추가 패키지 설치 (wandb, liger-kernel 등)
bash /path/to/text-to-json/src/train/extra_install.sh

# 학습 실행
# yaml의 deepspeed 경로가 LLaMA-Factory 기준 상대경로이므로 /LLaMA-Factory에서 실행
cd /LLaMA-Factory
llamafactory-cli train /workspace/text-to-json/src/train/qwen3_4B_full_guide.yaml
```

주요 YAML 설정:

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model_name_or_path` | `Qwen/Qwen3-1.7B` | 베이스 모델 |
| `dataset` | `sunny_reasoning` | 학습 데이터셋 이름 |
| `output_dir` | `saves/qwen3-4b/full/sft` | 체크포인트 저장 위치 |
| `num_train_epochs` | `3.0` | 에폭 수 |
| `deepspeed` | `ds_z0_config.json` | 1.7B: z0 / 4B+: z2 권장 |

---

## 추론 세팅 (원격 서버)

### 1. 파인튜닝된 모델 다운로드

```bash
python src/get_model.py
```

기본으로 `boradorish/qwen3-0.6b-finetuned` 를 내려받습니다. 다른 모델을 받으려면 `get_model.py` 의 `repo_id` 를 수정하세요.

```python
# 예: 4B 모델
snapshot_download(
    repo_id="boradorish/qwen3-4b-finetuned",
    local_dir="models/qwen3-4b-finetuned",
)
```

### 2. 추론 실행

```bash
# 전체 data/user_prompt/*.txt 처리
python src/infer.py

# 단일 파일
python src/infer.py --input data/user_prompt/data1.txt

# 모델/출력 경로 직접 지정
python src/infer.py \
  --model models/qwen3-4b-finetuned \
  --output data/json_infer \
  --batch-size 16
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

결과는 `data/eval_result.json` 에 저장됩니다.

---

## 학습 데이터 포맷

`data/custom-reasoning.json` — sharegpt 포맷:

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
