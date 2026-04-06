# text-to-json

엑셀/스프레드시트 데이터를 JSON으로 추출하는 LLM 파인튜닝 파이프라인.

---

## 디렉토리 구조

```
text-to-json/
├── prompt/                      # 프롬프트 템플릿
├── src/
│   ├── process_data.py          # (데이터 생성) API로 xlsx → JSON 생성
│   ├── generate_user_prompts.py # (데이터 생성) user_prompt 텍스트 생성
│   ├── infer.py                 # (추론) 로컬 모델로 JSON 추론
│   ├── get_model.py             # HuggingFace에서 모델 다운로드
│   ├── finetune_unsloth.py      # Unsloth LoRA 파인튜닝 (대안)
│   ├── train/
│   │   ├── prepare_dataset.ipynb      # 학습 데이터셋 구성
│   │   ├── qwen3_4B_full_guide.yaml   # LLaMA-Factory 학습 설정
│   │   └── extra_install.sh           # 서버 추가 설치
│   └── utils/
│       ├── parsing.py                 # xlsx → 마크다운 변환
│       ├── parsing_answer.py          # LLM 응답 파싱 (JSON/Schema 추출)
│       ├── prompt_loader.py           # 프롬프트 파일 로드
│       ├── evaluate.py                # 추론 결과 평가
│       ├── validate_json_with_schema.py  # JSON 스키마 검증 + 불량 파일 삭제
│       ├── rename_xlsx.py             # xlsx 파일 순차 이름 변경
│       ├── rename_files_sequentially.py  # json/schema/report 번호 통일
│       ├── clean_sparse_xlsx.py       # 희소한 xlsx 파일 자동 삭제
│       ├── split_excel_sheets.py      # 시트별 xlsx 분리
│       ├── crawling.py                # data.go.kr 크롤링
│       └── crawling_google.py         # Google 검색으로 xlsx 수집
├── data/
│   ├── user_prompt/             # 학습용 사용자 요청문 (stem별 .txt)
│   ├── report/                  # 분석 보고서 (stem별 .txt)
│   ├── json/                    # 정답 JSON (stem별 .json)
│   ├── json_schema/             # 정답 JSON Schema (stem별 .json)
│   └── json_infer/              # 모델 추론 결과 (infer.py 출력)
├── sheetpedia_xlsx/             # 원본 xlsx 파일들
└── model/                       # 다운로드된 모델
```

---

## 전체 파이프라인

```
xlsx 수집 → 전처리 → API로 정답 데이터 생성 → 학습 데이터 구성 → 학습 → 추론 → 평가
```

---

## 단계별 실행 방법

### 1. 환경 설정

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
pip install litellm python-dotenv openpyxl pandas jsonschema transformers torch
```

`.env` 파일 설정:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
LLM_MODEL=gpt-4.1-mini   # 또는 gemini/gemini-2.0-flash
```

---

### 2. 데이터 수집 (선택)

```bash
# data.go.kr 크롤링
python src/utils/crawling.py

# Google 검색으로 수집
python src/utils/crawling_google.py
```

---

### 3. xlsx 전처리

```bash
# 희소한 파일 제거 (채워진 셀 비율이 낮은 파일)
python src/utils/clean_sparse_xlsx.py

# 파일 이름을 data_{num}.xlsx 규칙으로 변경
python src/utils/rename_xlsx.py
```

---

### 4. 정답 데이터 생성 (API)

xlsx 파일들을 LLM API로 처리해서 report, JSON, JSON Schema를 생성합니다.

```bash
# LLM_MODEL 환경 변수로 모델 지정 (기본: gpt-4.1-mini)
python src/process_data.py
```

출력:
- `data/report/{stem}.txt`
- `data/json/{stem}.json`
- `data/json_schema/{stem}.json`

---

### 5. user_prompt 생성

보고서를 읽고 자연스러운 사용자 요청문을 생성합니다.

```bash
python src/generate_user_prompts.py
```

출력: `data/user_prompt/{stem}.txt`

---

### 6. 학습 데이터셋 구성

```bash
jupyter notebook src/train/prepare_dataset.ipynb
```

`data/user_prompt/`, `data/report/`, `data/json/` 을 읽어 `data/custom-reasoning.json` 생성.
포맷: sharegpt (`system` + `user` + `assistant: <think>...</think>\n{json}`)

---

### 7. 학습 (LLaMA-Factory)

```bash
# 서버에서 실행
cd /LLaMA-Factory
bash src/train/extra_install.sh   # wandb, liger-kernel 설치

llamafactory-cli train src/train/qwen3_4B_full_guide.yaml
```

주요 설정 (`qwen3_4B_full_guide.yaml`):

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model_name_or_path` | `Qwen/Qwen3-1.7B` | 베이스 모델 |
| `dataset` | `sunny_reasoning` | 학습 데이터셋 이름 |
| `output_dir` | `saves/qwen3-4b/full/sft` | 체크포인트 저장 위치 |
| `num_train_epochs` | `3.0` | 에폭 수 |
| `deepspeed` | `ds_z0_config.json` | 1.7B: z0, 4B+: z2 권장 |

---

### 8. 모델 다운로드

```bash
python src/get_model.py
```

모델은 `model/qwen3-0.6b-finetuned/` 에 저장됨.

---

### 9. 추론

로컬 모델로 `data/user_prompt/*.txt` 를 처리해서 JSON을 생성합니다.

```bash
# 전체 처리
python src/infer.py

# 파일 하나만
python src/infer.py --input data/user_prompt/data1.txt

# 모델/출력 경로 지정
python src/infer.py --model model/qwen3-0.6b-finetuned --output data/json_infer
```

출력:
- `data/json_infer/{stem}.json` (파싱 성공)
- `data/json_schema_infer/{stem}.json`
- `data/json_infer_raw/{stem}.txt` (파싱 실패 시 raw 텍스트)

---

### 10. 평가

```bash
# 기본 실행
python src/utils/evaluate.py

# 경로 직접 지정
python src/utils/evaluate.py \
  --pred data/json_infer \
  --gold data/json \
  --schema data/json_schema \
  --output data/eval_result.json

# LLM 기반 평가 포함 (gemini-flash 사용)
python src/utils/evaluate.py --llm
```

출력 메트릭:

| 메트릭 | 설명 |
|--------|------|
| `no_output_rate` | JSON 파싱 실패 비율 |
| `exact_match_rate` | 정답과 완전히 동일한 비율 |
| `schema_match_rate` | JSON Schema 검증 통과 비율 |
| `mean_noise_ratio` | Schema에 없는 여분 key 비율 (schema 위반 샘플 기준) |
| `mean_value_match_rule` | leaf value 정확 매칭 비율 평균 |
| `mean_value_match_llm` | LLM이 1-5로 채점 후 0-1 정규화 평균 |

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
