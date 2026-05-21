# text-to-json

JSON Schema가 포함된 사용자 프롬프트에서 schema-valid JSON을 생성하도록 Qwen3 모델을 SFT/DPO/ORPO로 학습하고 평가하는 레포입니다.

## 서버 구조

원격 서버는 같은 부모 디렉토리 아래에 세 레포가 있다고 가정합니다.

```text
boradorish/
├── text-to-json/
├── LLaMA-Factory/
└── DeepJSONEval/
```

모든 명령은 `boradorish/text-to-json`에서 실행합니다.

```bash
cd text-to-json
```

## 설치

```bash
git clone <text-to-json-repo-url> text-to-json
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
git clone <DeepJSONEval-repo-url> DeepJSONEval

cd text-to-json
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd ../LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../text-to-json
```

## 데이터 받기

`text-to-json/data` 아래에 최소한 다음 디렉토리가 있어야 합니다.

```text
data/
├── user_prompt/
├── json/
└── json_schema/
```

HuggingFace에서 받을 경우:

```bash
huggingface-cli download boradorish/text-to-json-data \
  --repo-type dataset \
  --local-dir data \
  --local-dir-use-symlinks False
```

## 1. SFT 데이터 준비

기존 notebook을 실행해 SFT 학습 파일과 test split을 만듭니다.

```bash
jupyter nbconvert --to notebook --execute src/train/prepare_dataset.ipynb \
  --output prepare_dataset.executed.ipynb \
  --output-dir /tmp
```

생성물:

- `../LLaMA-Factory/data/custom-reasoning.json`
- `../LLaMA-Factory/data/dataset_info.json`의 `sunny_reasoning`
- `data/test_stems.txt`

SFT 데이터의 system prompt는 `prompt/infer_SYSTEM_prompt.txt`입니다.

## 2. SFT 학습

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 \
llamafactory-cli train src/train/qwen3_4B_full_guide.yaml
```

출력:

```text
saves/qwen3-0.6b/full/sft
```

## Glaive function calling SFT

Glaive function calling 데이터만 따로 SFT할 때는 아래 순서로 실행합니다.

```bash
python3 src/prepare_glaive_sft.py \
  --num-samples 20000 \
  --output data/sft/glaive_sft.jsonl

cp data/sft/glaive_sft.jsonl ../LLaMA-Factory/data/
```

`../LLaMA-Factory/data/dataset_info.json`에 아래 항목을 추가합니다.

```json
{
  "glaive_sft": {
    "file_name": "glaive_sft.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
}
```

등록 후 학습합니다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 \
llamafactory-cli train src/train/qwen3_4B_glaive_sft.yaml
```

Glaive/Hermes function-call JSON 응답 샘플을 기존 `custom-reasoning.json`에 추가하려면 아래를 사용합니다.

```bash
python3 src/train/add_function_calling_data.py --inspect

python3 src/train/add_function_calling_data.py \
  --glaive 5000 \
  --hermes 2000 \
  --output ../LLaMA-Factory/data/custom-reasoning.json
```

## ScrapeGraphAI 100k SFT

`scrapegraphai/scrapegraphai-100k`에서 schema-valid 응답 샘플을 약 1.5K개 잘라 Qwen3-4B LoRA SFT를 할 때는 아래 순서로 실행합니다.

```bash
python3 src/prepare_scrapegraph_sft.py \
  --num-samples 1500 \
  --output data/sft/scrapegraph_sft_1_5k.jsonl

cp data/sft/scrapegraph_sft_1_5k.jsonl ../LLaMA-Factory/data/
```

`../LLaMA-Factory/data/dataset_info.json`에 아래 항목을 추가합니다.

```json
{
  "scrapegraph_sft": {
    "file_name": "scrapegraph_sft_1_5k.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
}
```

등록 후 학습합니다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 \
llamafactory-cli train src/train/qwen3_4B_scrapegraph_sft.yaml
```

## MasterControl JSON-Unstructured-Structured SFT

`MasterControlAIML/JSON-Unstructured-Structured`는 unstructured text, JSON Schema, gold structured JSON이 같이 들어 있어 위와 같은 ShareGPT SFT 포맷으로 변환할 수 있습니다. 컬럼명이 명확하지 않은 경우에도 스크립트가 JSON Schema/gold JSON/report 필드를 값 기반으로 추론합니다.

```bash
python3 src/prepare_mastercontrol_sft.py \
  --num-samples 1500 \
  --output data/sft/mastercontrol_sft_1_5k.jsonl

cp data/sft/mastercontrol_sft_1_5k.jsonl ../LLaMA-Factory/data/
```

`../LLaMA-Factory/data/dataset_info.json`에 아래 항목을 추가합니다.

```json
{
  "mastercontrol_sft": {
    "file_name": "mastercontrol_sft_1_5k.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
}
```

등록 후 학습합니다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 \
llamafactory-cli train src/train/qwen3_4B_mastercontrol_sft.yaml
```

## 3. DPO/ORPO 데이터 생성

SFT 모델에서 여러 샘플을 생성하고, gold schema를 통과하지 못한 출력을 rejected로 저장합니다.
추론은 vLLM으로 실행됩니다.

```bash
python3 src/generate_dpo_data.py \
  --model saves/qwen3-4b/full/sft \
  --num-samples 8 \
  --batch-size 2 \
  --gpu-memory-utilization 0.9
```

출력:

```text
../LLaMA-Factory/data/sunny_dpo.jsonl
```

`../LLaMA-Factory/data/dataset_info.json`에 아래 항목이 필요합니다.

```json
{
  "sunny_dpo": {
    "file_name": "sunny_dpo.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

## 4. DPO 학습

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 \
llamafactory-cli train src/train/qwen3_0.6B_dpo.yaml
```

출력:

```text
saves/qwen3-0.6b/full/dpo
```

## 5. ORPO 학습

ORPO도 같은 `sunny_dpo` preference 데이터를 사용합니다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 \
llamafactory-cli train src/train/qwen3_0.6B_orpo.yaml
```

출력:

```text
saves/qwen3-0.6b/full/orpo
```

## 6. Infer

추론 스크립트도 vLLM을 사용합니다. 여러 GPU를 쓸 때는 `--tensor-parallel-size`를 지정합니다.

```bash
python3 src/test/infer.py \
  --model saves/qwen3-0.6b/full/sft \
  --test-only \
  --output data/infer_sft

python3 src/test/infer.py \
  --model saves/qwen3-0.6b/full/dpo \
  --test-only \
  --output data/infer_dpo

python3 src/test/infer.py \
  --model saves/qwen3-0.6b/full/orpo \
  --test-only \
  --output data/infer_orpo
```

각 실행은 `.jsonl`과 `.xlsx`를 함께 저장합니다.

## 7. Eval

```bash
python src/test/evaluate.py --input data/infer_sft.jsonl
python src/test/evaluate.py --input data/infer_dpo.jsonl
python src/test/evaluate.py --input data/infer_orpo.jsonl
```

평가 결과는 입력 파일과 같은 위치의 `.xlsx`로 저장됩니다.

## 핵심 파일

```text
prompt/infer_SYSTEM_prompt.txt
src/train/prepare_dataset.ipynb
src/train/qwen3_0.6B_full_guide.yaml
src/generate_dpo_data.py
src/train/qwen3_0.6B_dpo.yaml
src/train/qwen3_0.6B_orpo.yaml
src/test/infer.py
src/test/evaluate.py
```

이 파이프라인 밖의 예전 실험/데이터 생성/업로드 코드는 `legacy/` 아래에 보관했습니다.
