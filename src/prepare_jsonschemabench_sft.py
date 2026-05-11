"""
JSONSchemaBench SFT 데이터 준비 스크립트 (Baseline)

epfl-dlab/JSONSchemaBench (train split)에서 N개의 SFT 학습 데이터를 생성합니다.
각 스키마에 대해 hypothesis-jsonschema로 유효한 JSON을 생성하고 jsonschema로 검증합니다.

출력 포맷: LLaMA-Factory sharegpt
  {
    "conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt",   "value": "..."}
    ]
  }

사용법:
    python src/prepare_jsonschemabench_sft.py
    python src/prepare_jsonschemabench_sft.py --num-samples 2500 --output data/sft/jsonschemabench_sft_baseline.jsonl

LLaMA-Factory 등록 방법:
    1. 출력 JSONL을 /LLaMA-Factory/data/ 에 복사
    2. /LLaMA-Factory/data/dataset_info.json 에 아래 항목 추가:
       "jsonschemabench_sft_baseline": {
         "file_name": "jsonschemabench_sft_baseline.jsonl",
         "formatting": "sharegpt",
         "columns": {"messages": "conversations"},
         "tags": {"role_tag": "from", "content_tag": "value",
                  "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}
       }
    3. yaml의 dataset: jsonschemabench_sft_baseline 으로 설정
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()

SYSTEM_PROMPT = (
    "You are a JSON generation assistant. "
    "Given a JSON Schema, generate a valid JSON object that strictly conforms to it. "
    "Output ONLY the raw JSON object — no explanation, no markdown, no code fences."
)

USER_TEMPLATE = "Generate a valid JSON object conforming to the following JSON Schema:\n\n{schema}"

TRIVIAL_SCHEMAS = ({}, {"type": "object"}, {"type": "object", "properties": {}})


def generate_valid_json(schema_obj: dict, max_attempts: int = 5) -> object | None:
    """hypothesis-jsonschema로 유효한 JSON을 생성하고 jsonschema로 검증합니다. 실패 시 None 반환."""
    for _ in range(max_attempts):
        try:
            result = from_schema(schema_obj).example()
            jsonschema.validate(instance=result, schema=schema_obj)
            return result
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="JSONSchemaBench → SFT baseline 데이터 준비")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=2500, help="목표 샘플 수 (기본: 2500)")
    parser.add_argument(
        "--output",
        default="data/sft/jsonschemabench_sft_baseline.jsonl",
        help="출력 JSONL 경로",
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="스키마당 JSON 생성 재시도 횟수 (기본: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"epfl-dlab/JSONSchemaBench ({args.split} split) 다운로드 중...")
    ds = load_dataset("epfl-dlab/JSONSchemaBench", split=args.split)
    print(f"총 {len(ds)}개 로드 완료\n")

    written = 0
    skipped_parse = 0
    skipped_trivial = 0
    skipped_generate = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="처리 중", total=len(ds)):
            if written >= args.num_samples:
                break

            schema_str: str = item["json_schema"]
            unique_id: str = item["unique_id"]

            # 스키마 파싱 및 기본 검증
            try:
                schema_obj = json.loads(schema_str)
                if not isinstance(schema_obj, dict):
                    skipped_parse += 1
                    continue
            except (json.JSONDecodeError, TypeError):
                skipped_parse += 1
                continue

            # trivial 스키마 스킵
            if schema_obj in TRIVIAL_SCHEMAS:
                skipped_trivial += 1
                continue

            # jsf로 valid JSON 생성
            result = generate_valid_json(schema_obj, args.max_attempts)
            if result is None:
                skipped_generate += 1
                continue

            result_str = json.dumps(result, ensure_ascii=False)

            record = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": USER_TEMPLATE.format(schema=schema_str)},
                    {"from": "gpt", "value": result_str},
                ],
                "unique_id": unique_id,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n완료.")
    print(f"  저장:                  {written}개")
    print(f"  스킵 (스키마 파싱 실패): {skipped_parse}개")
    print(f"  스킵 (trivial 스키마):  {skipped_trivial}개")
    print(f"  스킵 (JSON 생성 실패):  {skipped_generate}개")
    print(f"  출력: {output_path}")

    dataset_name = "jsonschemabench_sft_baseline"
    print(f"\n[LLaMA-Factory 데이터셋 등록]")
    print(f"  1. cp {output_path} /LLaMA-Factory/data/{output_path.name}")
    print(f"  2. /LLaMA-Factory/data/dataset_info.json 에 아래 항목 추가:")
    print(json.dumps({
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()