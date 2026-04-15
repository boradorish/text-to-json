"""
JSONSchemaBench 데이터 준비 스크립트

epfl-dlab/JSONSchemaBench (train split, 5754개) 를 다운로드하고
GRPO 강화학습용 JSONL 데이터로 변환합니다.

각 샘플:
  - prompt      : 모델에 넘길 chat messages (list[dict])
  - schema_str  : reward 함수에서 스키마 검증에 사용할 JSON Schema 문자열

사용법:
    python src/prepare_jsonschemabench.py
    python src/prepare_jsonschemabench.py --split train --max-tokens 1024 --output data/grpo/jsonschemabench.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()

GRPO_SYSTEM_PROMPT = (
    "You are a JSON generation assistant. "
    "Given a JSON Schema, generate a valid JSON object that strictly conforms to it. "
    "Output ONLY the raw JSON object — no explanation, no markdown, no code fences."
)

GRPO_USER_TEMPLATE = (
    "Generate a valid JSON object conforming to the following JSON Schema:\n\n{schema}"
)


def build_messages(schema_str: str) -> list[dict]:
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": GRPO_USER_TEMPLATE.format(schema=schema_str)},
    ]


def count_tokens(tokenizer, schema_str: str) -> int:
    msgs = build_messages(schema_str)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def main():
    parser = argparse.ArgumentParser(description="JSONSchemaBench → GRPO 학습 데이터 준비")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--output",
        default="data/grpo/jsonschemabench.jsonl",
        help="출력 JSONL 경로 (기본: data/grpo/jsonschemabench.jsonl)",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="토큰 수 필터링에 사용할 토크나이저 (기본: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="프롬프트 최대 토큰 수 (초과 시 스킵, 기본: 1024)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="처리할 최대 샘플 수 (기본: 전체)",
    )
    parser.add_argument(
        "--no-token-filter",
        action="store_true",
        help="토크나이저 없이 토큰 필터를 건너뜁니다 (빠르게 실행할 때 사용)",
    )
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"epfl-dlab/JSONSchemaBench ({args.split} split) 다운로드 중...")
    ds = load_dataset("epfl-dlab/JSONSchemaBench", split=args.split)
    print(f"총 {len(ds)}개 로드 완료")

    tokenizer = None
    if not args.no_token_filter:
        print(f"토크나이저 로드 중: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    samples = ds
    if args.max_samples:
        samples = samples.select(range(min(args.max_samples, len(ds))))

    written = 0
    skipped_token = 0
    skipped_invalid = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in samples:
            schema_str: str = item["json_schema"]
            unique_id: str = item["unique_id"]

            # schema가 파싱 가능한지 확인
            try:
                schema_obj = json.loads(schema_str)
                if not isinstance(schema_obj, dict):
                    skipped_invalid += 1
                    continue
                # 빈 스키마 스킵
                if schema_obj in ({}, {"type": "object"}):
                    skipped_invalid += 1
                    continue
            except (json.JSONDecodeError, TypeError):
                skipped_invalid += 1
                continue

            # 토큰 수 필터
            if tokenizer is not None:
                n_tok = count_tokens(tokenizer, schema_str)
                if n_tok > args.max_tokens:
                    skipped_token += 1
                    continue

            messages = build_messages(schema_str)
            record = {
                "prompt": messages,
                "schema_str": schema_str,   # reward 함수에서 사용
                "unique_id": unique_id,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n완료.")
    print(f"  저장:  {written}개")
    print(f"  스킵 (토큰 초과): {skipped_token}개")
    print(f"  스킵 (스키마 오류): {skipped_invalid}개")
    print(f"  출력: {output_path}")


if __name__ == "__main__":
    main()
