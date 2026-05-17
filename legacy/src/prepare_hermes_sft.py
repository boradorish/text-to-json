"""
NousResearch/hermes-function-calling-v1 → LLaMA-Factory SFT 데이터 준비

데이터가 이미 sharegpt 형식(from/value)이라 변환 없이 conversations 필드만 추출합니다.

사용법:
    python src/prepare_hermes_sft.py
    python src/prepare_hermes_sft.py --num-samples 20000 --output data/sft/hermes_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()


def is_valid(conversations: list[dict]) -> bool:
    if not conversations:
        return False
    roles = {t.get("from") for t in conversations}
    return "human" in roles and "gpt" in roles


def main():
    parser = argparse.ArgumentParser(description="hermes-function-calling-v1 → SFT 데이터 준비")
    parser.add_argument("--num-samples", type=int, default=None, help="최대 샘플 수 (기본: 전체)")
    parser.add_argument("--output", default="data/sft/hermes_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("NousResearch/hermes-function-calling-v1 로드 중...")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    ds = ds.shuffle(seed=args.seed)
    print(f"총 {len(ds)}개 로드 완료\n")

    written = 0
    skipped = 0
    limit = args.num_samples or len(ds)

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="처리 중"):
            if written >= limit:
                break

            conversations = item.get("conversations", [])
            if not is_valid(conversations):
                skipped += 1
                continue

            record = {"conversations": conversations}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n완료.")
    print(f"  저장:  {written}개")
    print(f"  스킵:  {skipped}개")
    print(f"  출력: {output_path}")

    dataset_name = "hermes_sft"
    print(f"\n[LLaMA-Factory 데이터셋 등록]")
    print(f"  cp {output_path} /LLaMA-Factory/data/{output_path.name}")
    print(f"  dataset_info.json 추가:")
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
