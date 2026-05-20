"""
scrapegraphai/scrapegraphai-100k -> LLaMA-Factory SFT data.

Output format: sharegpt jsonl
  {"conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
  ]}

Usage:
    python3 src/prepare_scrapegraph_sft.py
    python3 src/prepare_scrapegraph_sft.py --num-samples 1500 --output data/sft/scrapegraph_sft_1_5k.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()

DEFAULT_SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. Extract information from "
    "web content and return only valid JSON that matches the provided JSON schema."
)


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]"


def _normalize_json_text(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(obj, ensure_ascii=False)


def build_user_prompt(row: dict[str, Any], max_content_chars: int, max_schema_chars: int) -> str | None:
    prompt = (row.get("prompt") or "").strip()
    schema = _truncate(row.get("schema") or "", max_schema_chars)
    content = _truncate(row.get("content") or "", max_content_chars)

    if not schema or not content:
        return None

    if not prompt:
        prompt = "Extract data from the content according to the JSON schema."

    return (
        f"{prompt}\n\n"
        f"JSON Schema:\n{schema}\n\n"
        f"Content:\n{content}\n\n"
        "Return ONLY valid JSON matching the schema."
    )


def convert_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if args.valid_only and row.get("response_is_valid") is False:
        return None

    user_prompt = build_user_prompt(row, args.max_content_chars, args.max_schema_chars)
    assistant = _normalize_json_text(row.get("response") or "")
    if not user_prompt or not assistant:
        return None

    return {
        "conversations": [
            {"from": "system", "value": args.system_prompt},
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant},
        ]
    }


def print_dataset_info(output_path: Path) -> None:
    dataset_name = "scrapegraph_sft"
    print("\n[LLaMA-Factory dataset_info.json entry]")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ScrapeGraphAI 100k SFT data")
    parser.add_argument("--dataset", default="scrapegraphai/scrapegraphai-100k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--output", default="data/sft/scrapegraph_sft_1_5k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-content-chars", type=int, default=12000)
    parser.add_argument("--max-schema-chars", type=int, default=8000)
    parser.add_argument("--valid-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    print(f"{args.dataset} 로드 중...")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.shuffle(seed=args.seed)
    print(f"총 {len(ds):,}개 로드 완료")

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(ds, desc="처리 중"):
            if written >= args.num_samples:
                break

            record = convert_row(row, args)
            if record is None:
                skipped += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print("\n완료.")
    print(f"  저장: {written:,}개")
    print(f"  스킵: {skipped:,}개")
    print(f"  출력: {output_path}")
    print(f"  복사: cp {output_path} ../LLaMA-Factory/data/{output_path.name}")
    print_dataset_info(output_path)


if __name__ == "__main__":
    main()
