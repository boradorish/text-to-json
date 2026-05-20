"""
DPO JSONL을 max(prompt+chosen, prompt+rejected) 토큰 길이 기준으로 필터링합니다.

사용법:
    python src/train/filter_dpo_by_token_length.py \
      --data data/dpo/sunny_dpo.clean.jsonl \
      --output data/dpo/sunny_dpo.max8192.jsonl \
      --tokenizer boradorish/qwen3-4b-new-prompt \
      --max-pair-tokens 8192
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root


PROJECT_ROOT = find_project_root()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: JSON parse failed: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: record must be an object")
            records.append(record)
    return records


def get_value(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, dict):
        return ""
    text = value.get("value")
    return text if isinstance(text, str) else ""


def get_conversations(record: dict) -> list[dict]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return []

    messages: list[dict] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        content = message.get("value")
        if role == "system":
            role = "system"
        elif role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        else:
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def prompt_token_len(tokenizer, record: dict) -> int:
    messages = get_conversations(record)
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = "\n".join(message.get("content", "") for message in messages)
    return token_len(tokenizer, text)


def pair_lengths(tokenizer, record: dict) -> dict:
    prompt_len = prompt_token_len(tokenizer, record)
    chosen_len = token_len(tokenizer, get_value(record, "chosen"))
    rejected_len = token_len(tokenizer, get_value(record, "rejected"))
    chosen_total = prompt_len + chosen_len
    rejected_total = prompt_len + rejected_len
    return {
        "prompt_tokens": prompt_len,
        "chosen_tokens": chosen_len,
        "rejected_tokens": rejected_len,
        "chosen_total_tokens": chosen_total,
        "rejected_total_tokens": rejected_total,
        "pair_max_tokens": max(chosen_total, rejected_total),
    }


def print_distribution(values: list[int]) -> None:
    arr = np.array(values, dtype=np.int64)
    if len(arr) == 0:
        print("  no kept records")
        return
    print(f"  kept pair_max mean: {arr.mean():,.0f}")
    print(f"  kept pair_max p50:  {np.percentile(arr, 50):,.0f}")
    print(f"  kept pair_max p90:  {np.percentile(arr, 90):,.0f}")
    print(f"  kept pair_max p95:  {np.percentile(arr, 95):,.0f}")
    print(f"  kept pair_max max:  {arr.max():,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO JSONL token length filter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="boradorish/qwen3-4b-new-prompt")
    parser.add_argument("--max-pair-tokens", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-chosen-tokens", type=int, default=None)
    parser.add_argument("--max-rejected-tokens", type=int, default=None)
    parser.add_argument("--max-per-stem", type=int, default=None)
    parser.add_argument("--write-dropped", default=None)
    args = parser.parse_args()

    data_path = resolve_project_path(args.data)
    output_path = resolve_project_path(args.output)
    if not data_path.exists():
        print(f"[ERROR] file not found: {data_path}")
        sys.exit(1)

    print(f"load data: {data_path}")
    records = read_jsonl(data_path)
    print(f"records: {len(records):,}")
    print(f"tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    kept: list[dict] = []
    dropped: list[dict] = []
    kept_by_stem: Counter = Counter()
    drop_reasons: Counter = Counter()
    kept_pair_max: list[int] = []

    for idx, record in enumerate(tqdm(records, desc="filter DPO")):
        stem = str(record.get("_stem") or idx)
        lengths = pair_lengths(tokenizer, record)

        reasons: list[str] = []
        if lengths["pair_max_tokens"] > args.max_pair_tokens:
            reasons.append("pair_max_tokens")
        if args.max_prompt_tokens is not None and lengths["prompt_tokens"] > args.max_prompt_tokens:
            reasons.append("prompt_tokens")
        if args.max_chosen_tokens is not None and lengths["chosen_tokens"] > args.max_chosen_tokens:
            reasons.append("chosen_tokens")
        if args.max_rejected_tokens is not None and lengths["rejected_tokens"] > args.max_rejected_tokens:
            reasons.append("rejected_tokens")
        if args.max_per_stem is not None and kept_by_stem[stem] >= args.max_per_stem:
            reasons.append("max_per_stem")

        if reasons:
            dropped_row = {
                "_idx": idx,
                "_stem": stem,
                "_drop_reasons": reasons,
                **lengths,
            }
            dropped.append(dropped_row)
            drop_reasons.update(reasons)
            continue

        kept.append(record)
        kept_by_stem[stem] += 1
        kept_pair_max.append(lengths["pair_max_tokens"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDone")
    print(f"  input:   {len(records):>8,}")
    print(f"  kept:    {len(kept):>8,} ({len(kept) / len(records) * 100:5.1f}%)")
    print(f"  dropped: {len(dropped):>8,} ({len(dropped) / len(records) * 100:5.1f}%)")
    print(f"  output:  {output_path}")
    if drop_reasons:
        print("  drop reasons:")
        for reason, count in drop_reasons.most_common():
            print(f"    {reason:<18} {count:>8,}")
    print_distribution(kept_pair_max)

    if args.write_dropped:
        dropped_path = resolve_project_path(args.write_dropped)
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        with dropped_path.open("w", encoding="utf-8") as f:
            for row in dropped:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  dropped report: {dropped_path}")


if __name__ == "__main__":
    main()
