"""
DPO JSONL 토큰 길이 분석 스크립트.

LLaMA-Factory sharegpt DPO 포맷의 prompt/chosen/rejected 길이를 측정해서
cutoff_len 설정과 OOM 위험을 판단하는 데 사용합니다.

사용법:
    python src/train/analyze_dpo_token_lengths.py \
      --data data/dpo/sunny_dpo.clean.jsonl \
      --tokenizer boradorish/qwen3-4b-new-prompt \
      --cutoffs 2048 4096 8192
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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
    messages = []
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


def chat_len(tokenizer, messages: list[dict]) -> int:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = "\n".join(message.get("content", "") for message in messages)
    return token_len(tokenizer, text)


def percentile(values: np.ndarray, pct: int) -> int:
    return int(np.percentile(values, pct)) if len(values) else 0


def print_distribution(name: str, values: list[int], cutoffs: list[int]) -> None:
    arr = np.array(values, dtype=np.int64)
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    if len(arr) == 0:
        print("  empty")
        return
    print(f"  count:  {len(arr):>10,}")
    print(f"  mean:   {arr.mean():>10,.0f}")
    print(f"  p50:    {percentile(arr, 50):>10,}")
    print(f"  p75:    {percentile(arr, 75):>10,}")
    print(f"  p90:    {percentile(arr, 90):>10,}")
    print(f"  p95:    {percentile(arr, 95):>10,}")
    print(f"  p99:    {percentile(arr, 99):>10,}")
    print(f"  max:    {arr.max():>10,}")
    for cutoff in cutoffs:
        over = int((arr > cutoff).sum())
        print(f"  > {cutoff:<5}: {over:>10,} ({over / len(arr) * 100:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO 데이터 토큰 길이 분석")
    parser.add_argument("--data", default="data/dpo/sunny_dpo.clean.jsonl")
    parser.add_argument("--tokenizer", default="boradorish/qwen3-4b-new-prompt")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--write-csv", default=None)
    args = parser.parse_args()

    data_path = resolve_project_path(args.data)
    if not data_path.exists():
        print(f"[ERROR] file not found: {data_path}")
        sys.exit(1)

    print(f"load data: {data_path}")
    records = read_jsonl(data_path)
    print(f"records: {len(records):,}")
    print(f"tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows: list[dict] = []
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    pair_max_lengths: list[int] = []
    pair_sum_lengths: list[int] = []

    for idx, record in enumerate(tqdm(records, desc="tokenize DPO")):
        stem = str(record.get("_stem") or idx)
        prompt_messages = get_conversations(record)
        prompt_len = chat_len(tokenizer, prompt_messages)
        chosen_len = token_len(tokenizer, get_value(record, "chosen"))
        rejected_len = token_len(tokenizer, get_value(record, "rejected"))

        chosen_total = prompt_len + chosen_len
        rejected_total = prompt_len + rejected_len
        pair_max = max(chosen_total, rejected_total)
        pair_sum = chosen_total + rejected_total

        row = {
            "idx": idx,
            "stem": stem,
            "prompt_tokens": prompt_len,
            "chosen_tokens": chosen_len,
            "rejected_tokens": rejected_len,
            "chosen_total_tokens": chosen_total,
            "rejected_total_tokens": rejected_total,
            "pair_max_tokens": pair_max,
            "pair_sum_tokens": pair_sum,
        }
        rows.append(row)
        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)
        pair_max_lengths.append(pair_max)
        pair_sum_lengths.append(pair_sum)

    print_distribution("prompt tokens", prompt_lengths, args.cutoffs)
    print_distribution("chosen tokens", chosen_lengths, args.cutoffs)
    print_distribution("rejected tokens", rejected_lengths, args.cutoffs)
    print_distribution("max(prompt+chosen, prompt+rejected)", pair_max_lengths, args.cutoffs)
    print_distribution("(prompt+chosen) + (prompt+rejected)", pair_sum_lengths, args.cutoffs)

    print(f"\n{'=' * 60}")
    print(f"Top {args.top_k} longest by pair_max_tokens")
    print(f"{'=' * 60}")
    for row in sorted(rows, key=lambda item: item["pair_max_tokens"], reverse=True)[: args.top_k]:
        print(
            f"  [{row['idx']}] {row['stem']}: "
            f"pair_max={row['pair_max_tokens']:,}, "
            f"prompt={row['prompt_tokens']:,}, "
            f"chosen={row['chosen_tokens']:,}, "
            f"rejected={row['rejected_tokens']:,}"
        )

    if args.write_csv:
        csv_path = resolve_project_path(args.write_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
