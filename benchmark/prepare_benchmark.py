"""
Build a fixed text-to-json benchmark split under benchmark/.

The default path mirrors src/train/prepare_dataset.ipynb:
1. load valid local data rows sorted by user_prompt stem
2. shuffle with seed=42
3. keep the 10% test split
4. optionally filter the test split by max input tokens
5. select a seed-stable random sample from that filtered test split

Outputs:
  benchmark/benchmark_samples.jsonl
  benchmark/benchmark_samples.xlsx
  benchmark/test_stems.txt
  benchmark/hf_splits/train.jsonl
  benchmark/hf_splits/test.jsonl
  benchmark/benchmark_metadata.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ID = "boradorish/text-to-json-data"
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = "prompt/infer_SYSTEM_prompt.txt"
EXCEL_TRUNCATE = 32000


@dataclass
class BenchmarkRow:
    stem: str
    user_prompt: str
    gold_json: str
    json_schema: str
    input_tokens: int
    user_tokens: int
    gold_tokens: int
    total_tokens: int
    source_split: str


def make_benchmark_row(
    row: dict,
    tokenizer_bundle: tuple[str, Any, str],
    system_prompt: str,
    *,
    source_split: str,
) -> BenchmarkRow:
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["user_prompt"]},
    ]
    full_messages = [*input_messages, {"role": "assistant", "content": row["gold_json"]}]
    return BenchmarkRow(
        stem=row["stem"],
        user_prompt=row["user_prompt"],
        gold_json=row["gold_json"],
        json_schema=row["json_schema"],
        input_tokens=token_len(
            tokenizer_bundle,
            chat_text(tokenizer_bundle, input_messages, add_generation_prompt=True),
        ),
        user_tokens=token_len(tokenizer_bundle, row["user_prompt"]),
        gold_tokens=token_len(tokenizer_bundle, row["gold_json"]),
        total_tokens=token_len(
            tokenizer_bundle,
            chat_text(tokenizer_bundle, full_messages, add_generation_prompt=False),
        ),
        source_split=source_split,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed benchmark sample files.")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--dataset", default=REPO_ID, help="HF dataset id when --source hf")
    parser.add_argument("--split", default="train", help="HF split when --source hf")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="benchmark")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Keep only test rows at or below this input token length.")
    parser.add_argument("--write-hf-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xlsx", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
            return ("transformers", tokenizer, tokenizer_id)
        except Exception:
            if "/" in tokenizer_id:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{tokenizer_id.replace('/', '--')}"
                ref_path = cache_dir / "refs" / "main"
                if ref_path.exists():
                    snapshot = cache_dir / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
                    if snapshot.exists():
                        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
                        return ("transformers", tokenizer, tokenizer_id)
            raise
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] tokenizer 로드 실패: {tokenizer_id} ({exc})")
        print("[WARN] regex 기반 fallback token count를 사용합니다.")
        return ("regex", None, "regex_fallback")


def token_len(tokenizer_bundle: tuple[str, Any, str], text: str) -> int:
    kind, tokenizer, _ = tokenizer_bundle
    if kind == "transformers":
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return int(math.ceil(len(pieces) * 1.15))


def chat_text(tokenizer_bundle: tuple[str, Any, str], messages: list[dict], *, add_generation_prompt: bool) -> str:
    kind, tokenizer, _ = tokenizer_bundle
    if kind == "transformers":
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
    return text + ("\nassistant:" if add_generation_prompt else "")


def dump_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except Exception:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def load_local_rows(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for user_file in sorted((data_dir / "user_prompt").glob("*.txt")):
        stem = user_file.stem
        gold_path = data_dir / "json" / f"{stem}.json"
        schema_path = data_dir / "json_schema" / f"{stem}.json"
        if not gold_path.exists() or not schema_path.exists():
            continue
        try:
            gold_json = dump_json(json.loads(gold_path.read_text(encoding="utf-8")))
            json_schema = dump_json(json.loads(schema_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "stem": stem,
                "user_prompt": user_file.read_text(encoding="utf-8"),
                "gold_json": gold_json,
                "json_schema": json_schema,
            }
        )
    return rows


def load_hf_rows(dataset: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install datasets first: pip install datasets") from exc

    token = os.environ.get("HF_TOKEN") or None
    ds = load_dataset(dataset, split=split, token=token)
    rows: list[dict] = []
    for row in ds:
        stem = str(row["id"])
        if not row.get("user_prompt") or not row.get("json") or not row.get("json_schema"):
            continue
        rows.append(
            {
                "stem": stem,
                "user_prompt": str(row["user_prompt"]),
                "gold_json": dump_json(row["json"]),
                "json_schema": dump_json(row["json_schema"]),
            }
        )
    return rows


def add_lengths(
    rows: list[dict],
    tokenizer_bundle: tuple[str, Any, str],
    system_prompt: str,
    *,
    source_split: str,
) -> list[BenchmarkRow]:
    return [
        make_benchmark_row(row, tokenizer_bundle, system_prompt, source_split=source_split)
        for row in rows
    ]


def write_jsonl(rows: list[BenchmarkRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_split_jsonl(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, path)


def write_plain_split_jsonl(rows: list[dict], path: Path, *, source_split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "stem": row["stem"],
                "user_prompt": row["user_prompt"],
                "gold_json": row["gold_json"],
                "json_schema": row["json_schema"],
                "input_tokens": None,
                "user_tokens": None,
                "gold_tokens": None,
                "total_tokens": None,
                "source_split": source_split,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_xlsx(rows: list[BenchmarkRow], path: Path) -> None:
    records = []
    for row in rows:
        record = asdict(row)
        for col in ("user_prompt", "gold_json", "json_schema"):
            if isinstance(record[col], str) and len(record[col]) > EXCEL_TRUNCATE:
                record[col] = record[col][:EXCEL_TRUNCATE] + "..."
        records.append(record)
    pd.DataFrame(records).to_excel(path, index=False)


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    system_prompt = resolve_path(args.system_prompt).read_text(encoding="utf-8")

    source_rows = load_local_rows(data_dir) if args.source == "local" else load_hf_rows(args.dataset, args.split)
    shuffled = list(source_rows)
    random.Random(args.seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * args.train_ratio)
    test_rows = shuffled[split_idx:]
    train_rows = shuffled[:split_idx]

    tokenizer_bundle = load_tokenizer(args.tokenizer)
    measured = add_lengths(test_rows, tokenizer_bundle, system_prompt, source_split="test")
    if args.max_input_tokens is not None:
        measured = [row for row in measured if row.input_tokens <= args.max_input_tokens]
    selection_pool = sorted(measured, key=lambda row: row.stem)
    random.Random(args.seed).shuffle(selection_pool)
    selected = selection_pool[: args.count]

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "benchmark_samples.jsonl"
    xlsx_path = output_dir / "benchmark_samples.xlsx"
    stems_path = output_dir / "test_stems.txt"
    metadata_path = output_dir / "benchmark_metadata.json"
    train_split_path = output_dir / "hf_splits" / "train.jsonl"
    test_split_path = output_dir / "hf_splits" / "test.jsonl"

    write_jsonl(selected, jsonl_path)
    stems_path.write_text("\n".join(row.stem for row in selected) + "\n", encoding="utf-8")
    if args.xlsx:
        write_xlsx(selected, xlsx_path)
    if args.write_hf_splits:
        write_plain_split_jsonl(train_rows, train_split_path, source_split="train")
        write_split_jsonl(selected, test_split_path)

    _, _, tokenizer_name = tokenizer_bundle
    metadata = {
        "source": args.source,
        "dataset": args.dataset if args.source == "hf" else args.data_dir,
        "hf_split": args.split if args.source == "hf" else None,
        "loaded_valid_rows": len(source_rows),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_rows": split_idx,
        "test_rows": len(test_rows),
        "selection": "random_from_test_split_after_max_input_token_filter",
        "requested_count": args.count,
        "selected_count": len(selected),
        "candidate_rows_after_token_filter": len(measured),
        "max_input_tokens": args.max_input_tokens,
        "hf_train_split_file": str(train_split_path.relative_to(PROJECT_ROOT)) if args.write_hf_splits else None,
        "hf_test_split_file": str(test_split_path.relative_to(PROJECT_ROOT)) if args.write_hf_splits else None,
        "tokenizer": tokenizer_name,
        "system_prompt": args.system_prompt,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"loaded valid rows: {len(source_rows)}")
    print(f"train/test: {split_idx}/{len(test_rows)}")
    if args.max_input_tokens is not None:
        print(f"candidates after max_input_tokens<={args.max_input_tokens}: {len(measured)}")
    print(f"selected: {len(selected)}")
    if selected:
        tokens = [row.input_tokens for row in selected]
        print(f"input tokens min/p50/max: {min(tokens)} / {tokens[len(tokens)//2]} / {max(tokens)}")
    print(f"jsonl: {jsonl_path}")
    if args.xlsx:
        print(f"xlsx:  {xlsx_path}")
    print(f"stems: {stems_path}")
    if args.write_hf_splits:
        print(f"hf train split: {train_split_path}")
        print(f"hf test split:  {test_split_path}")
    print(f"meta:  {metadata_path}")


if __name__ == "__main__":
    main()
