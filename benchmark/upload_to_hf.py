"""
Upload the prepared benchmark train/test split to Hugging Face Datasets.

Run prepare_benchmark.py first so benchmark/hf_splits/*.jsonl exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload benchmark train/test splits to HF Datasets.")
    parser.add_argument("--repo-id", required=True, help="Example: boradorish/text-to-json-benchmark")
    parser.add_argument("--train-file", default="benchmark/hf_splits/train.jsonl")
    parser.add_argument("--test-file", default="benchmark/hf_splits/test.jsonl")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN env var")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--commit-message", default="Upload text-to-json benchmark splits")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = resolve_path(args.train_file)
    test_path = resolve_path(args.test_file)

    train_records = load_jsonl(train_path)
    test_records = load_jsonl(test_path)
    print(f"train: {len(train_records):,} ({train_path})")
    print(f"test:  {len(test_records):,} ({test_path})")

    if args.dry_run:
        print("[dry-run] upload skipped")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN is not set. Export HF_TOKEN or pass --token.", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install datasets first: pip install datasets") from exc

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "test": Dataset.from_list(test_records),
        }
    )
    push_kwargs = {
        "repo_id": args.repo_id,
        "token": token,
        "private": args.private,
        "commit_message": args.commit_message,
    }
    if args.revision:
        push_kwargs["revision"] = args.revision

    print(f"uploading to https://huggingface.co/datasets/{args.repo_id}")
    dataset_dict.push_to_hub(**push_kwargs)
    print("done")


if __name__ == "__main__":
    main()
