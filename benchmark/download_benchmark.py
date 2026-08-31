"""
Download the text-to-json benchmark split from Hugging Face.

This is the public/reproducibility entrypoint: it does not rebuild the split.
It simply downloads the already-published benchmark data and saves it locally
as JSONL for benchmark/inference.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "boradorish/text-to-json-benchmark"
DISPLAY_TRUNCATE = 32000


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_row(row: dict) -> dict:
    return {
        "stem": row.get("stem") or row.get("id"),
        "user_prompt": row.get("user_prompt") or "",
        "gold_json": normalize_json_text(row.get("gold_json") if "gold_json" in row else row.get("json")),
        "json_schema": normalize_json_text(row.get("json_schema")),
        "input_tokens": row.get("input_tokens"),
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_xlsx(records: list[dict], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install pandas and openpyxl for --xlsx: pip install pandas openpyxl") from exc

    display_records = []
    for record in records:
        row = dict(record)
        for col in ("user_prompt", "gold_json", "json_schema"):
            if isinstance(row.get(col), str) and len(row[col]) > DISPLAY_TRUNCATE:
                row[col] = row[col][:DISPLAY_TRUNCATE] + "..."
        display_records.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(display_records).to_excel(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the HF benchmark split as local JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="benchmark/data/test.jsonl")
    parser.add_argument("--xlsx", default=None, help="Optional Excel preview path.")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install datasets first: pip install datasets") from exc

    token = args.token or os.environ.get("HF_TOKEN") or None
    dataset = load_dataset(args.dataset, split=args.split, token=token)
    records = [normalize_row(row) for row in dataset]

    output_path = resolve_path(args.output)
    write_jsonl(records, output_path)

    if args.xlsx:
        write_xlsx(records, resolve_path(args.xlsx))

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "rows": len(records),
        "output": str(output_path.relative_to(PROJECT_ROOT)) if output_path.is_relative_to(PROJECT_ROOT) else str(output_path),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"dataset: {args.dataset}")
    print(f"split:   {args.split}")
    print(f"rows:    {len(records)}")
    print(f"jsonl:   {output_path}")
    print(f"meta:    {metadata_path}")


if __name__ == "__main__":
    main()
