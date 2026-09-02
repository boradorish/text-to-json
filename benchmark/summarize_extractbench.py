"""Summarize ExtractBench JSONL results overall and by source length bucket.

The prepared ExtractBench filenames retain their original ``short__`` and
``medium__`` prefixes.  This script deliberately uses that source annotation
rather than re-tokenizing a model-specific prompt, so every compared model is
scored in identical buckets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate import evaluate_row, resolve_path


METRICS = [
    "no_output",
    "exact_match",
    "schema_valid",
    "noise_ratio",
    "value_match",
]


def bucket(stem: str) -> str:
    """Return the dataset's documented length label, or ``other``."""
    prefix = stem.split("__", 1)[0]
    return prefix if prefix in {"short", "medium", "long"} else "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = (
        resolve_path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_summary.csv")
    )
    rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("skip_reason"):
                continue
            rows.append({"length_bucket": bucket(record.get("stem", "")), **evaluate_row(record)})
    if not rows:
        raise SystemExit("No evaluable records.")

    frame = pd.DataFrame(rows)
    groups = [("overall", frame)] + [(name, group) for name, group in frame.groupby("length_bucket", sort=False)]
    summary = []
    for name, group in groups:
        summary.append(
            {
                "length_bucket": name,
                "samples": len(group),
                "no_output_ratio": group.no_output.mean(),
                "exact_match_ratio": group.exact_match.mean(),
                "schema_valid_ratio": group.schema_valid.mean(),
                "mean_noise_ratio": group.noise_ratio.mean(),
                "mean_value_match": group.value_match.mean(),
            }
        )
    result = pd.DataFrame(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
