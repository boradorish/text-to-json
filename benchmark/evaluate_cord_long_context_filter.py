"""Evaluate a label-free CORD long-context slice chosen on validation data.

The threshold is the requested quantile of validation ``user_prompt`` length.
It never reads gold labels when choosing the slice.  Base and STAGE result
files must have the same ordered CORD rows and decoding configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate_row


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def metrics(rows: list[dict]) -> dict[str, float]:
    values = [evaluate_row(row) for row in rows]
    count = len(values)
    return {
        "n": count,
        "VA": 100 * sum(item["value_match"] for item in values) / count,
        "EMR": 100 * sum(item["exact_match"] for item in values) / count,
        "PFR": 100 * sum(not item["no_output"] for item in values) / count,
        "SCR": 100 * sum(item["schema_valid"] for item in values) / count,
        "NR": 100 * sum(item["noise_ratio"] for item in values) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-reference", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.75)
    args = parser.parse_args()
    if not 0 < args.quantile < 1:
        raise SystemExit("--quantile must be between zero and one")

    reference, base, stage = read(args.validation_reference), read(args.base), read(args.stage)
    if len(base) != len(stage) or [r["stem"] for r in base] != [r["stem"] for r in stage]:
        raise SystemExit("base and STAGE rows must have matching ordered stems")
    lengths = sorted(len(row["user_prompt"]) for row in reference)
    threshold = lengths[int(len(lengths) * args.quantile)]
    selected = [index for index, row in enumerate(base) if len(row["user_prompt"]) > threshold]
    if not selected:
        raise SystemExit("filter selected no rows")
    base_metrics, stage_metrics = metrics([base[i] for i in selected]), metrics([stage[i] for i in selected])
    print(json.dumps({
        "rule": f"len(user_prompt) > validation_p{args.quantile:g} ({threshold})",
        "threshold": threshold,
        "base": base_metrics,
        "stage": stage_metrics,
        "VA_delta_stage_minus_base": stage_metrics["VA"] - base_metrics["VA"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
