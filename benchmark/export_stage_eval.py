"""Export the downloaded STAGE-Eval Parquet split to benchmark JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ("stem", "user_prompt", "gold_json", "json_schema", "input_tokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/STAGE-eval/data/test-00000-of-00001.parquet")
    parser.add_argument("--output", default="benchmark/data/stage_eval_test.jsonl")
    args = parser.parse_args()
    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    frame = pd.read_parquet(input_path, columns=list(REQUIRED_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(frame)} rows: {output_path}")


if __name__ == "__main__":
    main()
