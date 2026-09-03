"""Make a deterministic CORD-specialization/replay mix for fair init ablations.

Both Qwen3 base and STAGE-SFT receive byte-identical rows.  CORD train rows
are repeated only to set their sampling weight; the held-out validation/test
splits are never read here.  Replay rows are pre-existing STAGE-format
report-to-schema examples, not CORD data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cord", type=Path, default=ROOT / "benchmark/data/cord_v2_adaptation/train_50.jsonl")
    parser.add_argument("--stage-mix", type=Path, default=ROOT / "data/sft/stage_dialog_mix.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cord-repeat", type=int, default=4)
    parser.add_argument("--stage-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cord_repeat < 1 or args.stage_count < 1:
        raise SystemExit("--cord-repeat and --stage-count must be positive")

    cord = read_jsonl(args.cord)
    stage = [row for row in read_jsonl(args.stage_mix) if row.get("source") == "stage"]
    if len(stage) < args.stage_count:
        raise SystemExit(f"need {args.stage_count} STAGE replay rows; found {len(stage)}")
    rng = random.Random(args.seed)
    rng.shuffle(stage)
    rows = []
    for repeat in range(args.cord_repeat):
        for index, row in enumerate(cord):
            rows.append({"source": "cord", "repeat": repeat, "index": index, "messages": row["messages"]})
    rows.extend({"source": "stage_replay", "messages": row["messages"]} for row in stage[: args.stage_count])
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(rows), "cord_rows": len(cord), "cord_repeat": args.cord_repeat,
                      "cord_examples": len(cord) * args.cord_repeat, "stage_replay": args.stage_count,
                      "seed": args.seed, "output": str(args.output)}))


if __name__ == "__main__":
    main()
