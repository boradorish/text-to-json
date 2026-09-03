"""Package verified source-grounded dialogue-state examples for SFT/HF.

The input is produced by ``stage_dialog/generate_dialogs.py``.  Its validation
has already discarded any dialogue where a required user value is absent or an
unmentioned spreadsheet value leaks into the conversation.  This script only
converts its benchmark-row shape into the shared chat ``messages`` format.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/stage_dialog/full/stage_dialog_examples.jsonl")
    parser.add_argument("--validation-stats", type=Path, default=ROOT / "outputs/stage_dialog/full/stats.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation_stats.read_text(encoding="utf-8"))
    if validation.get("valid", 0) < 1:
        raise SystemExit("Expected a non-empty validated dialogue set.")
    formats: Counter[str] = Counter()
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            gold = json.loads(row["gold_json"])
            fmt = row["format"]
            formats[fmt] += 1
            target.write(json.dumps({
                "source": "source_grounded_dialogue_state",
                "format": fmt,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["user_prompt"]},
                    {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, indent=2)},
                ],
            }, ensure_ascii=False) + "\n")
            count += 1
    metadata = {
        "examples": count, "formats": dict(formats), "validated_dialogues": validation["valid"],
        "generation_jobs": validation["jobs"], "source_rule": "Spreadsheet-row values are compared against generated dialogue; required values must occur verbatim in USER turns and unmentioned values must be absent before examples are retained.",
        "task_rule": "Each example predicts only values mentioned by the cut point, either as an open schema or as all fields with no output for unmentioned fields.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
