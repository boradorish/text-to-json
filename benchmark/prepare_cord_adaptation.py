"""Prepare deterministic CORD adaptation splits in the inference prompt format."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prepare_cord import make_example, make_record, normalize_gold, ocr_report


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")


def pick_one_shot(train, layout: str, descriptions: bool, tolerance: float):
    for candidate in train:
        annotation = json.loads(candidate["ground_truth"])
        menu = normalize_gold(annotation["gt_parse"]).get("menu", [])
        rendered = ocr_report(annotation, layout, tolerance)
        if isinstance(menu, list) and len(menu) >= 3 and ("@" in rendered or "X" in rendered):
            return make_example(candidate, layout, descriptions, tolerance)
    raise RuntimeError("No eligible CORD train one-shot example")


def write_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmark/data/cord_v2_adaptation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from datasets import load_dataset

    train = load_dataset("naver-clova-ix/cord-v2", split="train")
    validation = load_dataset("naver-clova-ix/cord-v2", split="validation")
    test = load_dataset("naver-clova-ix/cord-v2", split="test")
    layout, descriptions, tolerance = "rows", True, 0.5
    example = pick_one_shot(train, layout, descriptions, tolerance)
    indices = list(range(len(train)))
    random.Random(args.seed).shuffle(indices)
    output = ROOT / args.output_dir

    for size in (50, 200, 800):
        benchmark_rows = [make_record(i, train[index], layout, descriptions, example, tolerance) for i, index in enumerate(indices[:size])]
        messages = [
            {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["user_prompt"]},
                {"role": "assistant", "content": row["gold_json"]},
            ]}
            for row in benchmark_rows
        ]
        write_records(output / f"train_{size}.jsonl", messages)

    for split_name, split in (("validation", validation), ("test", test)):
        records = [make_record(i, row, layout, descriptions, example, tolerance) for i, row in enumerate(split)]
        for row in records:
            row["stem"] = f"cord_{split_name}_{int(row['stem'].rsplit('_', 1)[1]):03d}"
            row["source_split"] = f"CORD-v2/{split_name}"
        write_records(output / f"{split_name}_100.jsonl", records)
    (output / "metadata.json").write_text(json.dumps({"seed": args.seed, "renderer": "A+B+C (rows, descriptions, fixed train one-shot)"}, indent=2) + "\n", encoding="utf-8")
    print(f"prepared CORD adaptation data in {output}")


if __name__ == "__main__":
    main()
