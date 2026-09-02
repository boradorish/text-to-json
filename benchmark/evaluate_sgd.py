"""Evaluate prepared SGD predictions with the official SGD goal metric code."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/mnt/nvme/cache/interns/schema-guided-dst-metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", choices=("standard", "explicit"), required=True)
    parser.add_argument("--sgd-root", type=Path, default=Path("/mnt/nvme/cache/interns/sgd"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def normalize(values: dict, explicit: bool) -> dict:
    return {key: value for key, value in values.items() if not (explicit and isinstance(value, str) and value.strip().lower() == "no output")}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from schema_guided_dst import metrics  # official Google Research implementation

    input_path = Path(args.input)
    rows = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    services = {item["service_name"]: item for item in json.loads((args.sgd_root / "test" / "schema.json").read_text())}
    totals = defaultdict(list)
    explicit = args.format == "explicit"
    for row in rows:
        gold = normalize(json.loads(row["gold_json"]), explicit)
        raw_prediction = row.get("pred_json")
        try:
            if not raw_prediction:
                raise ValueError("empty prediction")
            pred = normalize(json.loads(raw_prediction), explicit)
            parsed = True
        except Exception:
            pred, parsed = {}, False
        frame_ref = {"state": {"slot_values": {key: [value] for key, value in gold.items()}}}
        frame_hyp = {"state": {"slot_values": {key: [value] for key, value in pred.items()}}}
        score = metrics.get_average_and_joint_goal_accuracy(frame_ref, frame_hyp, services[row["service"]], True)
        inactive = [key for key in pred if key not in gold]
        missing = [key for key in gold if key not in pred]
        group_names = ["all", "seen" if row["seen_service"] else "unseen"]
        average_goal = score[metrics.AVERAGE_GOAL_ACCURACY]
        for group in group_names:
            totals[group].append({
                "joint_goal_accuracy": float(score[metrics.JOINT_GOAL_ACCURACY]),
                "slot_accuracy": None if average_goal == metrics.NAN_VAL else float(average_goal),
                "hallucinated_slot_rate": len(inactive) / max(1, len(pred)),
                "missing_slot_rate": len(missing) / max(1, len(gold)),
                "parse_failure": float(not parsed),
            })
    aggregate = {}
    for group, values in totals.items():
        summary = {}
        for metric in values[0]:
            numeric = [item[metric] for item in values if item[metric] is not None]
            summary[metric] = sum(numeric) / len(numeric) if numeric else None
        aggregate[group] = summary | {"samples": len(values)}
    output = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_eval.json")
    output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
