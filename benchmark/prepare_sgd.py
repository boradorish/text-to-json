"""Prepare fixed, schema-guided SGD dialogue-state-tracking examples."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SGD = Path("/mnt/nvme/cache/interns/sgd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("standard", "explicit"), required=True)
    parser.add_argument("--split", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--context",
        choices=("history", "latest-user"),
        default="history",
        help="Render the whole dialogue history or only the target user turn.",
    )
    parser.add_argument(
        "--filter",
        choices=("none", "latest-user-grounded"),
        default="none",
        help="Keep only examples whose non-empty gold values occur in the target user utterance.",
    )
    parser.add_argument(
        "--select-eligible-first",
        action="store_true",
        help="Apply --filter before balanced sampling (use a distinct --ids-output).",
    )
    parser.add_argument("--sgd-root", type=Path, default=DEFAULT_SGD)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ids-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_text(value: str) -> str:
    """Conservative lexical grounding check used before inference, never predictions."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def is_latest_user_grounded(item: dict) -> bool:
    """Whether every non-empty gold value is explicitly present in this user turn.

    This defines a source-grounded *single-turn* DST subset independently of
    model outputs.  It intentionally excludes carried-over and system-proposed
    values, which are a different state-update task from STAGE's report-grounded
    extraction objective.
    """
    source = normalized_text(item["latest_user"])
    values = [normalized_text(str(value)) for value in item["gold_slots"].values()]
    values = [value for value in values if value]
    return bool(values) and all(value in source for value in values)


def service_schema(service: dict, explicit: bool) -> dict:
    props, required = {}, []
    for slot in service["slots"]:
        spec = {"description": slot["description"]}
        if slot["is_categorical"]:
            spec["type"] = "string"
            spec["enum"] = list(slot["possible_values"]) + (["no output"] if explicit else [])
        else:
            spec["type"] = "string"
            if explicit:
                spec["description"] += ' Use "no output" if the user has not specified this slot.'
        props[slot["name"]] = spec
        if explicit:
            required.append(slot["name"])
    description = (
        "Fill every slot. Write \"no output\" for any slot the user has not specified so far."
        if explicit else "Include only slots the user has specified so far."
    )
    schema = {"type": "object", "description": description, "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def collect_examples(sgd_root: Path, schemas: dict[str, dict], seen_services: set[str]) -> list[dict]:
    examples = []
    for path in sorted((sgd_root / "test").glob("dialogues_*.json")):
        for dialogue in load_json(path):
            history = []
            for turn_index, turn in enumerate(dialogue["turns"]):
                history.append(f'{turn["speaker"]}: {turn["utterance"]}')
                if turn["speaker"] != "USER":
                    continue
                for frame in turn.get("frames", []):
                    service = frame["service"]
                    if service not in schemas or "state" not in frame:
                        continue
                    values = {name: vals[0] for name, vals in frame["state"].get("slot_values", {}).items() if vals}
                    examples.append({
                        "id": f'{dialogue["dialogue_id"]}:{turn_index}:{service}',
                        "dialogue_id": dialogue["dialogue_id"], "turn_index": turn_index,
                        "service": service, "seen_service": service in seen_services,
                        "history": "\n".join(history), "latest_user": turn["utterance"],
                        "gold_slots": values,
                    })
    return examples


def balanced_ids(examples: list[dict], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    grouped = {seen: defaultdict(list) for seen in (True, False)}
    for item in examples:
        grouped[item["seen_service"]][item["service"]].append(item)
    targets = {True: count // 2, False: count - count // 2}
    selected = []
    for seen, target in targets.items():
        queues = []
        for service in sorted(grouped[seen]):
            rows = grouped[seen][service]
            rng.shuffle(rows)
            queues.append(deque(rows))
        while len([x for x in selected if x["seen_service"] == seen]) < target:
            progressed = False
            for queue in queues:
                if queue and len([x for x in selected if x["seen_service"] == seen]) < target:
                    selected.append(queue.popleft())
                    progressed = True
            if not progressed:
                raise ValueError(f"Not enough {'seen' if seen else 'unseen'} SGD examples")
    rng.shuffle(selected)
    return [item["id"] for item in selected]


def main() -> None:
    args = parse_args()
    count = 100 if args.split == "pilot" else 2000
    output = args.output or ROOT / "benchmark" / "data" / f"sgd_{args.split}_{args.format}.jsonl"
    ids_output = args.ids_output or ROOT / "benchmark" / "data" / f"sgd_{args.split}_ids.json"
    test_schemas = {x["service_name"]: x for x in load_json(args.sgd_root / "test" / "schema.json")}
    train_services = {x["service_name"] for x in load_json(args.sgd_root / "train" / "schema.json")}
    examples = collect_examples(args.sgd_root, test_schemas, train_services)
    eligible = examples
    if args.select_eligible_first and args.filter == "latest-user-grounded":
        eligible = [item for item in examples if is_latest_user_grounded(item)]
    if ids_output.exists():
        chosen_ids = load_json(ids_output)
    else:
        chosen_ids = balanced_ids(eligible, count, args.seed)
        ids_output.parent.mkdir(parents=True, exist_ok=True)
        ids_output.write_text(json.dumps(chosen_ids, indent=2), encoding="utf-8")
    by_id = {item["id"]: item for item in examples}
    chosen = [by_id[item_id] for item_id in chosen_ids]
    if args.filter == "latest-user-grounded":
        chosen = [item for item in chosen if is_latest_user_grounded(item)]
    explicit = args.format == "explicit"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in chosen:
            schema = service_schema(test_schemas[item["service"]], explicit)
            gold = dict(item["gold_slots"])
            if explicit:
                gold = {slot: gold.get(slot, "no output") for slot in schema["properties"]}
            if args.context == "latest-user":
                report = f"USER: {item['latest_user']}"
                instruction = (
                    "Extract the dialogue state from the target USER utterance only. "
                    "Do not infer or copy values that are not explicitly stated by that user."
                )
            else:
                report = item["history"]
                instruction = "Extract the current dialogue state according to the JSON Schema."
            prompt = f"{instruction}\n\n=== Report ===\n{report}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False)}"
            row = {**item, "user_prompt": prompt, "json_schema": json.dumps(schema, ensure_ascii=False), "gold_json": json.dumps(gold, ensure_ascii=False)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote={len(chosen)} seen={sum(x['seen_service'] for x in chosen)} context={args.context} filter={args.filter} output={output}")


if __name__ == "__main__":
    main()
