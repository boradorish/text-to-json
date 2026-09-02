"""Create a small, held-out synthetic dataset for tool-selection SFT.

The examples use the same JSON-schema tool-call protocol as the BFCL STAGE
prompt, but their tools, names and requests are synthetic.  This prevents
BFCL prompt/test examples from leaking into the continued-SFT data.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.prompt_loader import find_project_root

SYSTEM_PROMPT = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")

TOOL_SPECS = [
    ("lookup_weather", "Look up weather for a city.", [("city", "string"), ("unit", "string")]),
    ("send_note", "Send a short note to a recipient.", [("recipient", "string"), ("message", "string")]),
    ("create_reminder", "Create a reminder.", [("title", "string"), ("time", "string")]),
    ("translate_text", "Translate text into a language.", [("text", "string"), ("language", "string")]),
    ("find_cafe", "Find a cafe in a city.", [("city", "string"), ("style", "string")]),
    ("convert_currency", "Convert an amount between currencies.", [("amount", "number"), ("from_currency", "string"), ("to_currency", "string")]),
    ("add_todo", "Add an item to a to-do list.", [("task", "string"), ("priority", "string")]),
    ("check_stock", "Look up a company stock quote.", [("symbol", "string")]),
]

VALUES = {
    "city": ["Seoul", "Busan", "Lisbon", "Kyoto", "Toronto", "Nairobi"],
    "unit": ["Celsius", "Fahrenheit"],
    "recipient": ["Mina", "Joon", "Alex", "Priya", "Sam"],
    "message": ["the meeting moved to Friday", "I will arrive at noon", "please review the draft"],
    "title": ["submit expense report", "call the dentist", "prepare slides"],
    "time": ["tomorrow at 9 AM", "Friday at 3 PM", "next Monday"],
    "text": ["good morning", "where is the station", "thank you"],
    "language": ["Korean", "Japanese", "Spanish", "French"],
    "style": ["quiet", "Italian", "vegan", "outdoor"],
    "amount": [25, 60, 120, 300],
    "from_currency": ["USD", "EUR", "KRW", "JPY"],
    "to_currency": ["USD", "EUR", "KRW", "JPY"],
    "task": ["buy milk", "book a haircut", "read the proposal", "call Mom"],
    "priority": ["low", "medium", "high"],
    "symbol": ["AAPL", "TSLA", "MSFT", "NVDA"],
}


def function(spec: tuple[str, str, list[tuple[str, str]]]) -> dict[str, Any]:
    name, description, fields = spec
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in fields},
            "required": [key for key, _ in fields],
        },
    }


def call_for(spec: tuple[str, str, list[tuple[str, str]]], rng: random.Random) -> dict[str, Any]:
    name, _, fields = spec
    args: dict[str, Any] = {}
    for key, _ in fields:
        value = rng.choice(VALUES[key])
        if key == "to_currency":
            choices = [v for v in VALUES[key] if v != args.get("from_currency")]
            value = rng.choice(choices)
        args[key] = value
    return {"name": name, "arguments": args}


def request_for(calls: list[dict[str, Any]]) -> str:
    pieces = []
    for call in calls:
        a = call["arguments"]
        name = call["name"]
        if name == "lookup_weather": pieces.append(f"check the weather in {a['city']} in {a['unit']}")
        elif name == "send_note": pieces.append(f"tell {a['recipient']}: {a['message']}")
        elif name == "create_reminder": pieces.append(f"remind me to {a['title']} {a['time']}")
        elif name == "translate_text": pieces.append(f"translate '{a['text']}' into {a['language']}")
        elif name == "find_cafe": pieces.append(f"find a {a['style']} cafe in {a['city']}")
        elif name == "convert_currency": pieces.append(f"convert {a['amount']} {a['from_currency']} to {a['to_currency']}")
        elif name == "add_todo": pieces.append(f"add '{a['task']}' as a {a['priority']} priority task")
        elif name == "check_stock": pieces.append(f"get the quote for {a['symbol']}")
    return "Please " + " and ".join(pieces) + "."


def call_schema(functions: list[dict[str, Any]]) -> dict[str, Any]:
    variants = []
    for fn in functions:
        parameters = dict(fn["parameters"])
        parameters["additionalProperties"] = False
        variants.append({"type": "object", "additionalProperties": False, "required": ["name", "arguments"], "properties": {"name": {"const": fn["name"]}, "arguments": parameters}})
    return {"type": "object", "additionalProperties": False, "required": ["calls"], "properties": {"calls": {"type": "array", "minItems": 1, "items": {"oneOf": variants}}}}


def make_example(rng: random.Random) -> dict[str, Any]:
    # Include 1--3 needed calls and 1--3 distractors.  Repeated calls are
    # intentionally allowed, matching BFCL multiple/parallel behaviour.
    selected_specs = [rng.choice(TOOL_SPECS) for _ in range(rng.choice([1, 1, 2, 2, 3]))]
    calls = [call_for(spec, rng) for spec in selected_specs]
    candidate_specs = list({spec[0]: spec for spec in selected_specs}.values())
    distractors = [s for s in TOOL_SPECS if s[0] not in {x[0] for x in candidate_specs}]
    rng.shuffle(distractors)
    candidate_specs.extend(distractors[: rng.choice([1, 2, 3])])
    rng.shuffle(candidate_specs)
    functions = [function(spec) for spec in candidate_specs]
    schema = call_schema(functions)
    user = "\n\n".join([
        "Extract the tool calls needed to fulfil the request below as JSON that conforms to the schema.",
        f"=== Report ===\n{request_for(calls)}",
        f"=== Available functions ===\n{json.dumps(functions, ensure_ascii=False)}",
        f"=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False)}",
    ])
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}, {"role": "assistant", "content": json.dumps({"calls": calls}, ensure_ascii=False)}]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "toolfew_sft" / "data")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    write_jsonl(args.output_dir / "train.jsonl", [make_example(rng) for _ in range(args.train_size)])
    write_jsonl(args.output_dir / "validation.jsonl", [make_example(rng) for _ in range(args.validation_size)])
    (args.output_dir / "metadata.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.train_size} train and {args.validation_size} held-out synthetic examples to {args.output_dir}")


if __name__ == "__main__":
    main()
