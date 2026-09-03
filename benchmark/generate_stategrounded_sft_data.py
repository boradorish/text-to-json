"""Generate held-out synthetic, source-grounded dialogue-state examples.

The generator deliberately never reads SGD.  It teaches the STAGE protocol on
invented services: a single USER utterance, a schema, literal value copying,
and an explicit ``no output`` value for every unmentioned slot.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")

SLOT_BANK = [
    ("city", "city named by the user", ["Brighton", "Dover", "Fairview", "Kingston", "Lakeside"]),
    ("date", "date requested by the user", ["April 8", "next Tuesday", "July 14", "the 3rd of May"]),
    ("time", "time requested by the user", ["7:15 pm", "half past nine", "6 am", "noon"]),
    ("party_size", "number of people", ["1", "2", "3", "4", "6"]),
    ("item", "item or service requested", ["garden tour", "window seat", "morning delivery", "museum pass"]),
    ("price_range", "requested budget", ["under 20 dollars", "around 45 dollars", "cheap", "premium"]),
    ("provider", "provider name explicitly requested", ["Northstar", "Maple House", "Riverline", "Cobalt"]),
    ("preference", "user preference", ["quiet", "outdoor", "vegetarian", "nonstop"]),
    ("confirmation_code", "confirmation code explicitly stated", ["AX42", "QK19", "MB77", "ZR05"]),
]
TEMPLATES = {
    "city": "in {value}", "date": "for {value}", "time": "at {value}",
    "party_size": "for {value} people", "item": "I need {value}",
    "price_range": "with a budget of {value}", "provider": "from {value}",
    "preference": "and it should be {value}", "confirmation_code": "my code is {value}",
}


def make_row(rng: random.Random, index: int) -> dict:
    # Random slot order prevents a positional all-fields heuristic.
    slots = rng.sample(SLOT_BANK, rng.randint(3, 7))
    # Every example contains an unmentioned field, the coverage-bias behaviour
    # this continuation set is intended to correct.
    active_count = rng.randint(1, min(3, len(slots) - 1))
    active = set(rng.sample([name for name, _, _ in slots], active_count))
    values = {name: rng.choice(candidates) for name, _, candidates in slots}
    properties = {}
    gold = {}
    phrases = []
    for name, description, candidates in slots:
        properties[name] = {
            "type": "string",
            "description": f"{description}. Use 'no output' when the user has not specified it.",
        }
        gold[name] = values[name] if name in active else "no output"
        if name in active:
            phrases.append(TEMPLATES[name].format(value=values[name]))
    rng.shuffle(phrases)
    utterance = "Hello, " + ", ".join(phrases) + "."
    schema = {
        "type": "object",
        "description": "Fill every slot. Copy only values explicitly stated by the USER. Write 'no output' for every unmentioned slot.",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    user = (
        "Extract the dialogue state from the target USER utterance only. "
        "Do not infer or copy values that are not explicitly stated by that user.\n\n"
        f"=== Report ===\nUSER: {utterance}\n\n=== JSON Schema ===\n{json.dumps(schema)}"
    )
    return {"id": f"synthetic_state_{index:05d}", "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(gold)},
    ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/sft/state_grounded_synthetic_2k.jsonl")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.size):
            handle.write(json.dumps(make_row(rng, index), ensure_ascii=False) + "\n")
    print(f"wrote={args.size} output={args.output}")


if __name__ == "__main__":
    main()
