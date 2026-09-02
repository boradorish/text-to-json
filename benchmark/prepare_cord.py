"""Prepare CORD-v2 receipts as STAGE-compatible text-to-JSON benchmark rows.

CORD ground truth contains OCR annotations (`valid_line`) inside the serialized
`ground_truth` field.  This exporter uses those annotations rather than the
receipt image, so evaluation measures source-grounded extraction rather than
OCR quality.  Top-level ``menu`` objects are normalized to a list because CORD
uses an object for one-item receipts and a list otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PREFIX = """Extract the receipt fields from the OCR text according to the JSON Schema.

=== Report ===
"""


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_gold(gt_parse: dict[str, Any]) -> dict[str, Any]:
    """Make CORD's singleton-vs-list menu convention uniform."""
    result = dict(gt_parse)
    if isinstance(result.get("menu"), dict):
        result["menu"] = [result["menu"]]
    return result


def schema_for(value: Any) -> dict[str, Any]:
    """Derive a validating, example-specific schema from a CORD gold value."""
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: schema_for(item) for key, item in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item_schemas: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in value:
            item_schema = schema_for(item)
            marker = json.dumps(item_schema, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                item_schemas.append(item_schema)
        if not item_schemas:
            return {"type": "array", "items": {}}
        items: dict[str, Any]
        if len(item_schemas) == 1:
            items = item_schemas[0]
        else:
            items = {"anyOf": item_schemas}
        return {"type": "array", "items": items}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "string"}


def line_position(line: dict[str, Any]) -> tuple[int, int]:
    words = line.get("words", [])
    if not words:
        return (10**9, 10**9)
    quads = [word.get("quad", {}) for word in words]
    return (
        min(quad.get("y1", quad.get("y2", 10**9)) for quad in quads),
        min(quad.get("x1", quad.get("x4", 10**9)) for quad in quads),
    )


def ocr_report(ground_truth: dict[str, Any]) -> str:
    """Render annotated OCR lines in visual reading order."""
    rendered: list[tuple[tuple[int, int], str]] = []
    for line in ground_truth.get("valid_line", []):
        words = line.get("words", [])
        ordered_words = sorted(
            words,
            key=lambda word: word.get("quad", {}).get("x1", word.get("quad", {}).get("x4", 10**9)),
        )
        text = " ".join(str(word.get("text", "")).strip() for word in ordered_words).strip()
        if text:
            rendered.append((line_position(line), text))
    return "\n".join(text for _, text in sorted(rendered, key=lambda pair: pair[0]))


def make_record(index: int, row: dict[str, Any]) -> dict[str, Any]:
    annotation = json.loads(row["ground_truth"])
    gold = normalize_gold(annotation["gt_parse"])
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema_for(gold)}
    report = ocr_report(annotation)
    return {
        "stem": f"cord_test_{index:03d}",
        "user_prompt": f"{PROMPT_PREFIX}{report}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}",
        "gold_json": json.dumps(gold, ensure_ascii=False),
        "json_schema": json.dumps(schema, ensure_ascii=False),
        "source_split": "CORD-v2/test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CORD-v2 test rows to STAGE benchmark JSONL.")
    parser.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="benchmark/data/cord_v2_test_100.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    records = [make_record(i, dataset[i]) for i in range(min(args.limit, len(dataset)))]
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"exported {len(records)} CORD rows: {output}")


if __name__ == "__main__":
    main()
