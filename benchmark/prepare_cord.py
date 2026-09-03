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
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PREFIX = """Extract the receipt fields from the OCR text according to the JSON Schema.

=== Report ===
"""

PRINTED_VALUE_INSTRUCTION = (
    "Return each value exactly as printed on the receipt. Preserve currency symbols, "
    "@, X, punctuation, and spacing conventions when they are part of a value."
)

# Definitions follow the CORD-v2 field conventions.  A key can occur in more
# than one nested object, so these are intentionally keyed by property name.
FIELD_DESCRIPTIONS = {
    "nm": "Menu item name as printed on the receipt.",
    "cnt": "Quantity of this menu item.",
    "unitprice": "Unit price of this menu item.",
    "price": "Line total price of this menu item.",
    "num": "Printed item code or menu number.",
    "itemsubtotal": "Subtotal for the listed menu items.",
    "subtotal_price": "Subtotal price before final payment.",
    "discount_price": "Discount amount.",
    "tax_price": "Tax amount.",
    "service_price": "Service charge amount.",
    "etc": "Other charge or adjustment as printed.",
    "total_price": "Final total amount due or paid.",
    "cashprice": "Amount paid in cash.",
    "changeprice": "Change returned to the customer.",
    "creditcardprice": "Amount paid by credit card.",
    "emoneyprice": "Amount paid using electronic money.",
    "menuqty_cnt": "Number of purchased menu items.",
    "menutype_cnt": "Number of distinct menu item types.",
}


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


def ocr_report_lines(ground_truth: dict[str, Any]) -> str:
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


def word_geometry(word: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Return x-center, y-center, left, right, and height for a CORD word."""
    quad = word.get("quad", {})
    xs = [float(quad.get(f"x{i}", 0)) for i in range(1, 5)]
    ys = [float(quad.get(f"y{i}", 0)) for i in range(1, 5)]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    height = max(bottom - top, 1.0)
    return ((left + right) / 2, (top + bottom) / 2, left, right, height)


def ocr_report_rows(ground_truth: dict[str, Any], row_tolerance: float = 0.5) -> str:
    """Render all OCR words as visual rows, preserving wide table-like gaps."""
    words = [word for line in ground_truth.get("valid_line", []) for word in line.get("words", [])]
    words = [word for word in words if str(word.get("text", "")).strip()]
    if not words:
        return ""
    typical_height = median(word_geometry(word)[4] for word in words)
    y_tolerance = max(4.0, row_tolerance * typical_height)
    rows: list[list[dict[str, Any]]] = []
    row_centers: list[float] = []
    for word in sorted(words, key=lambda item: (word_geometry(item)[1], word_geometry(item)[0])):
        y_center = word_geometry(word)[1]
        if rows and abs(y_center - row_centers[-1]) <= y_tolerance:
            rows[-1].append(word)
            row_centers[-1] = median(word_geometry(item)[1] for item in rows[-1])
        else:
            rows.append([word])
            row_centers.append(y_center)

    rendered_rows: list[str] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: word_geometry(item)[0])
        cells: list[list[str]] = [[]]
        previous: dict[str, Any] | None = None
        for word in ordered:
            if previous is not None:
                _, _, _, previous_right, previous_height = word_geometry(previous)
                _, _, current_left, _, current_height = word_geometry(word)
                if current_left - previous_right > 1.5 * max(previous_height, current_height):
                    cells.append([])
            cells[-1].append(str(word.get("text", "")).strip())
            previous = word
        rendered_rows.append(" | ".join(" ".join(cell) for cell in cells if cell))
    report = "\n".join(row for row in rendered_rows if row)
    # A few CORD quadrilaterals are skewed enough that words from one annotated
    # field land on adjacent visual rows.  Retain only those original fragments
    # that the row view split, so source strings are never lost while the main
    # representation remains a coordinate-based table.
    fragments = [line for line in ocr_report_lines(ground_truth).splitlines() if line and line not in report]
    if fragments:
        report += "\n\n=== OCR fragments (verbatim) ===\n" + "\n".join(fragments)
    return report


def ocr_report(ground_truth: dict[str, Any], layout: str = "lines", row_tolerance: float = 0.5) -> str:
    if layout == "lines":
        return ocr_report_lines(ground_truth)
    if layout == "rows":
        return ocr_report_rows(ground_truth, row_tolerance)
    raise ValueError(f"unknown layout: {layout}")


def add_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Add CORD field definitions without changing validation constraints."""
    enriched = json.loads(json.dumps(schema))
    enriched["description"] = PRINTED_VALUE_INSTRUCTION

    def visit(node: dict[str, Any]) -> None:
        for key, child in node.get("properties", {}).items():
            if key in FIELD_DESCRIPTIONS:
                child["description"] = FIELD_DESCRIPTIONS[key]
            visit(child)
        if isinstance(node.get("items"), dict):
            visit(node["items"])
        for child in node.get("anyOf", []):
            visit(child)

    visit(enriched)
    return enriched


def make_example(row: dict[str, Any], layout: str, descriptions: bool, row_tolerance: float) -> tuple[str, str, str]:
    annotation = json.loads(row["ground_truth"])
    gold = normalize_gold(annotation["gt_parse"])
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema_for(gold)}
    if descriptions:
        schema = add_descriptions(schema)
    return ocr_report(annotation, layout, row_tolerance), json.dumps(schema, ensure_ascii=False, indent=2), json.dumps(gold, ensure_ascii=False)


def make_record(index: int, row: dict[str, Any], layout: str = "lines", descriptions: bool = False, example: tuple[str, str, str] | None = None, row_tolerance: float = 0.5) -> dict[str, Any]:
    annotation = json.loads(row["ground_truth"])
    gold = normalize_gold(annotation["gt_parse"])
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema_for(gold)}
    if descriptions:
        schema = add_descriptions(schema)
    report = ocr_report(annotation, layout, row_tolerance)
    example_prefix = ""
    if example is not None:
        example_report, example_schema, example_gold = example
        example_prefix = (
            "=== Worked example OCR ===\n"
            f"{example_report}\n\n=== Worked example JSON Schema ===\n{example_schema}"
            f"\n\n=== Worked example JSON ===\n{example_gold}\n\n"
        )
    return {
        "stem": f"cord_test_{index:03d}",
        "user_prompt": f"{example_prefix}{PROMPT_PREFIX}{report}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}",
        "gold_json": json.dumps(gold, ensure_ascii=False),
        "json_schema": json.dumps(schema, ensure_ascii=False),
        "source_split": "CORD-v2/test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CORD-v2 test rows to STAGE benchmark JSONL.")
    parser.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--layout", choices=("lines", "rows"), default="lines")
    parser.add_argument("--row-tolerance", type=float, default=0.5, help="Rows: multiplier of median word height (0.4--0.6).")
    parser.add_argument("--field-descriptions", action="store_true")
    parser.add_argument("--one-shot", action="store_true", help="Prefix a deterministic CORD train example with 3+ items and @/X notation.")
    parser.add_argument("--output", default="benchmark/data/cord_v2_test_100.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not 0.4 <= args.row_tolerance <= 0.6:
        raise SystemExit("--row-tolerance must be between 0.4 and 0.6")

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    example = None
    if args.one_shot:
        train = load_dataset(args.dataset, split="train")
        for candidate in train:
            annotation = json.loads(candidate["ground_truth"])
            menu = normalize_gold(annotation["gt_parse"]).get("menu", [])
            rendered = ocr_report(annotation, args.layout, args.row_tolerance)
            if isinstance(menu, list) and len(menu) >= 3 and ("@" in rendered or "X" in rendered):
                example = make_example(candidate, args.layout, args.field_descriptions, args.row_tolerance)
                break
        if example is None:
            raise SystemExit("No CORD train example with 3+ menu items and @/X notation was found.")
    records = [make_record(i, dataset[i], args.layout, args.field_descriptions, example, args.row_tolerance) for i in range(min(args.limit, len(dataset)))]
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"exported {len(records)} CORD rows: {output}")


if __name__ == "__main__":
    main()
