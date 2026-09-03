"""Build table-grounded STAGE SFT examples without target-to-source leakage.

The original STAGE reports contain an explanatory narrative followed by one or
more ``## Sheet:`` Markdown tables.  This builder uses only those source
tables.  It never renders a source from ``gold_json``.  Gold is used solely to
(1) keep examples whose requested values occur literally in the tables and
(2) make a schema-field subset task, which teaches the model not to fill
unrequested fields.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")
PROMPT = """Extract only information supported by the source document according to the JSON Schema.
Return exactly one JSON object and no explanation.

=== Source document ===
{source}

=== JSON Schema ===
{schema}
"""
SHEET_HEADING = re.compile(r"^##\s+Sheet:\s*(.+?)\s*$", re.MULTILINE)


def report_only(prompt: str) -> str:
    return prompt.split("=== Report ===", 1)[-1].strip()


def parse_table(lines: list[str]) -> list[list[str]] | None:
    """Parse one GitHub-flavoured Markdown table, preserving every cell."""
    if len(lines) < 2 or not all("|" in line for line in lines):
        return None
    separator = [cell.strip().strip(":-") for cell in lines[1].strip().strip("|").split("|")]
    if not separator or any(cell for cell in separator):
        return None
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None
    return [rows[0], *rows[2:]]


def sheet_tables(report: str) -> list[tuple[str, list[list[str]]]]:
    """Return only tables that occur under an actual Sheet heading.

    The narrative often contains illustrative tables.  Those are deliberately
    excluded: they are not the spreadsheet source and can teach unsupported
    value completion.
    """
    headings = list(SHEET_HEADING.finditer(report))
    result: list[tuple[str, list[list[str]]]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(report)
        section = report[heading.end():end]
        lines = section.splitlines()
        cursor = 0
        while cursor < len(lines):
            if "|" not in lines[cursor]:
                cursor += 1
                continue
            start = cursor
            while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
                cursor += 1
            table = parse_table(lines[start:cursor])
            if table:
                result.append((heading.group(1), table))
    return result


def markdown_source(tables: list[tuple[str, list[list[str]]]]) -> str:
    chunks = []
    for name, rows in tables:
        width = len(rows[0])
        chunks.extend([f"## Sheet: {name}", "| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"])
        chunks.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(chunks)


def tsv_source(tables: list[tuple[str, list[list[str]]]]) -> str:
    return "\n\n".join("=== Sheet: " + name + " ===\n" + "\n".join("\t".join(row) for row in rows) for name, rows in tables)


def html_source(tables: list[tuple[str, list[list[str]]]]) -> str:
    chunks = []
    for name, rows in tables:
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in rows[0])
        body = "\n".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows[1:])
        chunks.append(f"<section data-sheet={json.dumps(name, ensure_ascii=False)}><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>")
    return "\n".join(chunks)


def normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def primitive_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in primitive_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in primitive_values(child)]
    if value is None:
        return []
    text = normalized(value)
    return [text] if text else []


def evidence_coverage(gold: Any, source: str) -> tuple[float, int, int]:
    values = primitive_values(gold)
    source_text = normalized(source)
    found = sum(value in source_text for value in values)
    return (found / len(values) if values else 1.0, found, len(values))


def select_properties(gold: dict[str, Any], schema: dict[str, Any], seed: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or len(properties) < 2:
        return None
    available = [key for key in properties if key in gold]
    if len(available) < 2:
        return None
    rng = random.Random(seed)
    rng.shuffle(available)
    chosen = sorted(available[: max(1, len(available) // 2)])
    subset_schema = {key: value for key, value in schema.items() if key not in {"properties", "required"}}
    subset_schema["properties"] = {key: properties[key] for key in chosen}
    subset_schema["required"] = chosen
    return {key: gold[key] for key in chosen}, subset_schema


def record(stem: str, source: str, gold: Any, schema: dict[str, Any], fmt: str, task: str, coverage: float) -> dict[str, Any]:
    return {
        "source": "stage_table_grounded",
        "source_stem": stem,
        "source_format": fmt,
        "task": task,
        "table_value_coverage": coverage,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(source=source, schema=json.dumps(schema, ensure_ascii=False))},
            {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, indent=2)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/STAGE-eval/data/train-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=1200)
    parser.add_argument("--min-report-chars", type=int, default=3500)
    parser.add_argument("--max-source-chars", type=int, default=16000)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = pq.read_table(args.input).to_pylist()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected: list[tuple[str, list[tuple[str, list[list[str]]]], Any, dict[str, Any], float]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        try:
            gold, schema = json.loads(row["gold_json"]), json.loads(row["json_schema"])
        except (KeyError, json.JSONDecodeError):
            skipped["invalid_gold_or_schema"] += 1
            continue
        if not isinstance(gold, dict) or not isinstance(schema, dict):
            skipped["non_object_target"] += 1
            continue
        report = report_only(row["user_prompt"])
        if len(report) < args.min_report_chars:
            skipped["short_report"] += 1
            continue
        tables = sheet_tables(report)
        if not tables:
            skipped["no_sheet_table"] += 1
            continue
        source = markdown_source(tables)
        if len(source) > args.max_source_chars:
            skipped["table_source_too_long"] += 1
            continue
        coverage, _, _ = evidence_coverage(gold, source)
        if coverage < args.min_coverage:
            skipped["gold_not_fully_in_sheet_tables"] += 1
            continue
        selected.append((row["stem"], tables, gold, schema, coverage))
        if len(selected) == args.sources:
            break
    if len(selected) < args.sources:
        raise SystemExit(f"only {len(selected)} eligible rows, requested {args.sources}; skipped={dict(skipped)}")
    examples: list[dict[str, Any]] = []
    for stem, tables, gold, schema, coverage in selected:
        representations = {"markdown_table": markdown_source(tables), "tsv_table": tsv_source(tables), "html_table": html_source(tables)}
        for fmt, source in representations.items():
            examples.append(record(stem, source, gold, schema, fmt, "full_schema", coverage))
        subset = select_properties(gold, schema, f"{args.seed}:{stem}")
        if subset is not None:
            subset_gold, subset_schema = subset
            for fmt in ("markdown_table", "tsv_table"):
                examples.append(record(stem, representations[fmt], subset_gold, subset_schema, fmt, "requested_field_subset", coverage))
    rng.shuffle(examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "sources": len(selected), "examples": len(examples), "formats": ["markdown_table", "tsv_table", "html_table"],
        "tasks": ["full_schema", "requested_field_subset"], "min_report_chars": args.min_report_chars,
        "max_source_chars": args.max_source_chars, "min_coverage": args.min_coverage, "seed": args.seed,
        "skipped_before_selection": dict(skipped), "source_rule": "Only parsed ## Sheet: Markdown tables; source is never rendered from gold_json.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
