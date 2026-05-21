"""
MasterControlAIML/JSON-Unstructured-Structured -> LLaMA-Factory SFT data.

The source dataset contains synthetic unstructured text, JSON Schema, filled
structured JSON, and sometimes extra layout/schema rules. This script converts
each row into the same ShareGPT JSONL shape used by the Glaive and ScrapeGraphAI
preparation scripts.

Output format: sharegpt jsonl
  {"conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
  ]}

Usage:
    python3 src/prepare_mastercontrol_sft.py
    python3 src/prepare_mastercontrol_sft.py --num-samples 1500 --output data/sft/mastercontrol_sft_1_5k.jsonl
    python3 src/prepare_mastercontrol_sft.py --inspect --num-samples 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()

DEFAULT_SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. Read the report and JSON Schema, "
    "then return only valid JSON that matches the schema. Do not include markdown, "
    "code fences, explanations, or XML-like tags."
)

USER_TEMPLATE = """Extract the structured JSON object from the unstructured report.

=== REPORT ===
{report}

=== JSON SCHEMA ===
{schema}
{rules_block}

Return ONLY valid JSON matching the schema."""

REPORT_FIELD_NAMES = (
    "text",
    "report",
    "content",
    "document",
    "unstructured_text",
    "input",
    "prompt",
)
SCHEMA_FIELD_NAMES = (
    "json_schema",
    "schema",
    "rules",
    "schema_rules",
    "set_of_rules",
    "set_of_rules_for_schema_creation",
)
JSON_FIELD_NAMES = (
    "object",
    "json",
    "structured_json",
    "filled_structured_json",
    "output",
    "answer",
    "response",
    "target",
)
EXTRA_RULE_HINTS = ("layout", "rule", "instruction", "format", "table")


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]"


def _json_loads(value: Any) -> Any | None:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text[0] not in "[{":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _json_dumps(obj: Any, pretty: bool) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _normalize_json_text(value: Any, pretty: bool) -> str | None:
    obj = _json_loads(value)
    if obj is None:
        return None
    return _json_dumps(obj, pretty)


def _is_schema_obj(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    keys = set(obj)
    if "$schema" in keys:
        return True
    if obj.get("type") == "object" and isinstance(obj.get("properties"), dict):
        return True
    if "required" in keys and "properties" in keys:
        return True
    return False


def _get_by_names(row: dict[str, Any], names: Iterable[str]) -> tuple[str, Any] | None:
    lower_to_key = {key.lower(): key for key in row}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is None:
            continue
        value = row.get(key)
        if value not in (None, ""):
            return key, value
    return None


def _infer_schema_field(row: dict[str, Any]) -> tuple[str, Any] | None:
    named = _get_by_names(row, SCHEMA_FIELD_NAMES)
    if named and _is_schema_obj(_json_loads(named[1])):
        return named

    for key, value in row.items():
        obj = _json_loads(value)
        if _is_schema_obj(obj):
            return key, value
    return named


def _infer_json_field(row: dict[str, Any], schema_key: str | None) -> tuple[str, Any] | None:
    named = _get_by_names(row, JSON_FIELD_NAMES)
    if named and named[0] != schema_key and _json_loads(named[1]) is not None:
        return named

    for key, value in row.items():
        if key == schema_key:
            continue
        obj = _json_loads(value)
        if obj is not None and not _is_schema_obj(obj):
            return key, value
    return named if named and named[0] != schema_key else None


def _infer_report_field(
    row: dict[str, Any],
    schema_key: str | None,
    json_key: str | None,
) -> tuple[str, Any] | None:
    named = _get_by_names(row, REPORT_FIELD_NAMES)
    if named and named[0] not in {schema_key, json_key}:
        return named

    candidates = []
    for key, value in row.items():
        if key in {schema_key, json_key}:
            continue
        if _json_loads(value) is not None:
            continue
        text = str(value or "").strip()
        if not text:
            continue
        candidates.append((len(text), key, value))

    if not candidates:
        return None
    _, key, value = max(candidates)
    return key, value


def _collect_extra_rules(
    row: dict[str, Any],
    used_keys: set[str],
    max_rule_chars: int,
) -> str:
    parts = []
    for key, value in row.items():
        if key in used_keys:
            continue
        text = str(value or "").strip()
        if not text or _json_loads(value) is not None:
            continue
        lower_key = key.lower()
        if any(hint in lower_key for hint in EXTRA_RULE_HINTS):
            parts.append(f"{key}:\n{_truncate(text, max_rule_chars)}")

    if not parts:
        return ""
    return "\n=== ADDITIONAL RULES ===\n" + "\n\n".join(parts)


def convert_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    schema_field = _infer_schema_field(row)
    schema_key = schema_field[0] if schema_field else None
    json_field = _infer_json_field(row, schema_key)
    json_key = json_field[0] if json_field else None
    report_field = _infer_report_field(row, schema_key, json_key)

    if schema_field is None or json_field is None or report_field is None:
        return None

    schema = _normalize_json_text(schema_field[1], args.pretty)
    assistant = _normalize_json_text(json_field[1], args.pretty)
    report = _truncate(str(report_field[1] or ""), args.max_report_chars)
    if not schema or not assistant or not report:
        return None

    used_keys = {schema_field[0], json_field[0], report_field[0]}
    rules_block = _collect_extra_rules(row, used_keys, args.max_rule_chars)

    return {
        "conversations": [
            {"from": "system", "value": args.system_prompt},
            {"from": "human", "value": USER_TEMPLATE.format(
                report=report,
                schema=_truncate(schema, args.max_schema_chars),
                rules_block=rules_block,
            )},
            {"from": "gpt", "value": assistant},
        ]
    }


def print_dataset_info(output_path: Path) -> None:
    dataset_name = "mastercontrol_sft"
    print("\n[LLaMA-Factory dataset_info.json entry]")
    print(json.dumps({
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }
    }, indent=2, ensure_ascii=False))


def _inspect_row(row: dict[str, Any]) -> None:
    schema_field = _infer_schema_field(row)
    json_field = _infer_json_field(row, schema_field[0] if schema_field else None)
    report_field = _infer_report_field(
        row,
        schema_field[0] if schema_field else None,
        json_field[0] if json_field else None,
    )

    print("\n[Detected fields]")
    print(f"  report: {report_field[0] if report_field else None}")
    print(f"  schema: {schema_field[0] if schema_field else None}")
    print(f"  json:   {json_field[0] if json_field else None}")
    print("\n[Columns]")
    for key, value in row.items():
        text = str(value or "").replace("\n", "\\n")
        print(f"  {key}: {text[:180]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MasterControl JSON-Unstructured-Structured SFT data")
    parser.add_argument("--dataset", default="MasterControlAIML/JSON-Unstructured-Structured")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--output", default="data/sft/mastercontrol_sft_1_5k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-report-chars", type=int, default=12000)
    parser.add_argument("--max-schema-chars", type=int, default=8000)
    parser.add_argument("--max-rule-chars", type=int, default=3000)
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    print(f"{args.dataset} 로드 중...")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.shuffle(seed=args.seed)
    print(f"총 {len(ds):,}개 로드 완료")

    if args.inspect:
        for row in ds.select(range(min(args.num_samples, len(ds)))):
            _inspect_row(dict(row))
        return

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(ds, desc="처리 중"):
            if written >= args.num_samples:
                break

            record = convert_row(dict(row), args)
            if record is None:
                skipped += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print("\n완료.")
    print(f"  저장: {written:,}개")
    print(f"  스킵: {skipped:,}개")
    print(f"  출력: {output_path}")
    print(f"  복사: cp {output_path} ../LLaMA-Factory/data/{output_path.name}")
    print_dataset_info(output_path)


if __name__ == "__main__":
    main()
