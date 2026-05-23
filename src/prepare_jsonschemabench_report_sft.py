"""
Prepare JSONSchemaBench report-grounded SFT data.

This script turns schema-only JSONSchemaBench rows into synthetic extraction
examples:

  1. Generate a minimal schema-valid gold JSON with deterministic rules.
  2. Ask Qwen3-4B to write a short report that contains the gold JSON values.
  3. Save ShareGPT SFT data where the user input is report + schema and the
     assistant output is the rule-based gold JSON.

Usage:
    python3 src/prepare_jsonschemabench_report_sft.py
    python3 src/prepare_jsonschemabench_report_sft.py --num-samples 1000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root
from utils.vllm_inference import VllmModel, generate_texts, load_vllm_model

PROJECT_ROOT = find_project_root()

DEFAULT_SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. Read the report and JSON Schema, "
    "then return only valid JSON that matches the schema. Do not include markdown, "
    "code fences, explanations, or XML-like tags."
)

USER_TEMPLATE = """Extract the structured information from the report according to the JSON Schema.

=== REPORT ===
{report}

=== JSON SCHEMA ===
{schema}

Return ONLY valid JSON matching the schema."""

REPORT_SYSTEM_PROMPT = (
    "You write concise factual reports for synthetic data generation. "
    "The report must explicitly contain every value from the provided target JSON, "
    "must not contain contradictory values, and must not include raw JSON, markdown, "
    "code fences, or explanations."
)

REPORT_USER_TEMPLATE = """Write a short realistic report using the target JSON values.

Requirements:
- Mention every scalar value from the target JSON exactly as written.
- Keep the report to 3-8 sentences.
- Do not add conflicting values.
- Do not include raw JSON, bullet lists, markdown, or code fences.

JSON Schema:
{schema}

Target JSON:
{gold_json}

Report:"""

TRIVIAL_SCHEMAS = ({}, {"type": "object"}, {"type": "object", "properties": {}})


def validate_jsonschema(instance: Any, schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=instance, schema=schema)


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    return node if isinstance(node, dict) else {}


def _generate(schema: dict[str, Any], root: dict[str, Any], depth: int = 0) -> Any:
    if depth > 10 or not isinstance(schema, dict):
        return {}

    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], root)
        merged = {**resolved, **{k: v for k, v in schema.items() if k != "$ref"}}
        return _generate(merged, root, depth + 1)

    if "const" in schema:
        return schema["const"]

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0])

    for key in ("anyOf", "oneOf"):
        if key in schema:
            for sub in schema[key]:
                try:
                    value = _generate(sub, root, depth + 1)
                    validate_jsonschema(value, sub)
                    return value
                except Exception:
                    continue
            return {}

    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for sub in schema["allOf"]:
            if isinstance(sub, dict):
                merged.update(sub)
        return _generate(merged, root, depth + 1)

    if "if" in schema and "then" in schema:
        return _generate(schema["then"], root, depth + 1)

    if schema_type == "null":
        return None
    if schema_type == "boolean":
        return True

    if schema_type == "integer":
        value = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        if isinstance(value, bool):
            value = 0
        value = int(value)
        if schema.get("exclusiveMinimum") == value:
            value += 1
        maximum = schema.get("maximum")
        if maximum is not None and value > int(maximum):
            value = int(maximum)
        multiple = schema.get("multipleOf")
        if multiple and value % multiple != 0:
            value = (value // multiple + 1) * multiple
        return value

    if schema_type == "number":
        value = schema.get("minimum", schema.get("exclusiveMinimum", 0.0))
        if isinstance(value, bool):
            value = 0.0
        return float(value)

    if schema_type == "string":
        min_len = schema.get("minLength", 1)
        fmt = schema.get("format", "")
        if fmt == "date":
            return "2024-01-01"
        if fmt in ("date-time", "datetime"):
            return "2024-01-01T00:00:00Z"
        if fmt == "email":
            return "user@example.com"
        if fmt == "uri":
            return "https://example.com"
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        if fmt == "ipv4":
            return "0.0.0.0"
        if fmt == "ipv6":
            return "::1"
        if fmt == "time":
            return "00:00:00"
        return "a" * max(int(min_len), 1)

    if schema_type == "array":
        min_items = int(schema.get("minItems", 0))
        prefix_items = schema.get("prefixItems", [])
        items_schema = schema.get("items", {})

        result = []
        if isinstance(prefix_items, list):
            for sub in prefix_items:
                result.append(_generate(sub, root, depth + 1))
        while len(result) < min_items:
            result.append(_generate(items_schema, root, depth + 1) if isinstance(items_schema, dict) else "item")
        return result

    if schema_type == "object" or "properties" in schema or "required" in schema:
        result: dict[str, Any] = {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if not isinstance(required, list):
            required = []
        if not isinstance(properties, dict):
            properties = {}

        for prop in required:
            if isinstance(prop, str):
                result[prop] = _generate(properties.get(prop, {}), root, depth + 1)

        if not required and properties:
            first = next(iter(properties))
            result[first] = _generate(properties[first], root, depth + 1)
        return result

    return {}


def generate_minimal(schema_obj: dict[str, Any]) -> Any | None:
    try:
        result = _generate(schema_obj, schema_obj)
        validate_jsonschema(result, schema_obj)
        return result
    except Exception:
        return None


def scalar_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            values.extend(scalar_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(scalar_values(child))
    elif value is None:
        values.append("null")
    elif isinstance(value, bool):
        values.append(str(value).lower())
    else:
        values.append(str(value))
    return values


def report_mentions_values(report: str, gold_obj: Any, max_missing: int) -> bool:
    report_lower = report.lower()
    missing = 0
    for value in scalar_values(gold_obj):
        if value.lower() not in report_lower:
            missing += 1
            if missing > max_missing:
                return False
    return True


def strip_think(text: str) -> str:
    return re.split(r"</think>", text, maxsplit=1)[-1].strip()


def clean_report(text: str) -> str:
    text = strip_think(text)
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_report_prompt(schema_str: str, gold_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REPORT_USER_TEMPLATE.format(schema=schema_str, gold_json=gold_json),
        },
    ]


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_reports(
    engine: VllmModel,
    jobs: list[tuple[str, str]],
    args: argparse.Namespace,
) -> list[list[str]]:
    repeated = [job for job in jobs for _ in range(args.retries + 1)]
    prompts = [
        render_chat_prompt(engine.tokenizer, build_report_prompt(schema_str, gold_json))
        for schema_str, gold_json in repeated
    ]

    decoded = generate_texts(
        engine,
        prompts,
        max_new_tokens=args.max_report_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    decoded = [clean_report(text) for text in decoded]

    width = args.retries + 1
    return [decoded[i * width : (i + 1) * width] for i in range(len(jobs))]


def print_dataset_info(output_path: Path) -> None:
    dataset_name = "jsonschemabench_report_sft"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONSchemaBench -> report-grounded SFT data")
    parser.add_argument("--dataset", default="epfl-dlab/JSONSchemaBench")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path/id for vLLM.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="data/sft/jsonschemabench_report_sft_1k.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-report-tokens", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-missing-values", type=int, default=0)
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Stop after scanning this many source rows, even if num-samples is not reached.",
    )
    parser.add_argument(
        "--check-report-values",
        action="store_true",
        help="Only accept reports that mention every gold scalar value verbatim, allowing max-missing-values misses.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    from datasets import load_dataset
    from tqdm import tqdm

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    source_limit = min(args.max_source_rows or len(dataset), len(dataset))
    print(f"Loaded {len(dataset):,} rows; scanning up to {source_limit:,} rows")

    engine = load_vllm_model(
        args.model,
        tokenizer_path=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    written = 0
    skipped_parse = 0
    skipped_trivial = 0
    skipped_gold = 0
    skipped_report = 0

    with output_path.open("w", encoding="utf-8") as fout:
        progress = tqdm(total=args.num_samples, desc="Written")
        for batch_start in range(0, source_limit, args.batch_size):
            if written >= args.num_samples:
                break

            batch_end = min(batch_start + args.batch_size, source_limit)
            batch = [dataset[i] for i in range(batch_start, batch_end)]
            rows = []

            for item in batch:
                schema_str = item["json_schema"]
                unique_id = item["unique_id"]

                try:
                    schema_obj = json.loads(schema_str)
                    if not isinstance(schema_obj, dict):
                        skipped_parse += 1
                        continue
                except (json.JSONDecodeError, TypeError):
                    skipped_parse += 1
                    continue

                if schema_obj in TRIVIAL_SCHEMAS:
                    skipped_trivial += 1
                    continue

                gold_obj = generate_minimal(schema_obj)
                if gold_obj is None:
                    skipped_gold += 1
                    continue

                gold_json = json.dumps(gold_obj, ensure_ascii=False)
                rows.append({
                    "schema_str": schema_str,
                    "gold_obj": gold_obj,
                    "gold_json": gold_json,
                    "unique_id": unique_id,
                })

            if not rows:
                continue

            report_candidates = generate_reports(
                engine,
                [(row["schema_str"], row["gold_json"]) for row in rows],
                args,
            )

            for row, candidates in zip(rows, report_candidates):
                if written >= args.num_samples:
                    break

                report = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate
                        and (
                            not args.check_report_values
                            or report_mentions_values(
                                candidate,
                                row["gold_obj"],
                                args.max_missing_values,
                            )
                        )
                    ),
                    None,
                )
                if report is None:
                    skipped_report += 1
                    continue

                record = {
                    "conversations": [
                        {"from": "system", "value": args.system_prompt},
                        {
                            "from": "human",
                            "value": USER_TEMPLATE.format(report=report, schema=row["schema_str"]),
                        },
                        {"from": "gpt", "value": row["gold_json"]},
                    ],
                    "unique_id": row["unique_id"],
                    "_synthetic_report": report,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                progress.update(1)
        progress.close()

    print("\nDone.")
    print(f"  written:              {written:,}")
    print(f"  skipped parse:        {skipped_parse:,}")
    print(f"  skipped trivial:      {skipped_trivial:,}")
    print(f"  skipped gold:         {skipped_gold:,}")
    print(f"  skipped report check: {skipped_report:,}")
    print(f"  output: {output_path}")
    print_dataset_info(output_path)


if __name__ == "__main__":
    main()
