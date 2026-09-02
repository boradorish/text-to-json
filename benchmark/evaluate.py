"""
Evaluate benchmark inference JSONL without LLM-as-a-judge metrics.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install jsonschema first: pip install jsonschema") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def extract_leaves(obj: Any, path: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else str(key)
            result.update(extract_leaves(value, new_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            result.update(extract_leaves(value, f"{path}[{index}]"))
    else:
        result[path] = obj
    return result


def schema_leaf_paths(schema: dict, path: str = "") -> set[str]:
    keys: set[str] = set()
    for key, sub_schema in schema.get("properties", {}).items():
        new_path = f"{path}.{key}" if path else key
        keys.add(new_path)
        if isinstance(sub_schema, dict):
            keys.update(schema_leaf_paths(sub_schema, new_path))
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        keys.update(schema_leaf_paths(additional, path + ".*"))
    return keys


def compute_noise_ratio(pred_obj: Any, schema: dict) -> float:
    pred_leaves = extract_leaves(pred_obj)
    if not pred_leaves:
        return 0.0
    schema_key_names = {p.split(".")[-1] for p in schema_leaf_paths(schema)}
    pred_key_names = {
        re.split(r"[.\[]", segment)[0]
        for path in pred_leaves
        for segment in path.split(".")
        if segment
    }
    extra = pred_key_names - schema_key_names
    return len(extra) / len(pred_key_names) if pred_key_names else 0.0


def value_match(pred_obj: Any, gold_obj: Any) -> float:
    gold_leaves = extract_leaves(gold_obj)
    if not gold_leaves:
        return 1.0
    pred_leaves = extract_leaves(pred_obj) if pred_obj is not None else {}
    matched = sum(1 for path, value in gold_leaves.items() if pred_leaves.get(path) == value)
    return matched / len(gold_leaves)


def evaluate_row(record: dict) -> dict:
    try:
        pred_obj = json.loads(record.get("pred_json") or "")
    except Exception:
        pred_obj = None
    try:
        gold_obj = json.loads(record.get("gold_json") or "")
    except Exception:
        gold_obj = None
    try:
        schema = json.loads(record.get("json_schema") or "")
    except Exception:
        schema = {}

    if pred_obj is None:
        return {
            "no_output": True,
            "exact_match": False,
            "schema_valid": False,
            "noise_ratio": 1.0,
            "value_match": 0.0,
        }

    try:
        jsonschema.validate(instance=pred_obj, schema=schema)
        schema_valid = True
        noise_ratio = 0.0
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        schema_valid = False
        noise_ratio = compute_noise_ratio(pred_obj, schema) if schema else 1.0

    return {
        "no_output": False,
        "exact_match": pred_obj == gold_obj,
        "schema_valid": schema_valid,
        "noise_ratio": noise_ratio,
        "value_match": value_match(pred_obj, gold_obj) if gold_obj is not None else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark inference results.")
    parser.add_argument("--input", default="benchmark/runs/infer_results.jsonl")
    parser.add_argument("--output", default=None, help="Default: input path with _eval.xlsx suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = (
        resolve_path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_eval.xlsx")
    )
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    skipped_records = [record for record in records if record.get("skip_reason")]
    records = [record for record in records if not record.get("skip_reason")]
    if not records:
        raise SystemExit("No evaluable records after excluding skipped rows.")
    metrics = [evaluate_row(record) for record in records]
    metrics_df = pd.DataFrame(metrics)
    out_df = pd.concat([pd.DataFrame(records), metrics_df], axis=1)
    summary_df = pd.DataFrame(
        [
            {
                "samples": len(metrics_df),
                "skipped_samples": len(skipped_records),
                "no_output_ratio": metrics_df["no_output"].mean(),
                "exact_match_ratio": metrics_df["exact_match"].mean(),
                "schema_valid_ratio": metrics_df["schema_valid"].mean(),
                "mean_noise_ratio": metrics_df["noise_ratio"].mean(),
                "mean_value_match": metrics_df["value_match"].mean(),
            }
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact raw records in JSONL, but XLSX is XML-backed and cannot
    # represent control bytes occasionally emitted by unconstrained models.
    excel_df = out_df.copy()
    for column in excel_df.select_dtypes(include="object"):
        excel_df[column] = excel_df[column].map(
            lambda value: re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", value)
            if isinstance(value, str)
            else value
        )
    with pd.ExcelWriter(output_path) as writer:
        excel_df.to_excel(writer, sheet_name="rows", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 55)
    print("Benchmark evaluation")
    print("=" * 55)
    print(f"  samples:                  {len(metrics_df)}")
    print(f"  skipped samples:          {len(skipped_records)}")
    print(f"  no_output ratio:           {metrics_df['no_output'].mean():.4f}")
    print(f"  exact_match ratio:         {metrics_df['exact_match'].mean():.4f}")
    print(f"  schema_valid ratio:        {metrics_df['schema_valid'].mean():.4f}")
    print(f"  mean noise_ratio:          {metrics_df['noise_ratio'].mean():.4f}")
    print(f"  mean value_match (rule):   {metrics_df['value_match'].mean():.4f}")
    print("=" * 55)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
