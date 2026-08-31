"""
Compare two JSON extraction result files and quantify recurring error patterns.

The script accepts DeepJSONEval-style XLSX files and benchmark JSONL/XLSX files.
It joins rows by id/stem/index, recomputes deterministic JSON/schema/value
signals, and writes an Excel workbook with error distributions, transition
patterns, path-level residual errors, and review examples.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install jsonschema first: pip install jsonschema") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_TRUNCATE = 32000
TRUNCATED_SUFFIXES = ("...", "…")
TOKEN_LIMIT_SENTINEL = 4096
HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
LATIN_RE = re.compile(r"[A-Za-z]")


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file: {path}")


def first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    key_col = first_existing(df, ("stem", "id", "Unnamed: 0", "Unnamed: 0.1"))
    schema_col = first_existing(df, ("json_schema", "schema"))
    gold_col = first_existing(df, ("gold_json", "json", "answer", "ground_truth"))
    pred_col = first_existing(df, ("pred_json", "model_output", "prediction", "output"))
    raw_col = first_existing(df, ("raw_output", "model_output", "prediction", "output"))
    text_col = first_existing(df, ("user_prompt", "text", "prompt", "input"))

    missing = [
        name
        for name, col in {
            "key": key_col,
            "schema": schema_col,
            "gold": gold_col,
            "pred/model_output": pred_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")

    out = pd.DataFrame()
    out["row_key"] = df[key_col].astype(str) if key_col else df.index.astype(str)
    out["schema"] = df[schema_col].fillna("").astype(str)
    out["gold_json"] = df[gold_col].fillna("").astype(str)
    out["pred_source"] = df[pred_col].fillna("").astype(str)
    out["raw_output"] = df[raw_col].fillna("").astype(str) if raw_col else out["pred_source"]
    out.loc[out["pred_source"].str.strip() == "", "pred_source"] = out.loc[
        out["pred_source"].str.strip() == "", "raw_output"
    ]
    out["text"] = df[text_col].fillna("").astype(str) if text_col else ""

    for col in (
        "category",
        "true_depth",
        "input_tokens",
        "prompt_tokens",
        "completion_tokens",
        "format_score",
        "detailed_score",
        "strict_score",
    ):
        out[col] = df[col] if col in df.columns else None
    return out


def strip_think(text: str) -> str:
    return re.split(r"</think>", text, maxsplit=1)[-1].strip()


def parse_jsonish(text: str) -> tuple[Any | None, str, str]:
    raw = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    content = strip_think(raw)

    candidates: list[tuple[str, str]] = []
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content, flags=re.IGNORECASE)
    if fenced:
        candidates.append(("fenced_json", fenced.group(1).strip()))
    candidates.append(("whole_text", content))
    loose = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
    if loose:
        candidates.append(("loose_json", loose.group(1).strip()))

    for source, candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate), source, ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    return None, "parse_fail", last_error if "last_error" in locals() else "empty"


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


def normalize_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def schema_error_kind(error: jsonschema.ValidationError) -> str:
    if error.validator == "required":
        return "missing_required"
    if error.validator == "additionalProperties":
        return "extra_key"
    if error.validator == "type":
        return "type_mismatch"
    if error.validator in {"enum", "const"}:
        return "enum_mismatch"
    if error.validator in {"minItems", "maxItems", "items", "prefixItems", "uniqueItems"}:
        return "array_schema"
    return f"schema_{error.validator}"


def schema_error_path(error: jsonschema.ValidationError) -> str:
    base = ".".join(str(part) for part in error.absolute_path)
    if error.validator == "required":
        missing = re.findall(r"'([^']+)' is a required property", error.message)
        if missing:
            return f"{base}.{missing[0]}" if base else missing[0]
    additional = re.findall(r"'([^']+)' was unexpected", error.message)
    if additional:
        return f"{base}.{additional[0]}" if base else additional[0]
    return base or "<root>"


def maybe_truncated(raw_output: str, completion_tokens: Any | None = None) -> bool:
    raw = str(raw_output or "").strip()
    if not raw:
        return False
    if raw.endswith(TRUNCATED_SUFFIXES):
        return True
    try:
        if completion_tokens is not None and not pd.isna(completion_tokens) and int(completion_tokens) >= TOKEN_LIMIT_SENTINEL:
            return True
    except Exception:
        pass
    opens = raw.count("{") + raw.count("[")
    closes = raw.count("}") + raw.count("]")
    return opens > closes + 1


def null_empty_ratio(obj: Any) -> float:
    leaves = extract_leaves(obj)
    if not leaves:
        return 0.0
    bad = sum(1 for value in leaves.values() if value is None or value == "" or value == [])
    return bad / len(leaves)


def detect_language_group(text: str) -> str:
    hangul = len(HANGUL_RE.findall(text or ""))
    latin = len(LATIN_RE.findall(text or ""))
    total = hangul + latin
    if total == 0:
        return "unknown"
    if hangul / total >= 0.2:
        return "ko"
    if hangul:
        return "mixed"
    return "non_ko"


def token_bucket(value: Any) -> str:
    try:
        tokens = int(value)
    except Exception:
        return "unknown"
    if tokens < 0:
        return "unknown"
    if tokens <= 512:
        return "0000-0512"
    if tokens <= 1024:
        return "0513-1024"
    if tokens <= 1536:
        return "1025-1536"
    if tokens <= 2048:
        return "1537-2048"
    if tokens <= 3072:
        return "2049-3072"
    if tokens <= 4096:
        return "3073-4096"
    return "4097+"


def analyze_record(row: pd.Series) -> dict:
    pred_obj, pred_parse_source, pred_parse_error = parse_jsonish(row["pred_source"])
    gold_obj, _, _ = parse_jsonish(row["gold_json"])
    schema_obj, _, _ = parse_jsonish(row["schema"])
    raw_output = str(row.get("raw_output", "") or "")
    completion_tokens = row.get("completion_tokens")

    out = {
        "parse_ok": pred_obj is not None,
        "parse_source": pred_parse_source,
        "parse_error": pred_parse_error,
        "schema_valid": False,
        "exact_match": False,
        "value_match": 0.0,
        "error_type": "unknown",
        "primary_path": "",
        "schema_error_message": "",
        "wrong_value_paths": "",
        "missing_value_paths": "",
        "extra_value_paths": "",
        "num_wrong_value_paths": 0,
        "num_missing_value_paths": 0,
        "num_extra_value_paths": 0,
        "null_empty_ratio": 0.0,
        "truncated_suspect": maybe_truncated(raw_output, completion_tokens),
    }

    if pred_obj is None:
        out["error_type"] = "truncated_output" if out["truncated_suspect"] else "parse_fail"
        return out
    if gold_obj is None:
        out["error_type"] = "gold_parse_fail"
        return out

    out["exact_match"] = pred_obj == gold_obj
    out["null_empty_ratio"] = null_empty_ratio(pred_obj)

    gold_leaves = extract_leaves(gold_obj)
    pred_leaves = extract_leaves(pred_obj)
    wrong_paths = []
    missing_paths = []
    for path, gold_value in gold_leaves.items():
        if path not in pred_leaves:
            missing_paths.append(path)
        elif pred_leaves[path] != gold_value:
            wrong_paths.append(path)
    extra_paths = [path for path in pred_leaves if path not in gold_leaves]

    total = len(gold_leaves)
    matched = total - len(wrong_paths) - len(missing_paths)
    out["value_match"] = matched / total if total else 1.0
    out["wrong_value_paths"] = "; ".join(normalize_path(p) for p in wrong_paths[:25])
    out["missing_value_paths"] = "; ".join(normalize_path(p) for p in missing_paths[:25])
    out["extra_value_paths"] = "; ".join(normalize_path(p) for p in extra_paths[:25])
    out["num_wrong_value_paths"] = len(wrong_paths)
    out["num_missing_value_paths"] = len(missing_paths)
    out["num_extra_value_paths"] = len(extra_paths)

    if schema_obj is not None:
        validator = jsonschema.Draft202012Validator(schema_obj)
        errors = sorted(validator.iter_errors(pred_obj), key=lambda err: len(err.absolute_path))
        if not errors:
            out["schema_valid"] = True
        else:
            primary = errors[0]
            kind = schema_error_kind(primary)
            path = normalize_path(schema_error_path(primary))
            out["error_type"] = f"schema_invalid_{kind}"
            out["primary_path"] = path
            out["schema_error_message"] = primary.message[:500]
            return out

    if out["exact_match"]:
        out["error_type"] = "correct"
    elif out["null_empty_ratio"] >= 0.35:
        out["error_type"] = "empty_or_null_heavy"
        out["primary_path"] = normalize_path((missing_paths or wrong_paths or extra_paths or [""])[0])
    elif missing_paths:
        out["error_type"] = "schema_valid_missing_values"
        out["primary_path"] = normalize_path(missing_paths[0])
    elif extra_paths and len(extra_paths) >= max(3, len(gold_leaves) * 0.25):
        out["error_type"] = "schema_valid_extra_values"
        out["primary_path"] = normalize_path(extra_paths[0])
    else:
        out["error_type"] = "schema_valid_value_wrong"
        out["primary_path"] = normalize_path((wrong_paths or extra_paths or [""])[0])
    return out


def add_analysis(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    metrics = pd.DataFrame([analyze_record(row) for _, row in df.iterrows()])
    return pd.concat([df.reset_index(drop=True), metrics.add_prefix(f"{prefix}_")], axis=1)


def compact_text(text: Any, limit: int = 700) -> str:
    s = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit] + ("..." if len(s) > limit else "")


def summarize_distribution(df: pd.DataFrame, col: str, model_label: str) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=False).rename_axis("error_type").reset_index(name="count")
    counts["ratio"] = counts["count"] / len(df) if len(df) else 0
    counts.insert(0, "model", model_label)
    return counts


def path_summary(df: pd.DataFrame, model_prefix: str) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for col in (f"{model_prefix}_wrong_value_paths", f"{model_prefix}_missing_value_paths", f"{model_prefix}_extra_value_paths", f"{model_prefix}_primary_path"):
        if col not in df.columns:
            continue
        for value in df[col].fillna(""):
            for path in str(value).split("; "):
                path = path.strip()
                if path:
                    counter[path] += 1
    return pd.DataFrame(counter.most_common(100), columns=["path", "count"])


def build_outputs(best: pd.DataFrame, base: pd.DataFrame, best_label: str, base_label: str) -> dict[str, pd.DataFrame]:
    merged = base.merge(best, on="row_key", suffixes=(f"_{base_label}", f"_{best_label}"))

    base_strict = f"strict_score_{base_label}"
    best_strict = f"strict_score_{best_label}"
    if base_strict in merged.columns and best_strict in merged.columns:
        merged["strict_delta"] = merged[best_strict] - merged[base_strict]
    merged["value_match_delta"] = merged[f"{best_label}_value_match"] - merged[f"{base_label}_value_match"]
    merged["transition"] = merged[f"{base_label}_error_type"] + " -> " + merged[f"{best_label}_error_type"]
    merged["outcome"] = "same"
    merged.loc[(merged[f"{base_label}_error_type"] != "correct") & (merged[f"{best_label}_error_type"] == "correct"), "outcome"] = "fixed_to_correct"
    merged.loc[(merged[f"{base_label}_error_type"] == "correct") & (merged[f"{best_label}_error_type"] != "correct"), "outcome"] = "regression_from_correct"
    merged.loc[(merged[f"{base_label}_schema_valid"] == False) & (merged[f"{best_label}_schema_valid"] == True), "outcome"] = "schema_improved"
    merged.loc[(merged[f"{base_label}_schema_valid"] == True) & (merged[f"{best_label}_schema_valid"] == False), "outcome"] = "schema_regressed"
    merged.loc[merged["value_match_delta"] >= 0.25, "outcome"] = "value_improved"
    merged.loc[merged["value_match_delta"] <= -0.25, "outcome"] = "value_regressed"

    overall = pd.DataFrame(
        [
            {
                "metric": "rows",
                base_label: len(merged),
                best_label: len(merged),
                "delta": 0,
            },
            {
                "metric": "parse_ok_rate",
                base_label: merged[f"{base_label}_parse_ok"].mean(),
                best_label: merged[f"{best_label}_parse_ok"].mean(),
                "delta": merged[f"{best_label}_parse_ok"].mean() - merged[f"{base_label}_parse_ok"].mean(),
            },
            {
                "metric": "schema_valid_rate",
                base_label: merged[f"{base_label}_schema_valid"].mean(),
                best_label: merged[f"{best_label}_schema_valid"].mean(),
                "delta": merged[f"{best_label}_schema_valid"].mean() - merged[f"{base_label}_schema_valid"].mean(),
            },
            {
                "metric": "exact_match_rate",
                base_label: merged[f"{base_label}_exact_match"].mean(),
                best_label: merged[f"{best_label}_exact_match"].mean(),
                "delta": merged[f"{best_label}_exact_match"].mean() - merged[f"{base_label}_exact_match"].mean(),
            },
            {
                "metric": "value_match_mean",
                base_label: merged[f"{base_label}_value_match"].mean(),
                best_label: merged[f"{best_label}_value_match"].mean(),
                "delta": merged[f"{best_label}_value_match"].mean() - merged[f"{base_label}_value_match"].mean(),
            },
            {
                "metric": "truncated_suspect_rate",
                base_label: merged[f"{base_label}_truncated_suspect"].mean(),
                best_label: merged[f"{best_label}_truncated_suspect"].mean(),
                "delta": merged[f"{best_label}_truncated_suspect"].mean() - merged[f"{base_label}_truncated_suspect"].mean(),
            },
        ]
    )
    if base_strict in merged.columns and best_strict in merged.columns:
        overall.loc[len(overall)] = {
            "metric": "given_strict_score_mean",
            base_label: merged[base_strict].mean(),
            best_label: merged[best_strict].mean(),
            "delta": merged[best_strict].mean() - merged[base_strict].mean(),
        }

    by_category = pd.DataFrame()
    category_col = f"category_{base_label}"
    if category_col in merged.columns:
        by_category = (
            merged.groupby(category_col, dropna=False)
            .agg(
                rows=("row_key", "size"),
                base_exact=(f"{base_label}_exact_match", "mean"),
                best_exact=(f"{best_label}_exact_match", "mean"),
                base_schema=(f"{base_label}_schema_valid", "mean"),
                best_schema=(f"{best_label}_schema_valid", "mean"),
                base_value=(f"{base_label}_value_match", "mean"),
                best_value=(f"{best_label}_value_match", "mean"),
            )
            .reset_index()
        )
        by_category["value_delta"] = by_category["best_value"] - by_category["base_value"]
        by_category["schema_delta"] = by_category["best_schema"] - by_category["base_schema"]
        by_category["exact_delta"] = by_category["best_exact"] - by_category["base_exact"]
        by_category = by_category.sort_values("value_delta", ascending=False)

    text_col = f"text_{base_label}"
    if text_col in merged.columns:
        merged["language_group"] = merged[text_col].fillna("").map(detect_language_group)
    else:
        merged["language_group"] = "unknown"
    prompt_col = f"prompt_tokens_{base_label}"
    input_col = f"input_tokens_{base_label}"
    length_col = None
    for candidate in (prompt_col, input_col):
        if candidate in merged.columns and merged[candidate].notna().any():
            length_col = candidate
            break
    merged["input_token_bucket"] = merged[length_col].map(token_bucket) if length_col is not None else "unknown"

    by_language = (
        merged.groupby("language_group", dropna=False)
        .agg(
            rows=("row_key", "size"),
            base_exact=(f"{base_label}_exact_match", "mean"),
            best_exact=(f"{best_label}_exact_match", "mean"),
            base_schema=(f"{base_label}_schema_valid", "mean"),
            best_schema=(f"{best_label}_schema_valid", "mean"),
            base_value=(f"{base_label}_value_match", "mean"),
            best_value=(f"{best_label}_value_match", "mean"),
        )
        .reset_index()
    )
    by_language["value_delta"] = by_language["best_value"] - by_language["base_value"]
    by_language["schema_delta"] = by_language["best_schema"] - by_language["base_schema"]
    by_language["exact_delta"] = by_language["best_exact"] - by_language["base_exact"]

    by_token_bucket = (
        merged.groupby("input_token_bucket", dropna=False)
        .agg(
            rows=("row_key", "size"),
            base_exact=(f"{base_label}_exact_match", "mean"),
            best_exact=(f"{best_label}_exact_match", "mean"),
            base_schema=(f"{base_label}_schema_valid", "mean"),
            best_schema=(f"{best_label}_schema_valid", "mean"),
            base_value=(f"{base_label}_value_match", "mean"),
            best_value=(f"{best_label}_value_match", "mean"),
        )
        .reset_index()
        .sort_values("input_token_bucket")
    )
    by_token_bucket["value_delta"] = by_token_bucket["best_value"] - by_token_bucket["base_value"]
    by_token_bucket["schema_delta"] = by_token_bucket["best_schema"] - by_token_bucket["base_schema"]
    by_token_bucket["exact_delta"] = by_token_bucket["best_exact"] - by_token_bucket["base_exact"]

    transition = (
        merged.groupby([f"{base_label}_error_type", f"{best_label}_error_type"], dropna=False)
        .agg(
            count=("row_key", "size"),
            avg_value_delta=("value_match_delta", "mean"),
        )
        .reset_index()
        .sort_values(["count", "avg_value_delta"], ascending=[False, False])
    )

    outcome = (
        merged.groupby("outcome", dropna=False)
        .agg(count=("row_key", "size"), avg_value_delta=("value_match_delta", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    outcome["ratio"] = outcome["count"] / len(merged) if len(merged) else 0

    residual = merged[merged[f"{best_label}_error_type"] != "correct"].copy()
    residual_summary = (
        residual.groupby(f"{best_label}_error_type", dropna=False)
        .agg(
            count=("row_key", "size"),
            avg_best_value=(f"{best_label}_value_match", "mean"),
            avg_delta=("value_match_delta", "mean"),
            example_stems=("row_key", lambda x: ", ".join(list(map(str, x))[:8])),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    residual_by_category = pd.DataFrame()
    if category_col in merged.columns:
        residual_by_category = (
            residual.groupby([category_col, f"{best_label}_error_type"], dropna=False)
            .agg(
                count=("row_key", "size"),
                avg_best_value=(f"{best_label}_value_match", "mean"),
                avg_delta=("value_match_delta", "mean"),
                example_stems=("row_key", lambda x: ", ".join(list(map(str, x))[:8])),
            )
            .reset_index()
            .sort_values(["count", "avg_delta"], ascending=[False, True])
        )

    improvement_patterns = (
        merged[
            (merged[f"{base_label}_error_type"] != "correct")
            & (
                (merged[f"{best_label}_error_type"] == "correct")
                | (merged["value_match_delta"] >= 0.25)
                | ((merged[f"{base_label}_schema_valid"] == False) & (merged[f"{best_label}_schema_valid"] == True))
            )
        ]
        .groupby(["transition"], dropna=False)
        .agg(
            count=("row_key", "size"),
            avg_value_delta=("value_match_delta", "mean"),
            example_stems=("row_key", lambda x: ", ".join(list(map(str, x))[:10])),
        )
        .reset_index()
        .sort_values(["count", "avg_value_delta"], ascending=[False, False])
    )

    regression_patterns = (
        merged[
            (merged[f"{base_label}_error_type"] == "correct")
            | (merged["value_match_delta"] <= -0.25)
            | ((merged[f"{base_label}_schema_valid"] == True) & (merged[f"{best_label}_schema_valid"] == False))
        ]
        .query(f"{best_label}_error_type != 'correct' or value_match_delta <= -0.25")
        .groupby(["transition"], dropna=False)
        .agg(
            count=("row_key", "size"),
            avg_value_delta=("value_match_delta", "mean"),
            example_stems=("row_key", lambda x: ", ".join(list(map(str, x))[:10])),
        )
        .reset_index()
        .sort_values(["count", "avg_value_delta"], ascending=[False, True])
    )

    example_cols = [
        "row_key",
        "transition",
        "outcome",
        "value_match_delta",
        f"category_{base_label}",
        f"prompt_tokens_{base_label}",
        f"completion_tokens_{base_label}",
        f"completion_tokens_{best_label}",
        f"{base_label}_error_type",
        f"{base_label}_primary_path",
        f"{base_label}_schema_error_message",
        f"{best_label}_error_type",
        f"{best_label}_primary_path",
        f"{best_label}_schema_error_message",
        f"text_{base_label}",
        f"gold_json_{base_label}",
        f"raw_output_{base_label}",
        f"raw_output_{best_label}",
    ]
    present_example_cols = [col for col in example_cols if col in merged.columns]
    examples = merged.sort_values(["outcome", "value_match_delta"]).loc[:, present_example_cols].copy()
    for col in examples.columns:
        if examples[col].dtype == object:
            examples[col] = examples[col].map(lambda x: compact_text(x, 1000))

    return {
        "overall_summary": overall,
        "by_category": by_category,
        "by_language": by_language,
        "by_token_bucket": by_token_bucket,
        "outcome_summary": outcome,
        "base_error_distribution": summarize_distribution(merged, f"{base_label}_error_type", base_label),
        "best_error_distribution": summarize_distribution(merged, f"{best_label}_error_type", best_label),
        "transition_matrix": transition,
        "improvement_patterns": improvement_patterns,
        "regression_patterns": regression_patterns,
        "best_residual_errors": residual_summary,
        "best_residual_by_category": residual_by_category,
        "base_path_summary": path_summary(merged, base_label),
        "best_path_summary": path_summary(merged, best_label),
        "examples_for_review": examples.head(300),
        "joined_rows": merged,
    }


def write_workbook(outputs: dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in outputs.items():
            safe = df.copy()
            for col in safe.columns:
                if safe[col].dtype == object:
                    safe[col] = safe[col].map(
                        lambda x: x[:EXCEL_TRUNCATE] + "..." if isinstance(x, str) and len(x) > EXCEL_TRUNCATE else x
                    )
            safe.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and best JSON extraction result files.")
    parser.add_argument("--best", required=True, help="Best/ours result file (.xlsx/.jsonl/.csv)")
    parser.add_argument("--base", required=True, help="Base model result file (.xlsx/.jsonl/.csv)")
    parser.add_argument("--best-label", default="ours")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--output", default="outputs/error_analysis.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_df = add_analysis(normalize_df(load_records(resolve_path(args.best)), args.best_label), args.best_label)
    base_df = add_analysis(normalize_df(load_records(resolve_path(args.base)), args.base_label), args.base_label)
    outputs = build_outputs(best_df, base_df, args.best_label, args.base_label)
    output_path = resolve_path(args.output)
    write_workbook(outputs, output_path)

    overall = outputs["overall_summary"]
    print(overall.to_string(index=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
