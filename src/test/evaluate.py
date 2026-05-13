"""
infer.py 가 생성한 Excel 파일을 읽어 메트릭을 산출합니다.
JSON 파싱 실패(pred_json 비어 있음)는 모든 메트릭 0으로 집계됩니다.

사용법:
    python src/test/evaluate.py
    python src/test/evaluate.py --input data/infer_results.xlsx
    python src/test/evaluate.py --llm --llm-model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _extract_leaves(obj: Any, path: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            result.update(_extract_leaves(v, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            result.update(_extract_leaves(v, f"{path}[{i}]"))
    else:
        result[path] = obj
    return result


def _get_schema_leaf_paths(schema: dict, path: str = "") -> set[str]:
    keys: set[str] = set()
    for k, sub in schema.get("properties", {}).items():
        new_path = f"{path}.{k}" if path else k
        keys.add(new_path)
        if isinstance(sub, dict):
            keys.update(_get_schema_leaf_paths(sub, new_path))
    add_props = schema.get("additionalProperties")
    if isinstance(add_props, dict):
        keys.update(_get_schema_leaf_paths(add_props, path + ".*"))
    return keys


def _compute_noise_ratio(pred_obj: Any, schema: dict) -> float:
    pred_leaves = _extract_leaves(pred_obj)
    if not pred_leaves:
        return 0.0
    schema_key_names = {p.split(".")[-1] for p in _get_schema_leaf_paths(schema)}
    pred_key_names = {
        re.split(r"[.\[]", seg)[0]
        for path in pred_leaves
        for seg in path.split(".")
        if seg
    }
    extra = pred_key_names - schema_key_names
    return len(extra) / len(pred_key_names) if pred_key_names else 0.0


# ---------------------------------------------------------------------------
# 개별 메트릭
# ---------------------------------------------------------------------------

def _exact_match(pred_obj: Any, gold_obj: Any) -> bool:
    return pred_obj == gold_obj


def _schema_match(pred_obj: Any, schema: dict) -> dict:
    if not HAS_JSONSCHEMA:
        raise ImportError("pip install jsonschema")
    try:
        jsonschema.validate(instance=pred_obj, schema=schema)
        return {"valid": True, "noise_ratio": 0.0}
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return {"valid": False, "noise_ratio": _compute_noise_ratio(pred_obj, schema)}


def _value_match_rule(pred_obj: Any, gold_obj: Any) -> float:
    gold_leaves = _extract_leaves(gold_obj)
    if not gold_leaves:
        return 1.0
    pred_leaves = _extract_leaves(pred_obj) if pred_obj is not None else {}
    matched = sum(1 for path, val in gold_leaves.items() if pred_leaves.get(path) == val)
    return matched / len(gold_leaves)


def _value_match_llm(pred_text: str, gold_obj: Any, model: str = "gpt-4o-mini") -> float | None:
    import os
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    gold_str = json.dumps(gold_obj, ensure_ascii=False, indent=2)
    prompt = f"""You are evaluating a JSON output against a ground truth JSON.

## Ground Truth:
{gold_str}

## Model Output:
{pred_text}

Score from 1 to 5 based on value-level match:
5 = all values match exactly
4 = most values match, minor differences
3 = about half match
2 = few values match
1 = no values match or output is invalid JSON

Reply with ONLY a single integer 1-5."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        m = re.search(r"[1-5]", raw)
        if not m:
            return None
        return (float(m.group()) - 1) / 4  # 0-1 정규화
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 행 평가
# ---------------------------------------------------------------------------

def evaluate_row(
    pred_json_str: str,
    gold_json_str: str,
    schema_str: str,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    try:
        gold_obj = json.loads(gold_json_str) if gold_json_str else None
    except Exception:
        gold_obj = None

    try:
        schema = json.loads(schema_str) if schema_str else {}
    except Exception:
        schema = {}

    # 예측 파싱 실패 시 모든 메트릭 0
    pred_obj = None
    if pred_json_str and pred_json_str.strip():
        try:
            pred_obj = json.loads(pred_json_str)
        except Exception:
            pass

    if pred_obj is None:
        result = {
            "no_output": True,
            "exact_match": False,
            "schema_valid": False,
            "noise_ratio": 1.0,
            "value_match": 0.0,
        }
        if use_llm:
            result["llm_score"] = None
        return result

    sm = _schema_match(pred_obj, schema) if schema else {"valid": False, "noise_ratio": 1.0}
    result = {
        "no_output": False,
        "exact_match": _exact_match(pred_obj, gold_obj),
        "schema_valid": sm["valid"],
        "noise_ratio": sm["noise_ratio"],
        "value_match": _value_match_rule(pred_obj, gold_obj) if gold_obj is not None else 0.0,
    }
    if use_llm:
        result["llm_score"] = (
            _value_match_llm(pred_json_str, gold_obj, llm_model) if gold_obj is not None else None
        )

    return result


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="infer.py JSONL 결과 평가")
    parser.add_argument("--input", default="data/infer_results.jsonl", help="infer.py 출력 JSONL")
    parser.add_argument("--output", default=None, help="결과 저장 Excel 경로 (기본: input과 같은 위치에 .xlsx)")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.prompt_loader import find_project_root
    project_root = find_project_root()

    input_path = project_root / args.input
    output_path = project_root / (args.output if args.output else Path(args.input).with_suffix(".xlsx"))

    if not input_path.exists():
        print(f"[ERROR] 파일 없음: {input_path}")
        sys.exit(1)

    # JSONL 로드
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"로드: {input_path} ({len(records)}개)\n")

    metric_rows = []
    for i, row in enumerate(records):
        pred = row.get("pred_json", "") or ""
        gold = row.get("gold_json", "") or ""
        schema = row.get("json_schema", "") or ""

        m = evaluate_row(pred, gold, schema, use_llm=args.llm, llm_model=args.llm_model)
        metric_rows.append(m)

        stem = row.get("stem", i)
        status = "FAIL" if m["no_output"] else ("OK  " if m["schema_valid"] else "WARN")
        print(
            f"  [{status}] {stem}  exact={m['exact_match']}  "
            f"schema={m['schema_valid']}  value={m['value_match']:.2f}"
        )

    # 원본 + 메트릭 합쳐서 Excel 저장
    metrics_df = pd.DataFrame(metric_rows)
    base_df = pd.DataFrame(records)
    out_df = pd.concat([base_df, metrics_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)

    # 요약
    n = len(metrics_df)
    print("\n" + "=" * 55)
    print("평가 요약")
    print("=" * 55)
    print(f"  총 샘플 수:                 {n}")
    print(f"  no_output 비율:             {metrics_df['no_output'].mean():.4f}")
    print(f"  exact_match 비율:           {metrics_df['exact_match'].mean():.4f}")
    print(f"  schema_match 비율:          {metrics_df['schema_valid'].mean():.4f}")
    print(f"  평균 noise_ratio:           {metrics_df['noise_ratio'].mean():.4f}")
    print(f"  평균 value_match (rule):    {metrics_df['value_match'].mean():.4f}")
    if args.llm and "llm_score" in metrics_df.columns:
        valid_scores = metrics_df["llm_score"].dropna()
        if len(valid_scores):
            print(f"  평균 value_match (LLM):     {valid_scores.mean():.4f}  ({len(valid_scores)}개 평가)")
    print("=" * 55)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
