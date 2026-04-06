from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _extract_leaves(obj: Any, path: str = "") -> dict[str, Any]:
    """JSON 객체에서 모든 leaf (경로 → 값) 쌍을 추출합니다."""
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
    """schema에 정의된 모든 property 경로를 추출합니다."""
    keys: set[str] = set()
    props = schema.get("properties", {})
    for k, sub in props.items():
        new_path = f"{path}.{k}" if path else k
        keys.add(new_path)
        if isinstance(sub, dict):
            keys.update(_get_schema_leaf_paths(sub, new_path))
    add_props = schema.get("additionalProperties")
    if isinstance(add_props, dict):
        keys.update(_get_schema_leaf_paths(add_props, path + ".*"))
    return keys


def _compute_noise_ratio(pred_obj: Any, schema: dict) -> float:
    """pred_obj에서 schema에 없는 여분 key 이름의 비율을 계산합니다."""
    pred_leaves = _extract_leaves(pred_obj)
    if not pred_leaves:
        return 0.0

    schema_key_names = {p.split(".")[-1] for p in _get_schema_leaf_paths(schema)}
    pred_key_names = {
        re.split(r'[.\[]', seg)[0]
        for path in pred_leaves
        for seg in path.split(".")
        if seg
    }
    extra = pred_key_names - schema_key_names
    return len(extra) / len(pred_key_names) if pred_key_names else 0.0


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------

def parse_json_safe(text: str | None) -> tuple[Any, bool]:
    """텍스트에서 JSON을 파싱합니다. (parsed_obj, is_valid) 반환."""
    if not text:
        return None, False
    text = text.strip()
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1)), True
        except json.JSONDecodeError:
            pass
    return None, False


# ---------------------------------------------------------------------------
# 개별 메트릭
# ---------------------------------------------------------------------------

def exact_match(pred_obj: Any, gold_obj: Any) -> bool:
    """두 JSON 객체가 정확히 동일한지 비교합니다."""
    return pred_obj == gold_obj


def schema_match(pred_obj: Any, schema: dict) -> dict:
    """
    JSON Schema(Draft 2020-12) 검증 결과를 반환합니다.

    Returns:
        {
            "valid": bool,
            "noise_ratio": float,   # valid=False일 때 schema에 없는 key 비율
        }
    """
    if not HAS_JSONSCHEMA:
        raise ImportError("jsonschema 패키지가 필요합니다: pip install jsonschema")
    try:
        jsonschema.validate(instance=pred_obj, schema=schema)
        return {"valid": True, "noise_ratio": 0.0}
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return {"valid": False, "noise_ratio": _compute_noise_ratio(pred_obj, schema)}


def value_match_rule(pred_obj: Any, gold_obj: Any) -> dict:
    """
    gold_obj의 leaf value 중 pred_obj에서 동일 경로+값으로 매칭된 비율을 반환합니다.

    Returns:
        { "matched": int, "total": int, "ratio": float }
    """
    gold_leaves = _extract_leaves(gold_obj)
    if not gold_leaves:
        return {"matched": 0, "total": 0, "ratio": 1.0}

    pred_leaves = _extract_leaves(pred_obj) if pred_obj is not None else {}
    matched = sum(1 for path, val in gold_leaves.items() if pred_leaves.get(path) == val)
    total = len(gold_leaves)
    return {"matched": matched, "total": total, "ratio": matched / total}


def value_match_llm(
    pred_text: str,
    gold_obj: Any,
    model: str = "gemini/gemini-2.0-flash",
) -> dict:
    """
    LLM(gemini-flash)으로 pred_text와 gold_obj를 비교해 1-5 점수를 반환합니다.

    Returns:
        { "score_raw": float, "score_normalized": float }  (0-1 정규화)
    """
    import litellm

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
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        m = re.search(r"[1-5]", raw)
        if not m:
            raise ValueError(f"점수를 파싱할 수 없음: {raw!r}")
        score = float(m.group())
        return {"score_raw": score, "score_normalized": (score - 1) / 4}
    except Exception as e:
        return {"score_raw": None, "score_normalized": None, "error": str(e)}


# ---------------------------------------------------------------------------
# 단일 샘플 평가
# ---------------------------------------------------------------------------

def evaluate_single(
    pred_text: str | None,
    gold_obj: Any,
    schema: dict,
    use_llm: bool = False,
    llm_model: str = "gemini/gemini-2.0-flash",
) -> dict:
    """
    단일 샘플에 대한 전체 평가를 수행합니다.

    Args:
        pred_text: 모델 출력 텍스트 (JSON 포함)
        gold_obj:  정답 JSON 객체
        schema:    JSON Schema 객체
        use_llm:   LLM 기반 value_match 수행 여부
        llm_model: 사용할 LLM 모델 이름

    Returns:
        {
            "no_output": bool,
            "exact_match": bool,
            "schema_match": { "valid": bool, "noise_ratio": float },
            "value_match_rule": { "matched": int, "total": int, "ratio": float },
            "value_match_llm": { ... },   # use_llm=True일 때만 포함
        }
    """
    pred_obj, is_valid_json = parse_json_safe(pred_text)

    result: dict = {
        "no_output": not is_valid_json,
        "exact_match": False,
        "schema_match": {"valid": False, "noise_ratio": 1.0},
        "value_match_rule": {"matched": 0, "total": len(_extract_leaves(gold_obj)), "ratio": 0.0},
    }

    if not is_valid_json:
        if use_llm:
            result["value_match_llm"] = {"score_raw": None, "score_normalized": None, "error": "invalid json"}
        return result

    result["exact_match"] = exact_match(pred_obj, gold_obj)
    result["schema_match"] = schema_match(pred_obj, schema)
    result["value_match_rule"] = value_match_rule(pred_obj, gold_obj)

    if use_llm:
        result["value_match_llm"] = value_match_llm(pred_text, gold_obj, model=llm_model)

    return result


# ---------------------------------------------------------------------------
# 배치 평가
# ---------------------------------------------------------------------------

def evaluate_batch(
    pred_dir: str | Path,
    gold_dir: str | Path,
    schema_dir: str | Path,
    use_llm: bool = False,
    llm_model: str = "gemini/gemini-2.0-flash",
    output_path: str | Path | None = None,
) -> dict:
    """
    디렉토리 내 모든 JSON 파일을 배치로 평가합니다.

    Args:
        pred_dir:    모델 출력 JSON 파일 디렉토리
        gold_dir:    정답 JSON 디렉토리  (보통 data/json/)
        schema_dir:  JSON Schema 디렉토리 (보통 data/json_schema/)
        use_llm:     LLM 기반 value_match 수행 여부
        llm_model:   LLM 모델
        output_path: 결과 저장 경로 (None이면 저장 안 함)

    Returns:
        { "per_file": { stem: metrics }, "summary": aggregated_metrics }
    """
    pred_dir = Path(pred_dir)
    gold_dir = Path(gold_dir)
    schema_dir = Path(schema_dir)

    per_file: dict[str, dict] = {}

    for pred_path in sorted(pred_dir.glob("*.json")):
        stem = pred_path.stem
        gold_path = gold_dir / f"{stem}.json"
        schema_path = schema_dir / f"{stem}.json"

        if not gold_path.exists() or not schema_path.exists():
            print(f"[SKIP] {stem}: gold 또는 schema 파일 없음")
            continue

        try:
            pred_text = pred_path.read_text(encoding="utf-8")
            gold_obj = json.loads(gold_path.read_text(encoding="utf-8"))
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERROR] {stem}: 파일 읽기 실패 - {e}")
            continue

        metrics = evaluate_single(pred_text, gold_obj, schema, use_llm=use_llm, llm_model=llm_model)
        per_file[stem] = metrics

        vm_ratio = metrics["value_match_rule"]["ratio"]
        schema_ok = metrics["schema_match"]["valid"]
        print(
            f"[OK] {stem}  exact={metrics['exact_match']}  "
            f"schema={schema_ok}  value={vm_ratio:.2f}"
            + (f"  noise={metrics['schema_match']['noise_ratio']:.2f}" if not schema_ok else "")
        )

    summary = _summarize(per_file, use_llm=use_llm)

    result = {"per_file": per_file, "summary": summary}

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장 완료: {output_path}")

    _print_summary(summary)
    return result


def _summarize(per_file: dict, use_llm: bool = False) -> dict:
    n = len(per_file)
    if n == 0:
        return {"total": 0}

    no_output_count = sum(1 for m in per_file.values() if m["no_output"])
    exact_count = sum(1 for m in per_file.values() if m["exact_match"])
    schema_valid_count = sum(1 for m in per_file.values() if m["schema_match"]["valid"])

    valid_samples = [m for m in per_file.values() if not m["no_output"]]
    avg_noise = (
        sum(m["schema_match"]["noise_ratio"] for m in valid_samples) / len(valid_samples)
        if valid_samples else 0.0
    )
    avg_value_ratio = sum(m["value_match_rule"]["ratio"] for m in per_file.values()) / n

    summary: dict = {
        "total": n,
        "no_output_rate": no_output_count / n,
        "exact_match_rate": exact_count / n,
        "schema_match_rate": schema_valid_count / n,
        "mean_noise_ratio": avg_noise,
        "mean_value_match_rule": avg_value_ratio,
    }

    if use_llm:
        llm_scores = [
            m["value_match_llm"]["score_normalized"]
            for m in per_file.values()
            if m.get("value_match_llm", {}).get("score_normalized") is not None
        ]
        summary["mean_value_match_llm"] = sum(llm_scores) / len(llm_scores) if llm_scores else None
        summary["llm_evaluated_count"] = len(llm_scores)

    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("평가 요약")
    print("=" * 55)
    labels = {
        "total": "총 샘플 수",
        "no_output_rate": "no_output 비율  (JSON 파싱 실패)",
        "exact_match_rate": "exact_match 비율",
        "schema_match_rate": "schema_match 비율",
        "mean_noise_ratio": "평균 noise_ratio  (schema 위반 샘플 기준)",
        "mean_value_match_rule": "평균 value_match (rule-based)",
        "mean_value_match_llm": "평균 value_match (LLM-based, 0-1)",
        "llm_evaluated_count": "LLM 평가된 샘플 수",
    }
    for k, v in summary.items():
        label = labels.get(k, k)
        if isinstance(v, float):
            print(f"  {label}: {v:.4f}")
        else:
            print(f"  {label}: {v}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# 직접 실행 예시
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="JSON 추론 결과 평가")
    parser.add_argument("--pred", default="data/json_infer", help="모델 출력 JSON 디렉토리")
    parser.add_argument("--gold", default="data/json", help="정답 JSON 디렉토리")
    parser.add_argument("--schema", default="data/json_schema", help="JSON Schema 디렉토리")
    parser.add_argument("--output", default="data/eval_result.json", help="결과 저장 경로")
    parser.add_argument("--llm", action="store_true", help="LLM 기반 value_match 사용")
    parser.add_argument("--llm-model", default="gemini/gemini-2.0-flash", help="LLM 모델")
    args = parser.parse_args()

    evaluate_batch(
        pred_dir=args.pred,
        gold_dir=args.gold,
        schema_dir=args.schema,
        use_llm=args.llm,
        llm_model=args.llm_model,
        output_path=args.output,
    )
