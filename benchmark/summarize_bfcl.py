"""Add BFCL-v4 diagnostic metrics to the official AST evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


OFFLINE = ("simple_python", "multiple", "parallel")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_reference(category: str) -> tuple[dict[str, dict], dict[str, dict]]:
    from bfcl_eval.constants.eval_config import PROMPT_PATH, POSSIBLE_ANSWER_PATH

    questions = {row["id"]: row for row in read_jsonl(PROMPT_PATH / f"BFCL_v4_{category}.json")}
    answers = {row["id"]: row for row in read_jsonl(POSSIBLE_ANSWER_PATH / f"BFCL_v4_{category}.json")}
    return questions, answers


def decode(raw: Any) -> list[dict[str, dict]]:
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting

    if not isinstance(raw, str):
        return []
    try:
        return default_decode_ast_prompting(raw)
    except Exception:
        return []


def type_ok(value: Any, detail: dict[str, Any]) -> bool:
    expected = detail.get("type", "string")
    table = {"string": str, "integer": int, "float": (int, float), "boolean": bool, "array": list, "tuple": (tuple, list), "dict": dict}
    wanted = table.get(expected)
    return wanted is None or isinstance(value, wanted)


def summarize_category(category: str, result_path: Path) -> dict[str, Any]:
    questions, answers = load_reference(category)
    rows = {row["id"]: row for row in read_jsonl(result_path)}
    counters: Counter[str] = Counter()
    for item_id, question in questions.items():
        prediction = decode(rows.get(item_id, {}).get("result"))
        gold_calls = answers[item_id]["ground_truth"]
        counters["examples"] += 1
        counters["expected_calls"] += len(gold_calls)
        # Parallel calls are unordered; select an unused prediction with the
        # required function name. Multiple/simple retain the BFCL order.
        unused = list(range(len(prediction)))
        matched: list[tuple[dict, dict, dict]] = []
        for gold in gold_calls:
            name = next(iter(gold))
            chosen = next((i for i in unused if name in prediction[i]), None)
            if chosen is None:
                continue
            unused.remove(chosen)
            counters["function_selected"] += 1
            function = next((fn for fn in question["function"] if fn["name"] == name), None)
            if function:
                matched.append((gold, prediction[chosen], function))
        for gold, pred, function in matched:
            name = next(iter(gold))
            expected_args = gold[name]
            pred_args = pred[name]
            details = function["parameters"]["properties"]
            required = set(function["parameters"].get("required", []))
            schema_valid = required.issubset(pred_args) and set(pred_args).issubset(details)
            if schema_valid:
                schema_valid = all(type_ok(value, details[key]) for key, value in pred_args.items())
            counters["matched_calls"] += 1
            counters["argument_schema_valid"] += int(schema_valid)
            for key, allowed in expected_args.items():
                if key not in pred_args:
                    counters["expected_arguments"] += 1
                    continue
                counters["expected_arguments"] += 1
                if pred_args[key] in allowed:
                    counters["argument_value_correct"] += 1
    def ratio(num: str, denom: str) -> float:
        return counters[num] / counters[denom] if counters[denom] else 0.0
    return {
        "category": category,
        **dict(counters),
        "function_selection_accuracy": ratio("function_selected", "expected_calls"),
        "argument_schema_validity": ratio("argument_schema_valid", "matched_calls"),
        "argument_value_accuracy": ratio("argument_value_correct", "expected_arguments"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, help="BFCL root containing result/qwen3-4b/non_live/")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.result_root)
    rows = []
    for category in OFFLINE:
        path = root / "result" / "qwen3-4b" / "non_live" / f"BFCL_v4_{category}_result.json"
        rows.append(summarize_category(category, path))
    output = Path(args.output) if args.output else root / "score" / "stage_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
