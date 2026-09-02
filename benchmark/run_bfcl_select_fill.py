"""BFCL select-then-fill evaluation for STAGE-style JSON generation.

The planner selects a multiset of functions for atomic requests.  A second
pass fills arguments for one routed function at a time, so call-set planning
and schema-grounded value construction can be measured separately.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.vllm_inference import build_chat_prompts, generate_texts, load_vllm_model
from run_bfcl_stage_prompt import convert_parameter_schema


CATEGORIES = ("simple_python", "multiple", "parallel")
SYSTEM_PROMPT = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def clean_output(text: str) -> str:
    text = text.split("</think>", 1)[-1].strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def function_names(calls: list[dict[str, Any]]) -> Counter[str]:
    return Counter(next(iter(call)) for call in calls if isinstance(call, dict) and call)


def plan_schema(functions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["plan"],
        "properties": {
            "plan": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["request", "function"],
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "One atomic sub-request copied or minimally paraphrased from the user message",
                        },
                        "function": {"type": "string", "enum": [f["name"] for f in functions]},
                    },
                },
            }
        },
    }


def fill_schema(function: dict[str, Any]) -> dict[str, Any]:
    parameters = convert_parameter_schema(function.get("parameters", {}))
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    parameters.setdefault("additionalProperties", False)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "arguments"],
        "properties": {"name": {"const": function["name"]}, "arguments": parameters},
    }


def planner_prompt(question: str, functions: list[dict[str, Any]], schema: dict[str, Any]) -> str:
    function_lines = "\n".join(
        f"- {function['name']}: {function.get('description') or 'No description provided.'}"
        for function in functions
    )
    return (
        "Decompose the request into atomic sub-requests and assign exactly one available function to each. "
        "Return JSON conforming to the schema.\n\n"
        f"=== Report ===\n{question}\n\n"
        f"=== Available functions ===\n{function_lines}\n\n"
        f"=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def filler_prompt(question: str, subrequest: str, schema: dict[str, Any]) -> str:
    return (
        "Construct the one routed tool call as JSON that conforms to the schema.\n\n"
        f"=== Report ===\n{question}\n\n"
        f"=== Sub-request ===\n{subrequest}\n\n"
        f"=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def parse_plan(text: str, valid_names: set[str]) -> list[dict[str, str]]:
    try:
        obj = json.loads(clean_output(text))
    except (TypeError, json.JSONDecodeError):
        return []
    plan = obj.get("plan") if isinstance(obj, dict) else None
    if not isinstance(plan, list):
        return []
    return [
        {"request": entry["request"], "function": entry["function"]}
        for entry in plan
        if isinstance(entry, dict)
        and isinstance(entry.get("request"), str)
        and entry.get("function") in valid_names
    ]


def parse_call(text: str, expected_name: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(clean_output(text))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict) or obj.get("name") != expected_name or not isinstance(obj.get("arguments"), dict):
        return None
    return {"name": expected_name, "arguments": obj["arguments"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BFCL select-then-fill with a local STAGE-compatible model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--oracle-plan", action="store_true", help="Use gold function multiplicities and evaluate only filling.")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=3100)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def generate_batches(engine, prompts: list[str], args: argparse.Namespace) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = build_chat_prompts(engine.tokenizer, SYSTEM_PROMPT, prompts[start : start + args.batch_size])
        outputs.extend(
            generate_texts(
                engine,
                batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
        )
    return outputs


def main() -> None:
    args = parse_args()
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH
    from jsonschema import Draft202012Validator

    prepared: dict[str, list[dict[str, Any]]] = {}
    for category in args.categories:
        answers = {row["id"]: row["ground_truth"] for row in read_jsonl(POSSIBLE_ANSWER_PATH / f"BFCL_v4_{category}.json")}
        rows = []
        for item in read_jsonl(PROMPT_PATH / f"BFCL_v4_{category}.json"):
            functions = item["function"]
            schema = plan_schema(functions)
            Draft202012Validator.check_schema(schema)
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"][0][-1]["content"],
                    "functions": functions,
                    "gold": answers[item["id"]],
                    "plan_schema": schema,
                }
            )
        prepared[category] = rows[: args.limit] if args.limit else rows

    engine = load_vllm_model(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    output_base = ROOT / "outputs" / "bfcl" / args.run_name
    result_root = output_base / "result" / "qwen3-4b" / "non_live"
    result_root.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, list[dict[str, Any]]] = {}

    for category, rows in prepared.items():
        if args.oracle_plan:
            plans = [
                [
                    {"request": row["question"], "function": name}
                    for name, count in function_names(row["gold"]).items()
                    for _ in range(count)
                ]
                for row in rows
            ]
            raw_plans = [None] * len(rows)
        else:
            raw_plans = generate_batches(
                engine,
                [planner_prompt(row["question"], row["functions"], row["plan_schema"]) for row in rows],
                args,
            )
            plans = [parse_plan(raw, {f["name"] for f in row["functions"]}) for raw, row in zip(raw_plans, rows)]

        fill_jobs: list[tuple[int, str, str, dict[str, Any]]] = []
        for row_index, (row, plan) in enumerate(zip(rows, plans)):
            by_name = {function["name"]: function for function in row["functions"]}
            for entry in plan:
                function = by_name[entry["function"]]
                schema = fill_schema(function)
                Draft202012Validator.check_schema(schema)
                fill_jobs.append((row_index, entry["function"], entry["request"], schema))

        raw_fills = generate_batches(
            engine,
            [filler_prompt(rows[index]["question"], request, schema) for index, _, request, schema in fill_jobs],
            args,
        ) if fill_jobs else []
        calls_by_row: list[list[dict[str, Any]]] = [[] for _ in rows]
        for (row_index, name, _, _), raw in zip(fill_jobs, raw_fills):
            call = parse_call(raw, name)
            if call is not None:
                calls_by_row[row_index].append(call)

        records = [{"id": row["id"], "result": json.dumps({"calls": calls})} for row, calls in zip(rows, calls_by_row)]
        output_path = result_root / f"BFCL_v4_{category}_result.json"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        category_diagnostics = []
        for index, (row, plan) in enumerate(zip(rows, plans)):
            gold = function_names(row["gold"])
            predicted = Counter(entry["function"] for entry in plan)
            category_diagnostics.append(
                {
                    "id": row["id"],
                    "gold_function_multiset": dict(gold),
                    "predicted_function_multiset": dict(predicted),
                    "call_set_exact": predicted == gold,
                    "plan_length": len(plan),
                    "filled_calls": len(calls_by_row[index]),
                    "raw_plan": raw_plans[index],
                }
            )
        diagnostics[category] = category_diagnostics
        exact = sum(item["call_set_exact"] for item in category_diagnostics)
        print(f"{category}: plan call-set exact {exact}/{len(rows)}; filled calls {sum(len(calls) for calls in calls_by_row)}", flush=True)

    (output_base / "plan_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"BFCL select-fill artifacts: {output_base}")


if __name__ == "__main__":
    main()
