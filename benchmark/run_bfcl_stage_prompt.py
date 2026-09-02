"""Evaluate BFCL offline tasks using STAGE's schema-grounded JSON protocol.

The official BFCL Python-call prompt is deliberately not used here: this
experiment asks whether the STAGE model can compose tool-call arguments when
the calls themselves are represented by the JSON Schema it was trained on.
Results retain BFCL's directory layout so ``rescore_bfcl_json.py`` can apply
the unchanged official AST checker after decoding the JSON representation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.vllm_inference import build_chat_prompts, generate_texts, load_vllm_model


CATEGORIES = ("simple_python", "multiple", "parallel")
SYSTEM_PROMPT = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")


def convert_parameter_schema(schema: Any) -> dict[str, Any]:
    """Convert BFCL's Python-oriented parameter schema into JSON Schema."""
    if not isinstance(schema, dict):
        return {}
    type_map = {"dict": "object", "float": "number", "tuple": "array", "any": None}
    allowed = {"type", "properties", "required", "description", "enum", "items", "additionalProperties", "minItems", "maxItems"}
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "type":
            mapped_type = type_map.get(value, value)
            # BFCL's ``any`` means unconstrained JSON; JSON Schema expresses
            # that by omitting ``type``, not by writing an invalid null type.
            if mapped_type is not None:
                result[key] = mapped_type
        elif key == "properties" and isinstance(value, dict):
            result[key] = {name: convert_parameter_schema(child) for name, child in value.items()}
        elif key == "items":
            result[key] = convert_parameter_schema(value)
        else:
            result[key] = value
    if result.get("type") == "object":
        result.setdefault("additionalProperties", False)
    return result


def bfcl_function_to_call_schema(functions: list[dict[str, Any]]) -> dict[str, Any]:
    variants = []
    for function in functions:
        name = function["name"]
        parameters = convert_parameter_schema(function.get("parameters", {}))
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("additionalProperties", False)
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "arguments"],
                "properties": {"name": {"const": name}, "arguments": parameters},
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["calls"],
        "properties": {"calls": {"type": "array", "minItems": 1, "items": {"oneOf": variants}}},
    }


ONE_SHOT_EXAMPLE = """Example (select only the function needed; do not call every listed function):
Available functions: get_weather(city), book_flight(origin, destination), send_email(to, subject).
Request: What is the weather in Seoul today?
Answer: {"calls": [{"name": "get_weather", "arguments": {"city": "Seoul"}}]}

"""


def user_prompt(
    question: str,
    functions: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    include_function_dump: bool,
    one_shot: bool,
) -> str:
    parts = []
    if one_shot:
        parts.append(ONE_SHOT_EXAMPLE.rstrip())
    parts.extend(
        [
            "Extract the tool calls needed to fulfil the request below as JSON that conforms to the schema.",
            f"=== Report ===\n{question}",
        ]
    )
    if include_function_dump:
        parts.append(f"=== Available functions ===\n{json.dumps(functions, ensure_ascii=False, indent=2)}")
    parts.append(f"=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}")
    return "\n\n".join(parts)


def clean_output(text: str) -> str:
    text = text.split("</think>", 1)[-1].strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def generate_guided(engine, prompts: list[str], schemas: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    params = [
        SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
        for schema in schemas
    ]
    outputs = engine.llm.generate(prompts, params, lora_request=engine.lora_request, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BFCL with the STAGE JSON-schema prompt.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--guided-json", action="store_true")
    parser.add_argument(
        "--no-function-dump",
        action="store_true",
        help="Omit the redundant full function JSON; the call schema remains available.",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Prefix a fixed one-of-three-functions, one-call selection example.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=3100)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from bfcl_eval.constants.eval_config import PROMPT_PATH
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit(f"Missing BFCL/jsonschema dependency: {exc}") from exc

    prepared: dict[str, list[dict[str, Any]]] = {}
    for category in args.categories:
        rows = []
        for item in read_jsonl(PROMPT_PATH / f"BFCL_v4_{category}.json"):
            functions = item["function"]
            schema = bfcl_function_to_call_schema(functions)
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as exc:
                rows.append({"id": item["id"], "skip_reason": f"schema_invalid: {exc}"})
                continue
            question = item["question"][0][-1]["content"]
            rows.append(
                {
                    "id": item["id"],
                    "prompt": user_prompt(
                        question,
                        functions,
                        schema,
                        include_function_dump=not args.no_function_dump,
                        one_shot=args.one_shot,
                    ),
                    "schema": schema,
                }
            )
        prepared[category] = rows[: args.limit] if args.limit else rows

    engine = load_vllm_model(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        guided_decoding_backend="xgrammar" if args.guided_json else None,
    )
    output_root = ROOT / "outputs" / "bfcl" / args.run_name / "result" / "qwen3-4b" / "non_live"
    output_root.mkdir(parents=True, exist_ok=True)
    for category, rows in prepared.items():
        output_path = output_root / f"BFCL_v4_{category}_result.json"
        completed = {row["id"] for row in read_jsonl(output_path)} if output_path.exists() else set()
        pending = [row for row in rows if row["id"] not in completed]
        saved = read_jsonl(output_path) if output_path.exists() else []
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            usable = [row for row in batch if "prompt" in row]
            raw_outputs: list[str] = []
            if usable:
                prompts = build_chat_prompts(engine.tokenizer, SYSTEM_PROMPT, [row["prompt"] for row in usable])
                raw_outputs = (
                    generate_guided(engine, prompts, [row["schema"] for row in usable], args)
                    if args.guided_json
                    else generate_texts(engine, prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed)
                )
            output_iter = iter(raw_outputs)
            for row in batch:
                record = {"id": row["id"], "result": clean_output(next(output_iter))} if "prompt" in row else {"id": row["id"], "result": "", "skip_reason": row["skip_reason"]}
                saved.append(record)
            with output_path.open("w", encoding="utf-8") as handle:
                for record in saved:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{category}: {len(saved)}/{len(rows)}", flush=True)
    print(f"BFCL STAGE-prompt artifacts: {output_root.parents[2]}")


if __name__ == "__main__":
    main()
