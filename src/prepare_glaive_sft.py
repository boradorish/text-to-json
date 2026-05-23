"""
glaiveai/glaive-function-calling-v2 -> report + JSON Schema + gold JSON SFT 데이터 준비.

Glaive 원본은 system에 함수 정의/parameters schema가 있고, assistant의
<functioncall> 안에는 {"name": ..., "arguments": "...json string..."} 형태의
호출이 들어 있습니다. 이 스크립트는 호출된 함수의 parameters를 JSON Schema로
꺼내고 arguments를 실제 JSON 객체로 정규화해서 다음 형태로 학습 데이터를 만듭니다.

출력 포맷: sharegpt jsonl
  {"conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "Target function...\n=== REPORT ===\n...\n=== JSON SCHEMA ===\n..."},
      {"from": "gpt", "value": "{\"country\":\"United States\"}"}
  ]}

사용법:
    python3 src/prepare_glaive_sft.py
    python3 src/prepare_glaive_sft.py --num-samples 2000 --output data/sft/glaive_sft.jsonl
    python3 src/prepare_glaive_sft.py --gold-format call  # name + arguments 객체까지 출력
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
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

USER_TEMPLATE = """Extract the JSON object requested by the target function from the conversation report.

Target function: {function_name}
Function description: {function_description}

=== REPORT ===
{report}

=== JSON SCHEMA ===
{schema}

Return ONLY valid JSON matching the schema."""


def parse_system(system_str: str) -> str:
    return re.sub(r"^SYSTEM:\s*", "", system_str, flags=re.IGNORECASE).strip()


def parse_chat(chat_str: str) -> list[dict[str, str]]:
    parts = re.split(r"\b(USER|ASSISTANT|FUNCTION RESPONSE):\s*", chat_str or "")
    turns = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i]
        content = parts[i + 1].strip()
        if content:
            turns.append({"role": role, "content": content})
        i += 2
    return turns


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _iter_balanced_json(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        brace = text.find("{", start)
        if brace < 0:
            break
        chunk = _extract_balanced_json(text[brace:])
        if chunk is None:
            break
        chunks.append(chunk)
        start = brace + len(chunk)
    return chunks


def parse_function_definitions(system: str) -> dict[str, dict[str, Any]]:
    """Glaive system prompt에서 함수 정의 JSON 객체들을 추출합니다."""
    functions: dict[str, dict[str, Any]] = {}
    for chunk in _iter_balanced_json(system or ""):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        candidates = obj if isinstance(obj, list) else [obj]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name")
            params = candidate.get("parameters")
            if isinstance(name, str) and isinstance(params, dict):
                functions[name] = candidate
    return functions


def _functioncall_span(content: str) -> tuple[int, int, str] | None:
    """Return the <functioncall> span and payload up to its real delimiter."""
    tag = "<functioncall>"
    tag_start = (content or "").find(tag)
    if tag_start < 0:
        return None

    payload_start = tag_start + len(tag)
    delimiter_start = len(content)
    delimiter_end = delimiter_start
    for delimiter in ("</functioncall>", "<|endoftext|>"):
        found = content.find(delimiter, payload_start)
        if found >= 0 and found < delimiter_start:
            delimiter_start = found
            delimiter_end = found + len(delimiter)

    payload = content[payload_start:delimiter_start].strip()
    return tag_start, delimiter_end, payload


def extract_functioncall_obj(content: str) -> dict[str, Any] | None:
    """<functioncall> 내부 호출 객체를 파싱 검증하고 arguments를 객체로 정규화합니다."""
    span = _functioncall_span(content or "")
    if span is None:
        return None
    _, _, payload = span

    candidate = _extract_balanced_json(payload)
    if candidate is None:
        return None

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None
    except TypeError:
        return None

    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str):
        return None

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            try:
                args = ast.literal_eval(args)
            except (SyntaxError, ValueError):
                return None
    elif args is None:
        args = {}

    if not isinstance(args, dict):
        return None

    return {"name": name, "arguments": args}


def strip_functioncall(content: str) -> str:
    """report에 gold JSON이 새지 않도록 <functioncall> 블록을 제거."""
    content = content or ""
    span = _functioncall_span(content)
    if span is None:
        return content.replace("<|endoftext|>", "").strip()

    tag_start, remove_end, _ = span
    return (content[:tag_start] + content[remove_end:]).replace("<|endoftext|>", "").strip()


def _strip_raw_function_definitions(system: str) -> str:
    system = parse_system(system)
    if "with access to the following functions" not in system.lower():
        return system
    return "The assistant may use the provided target function when the conversation requires it."


def format_report(system: str, turns: list[dict[str, str]]) -> str:
    lines = []
    if system:
        lines.extend(["SYSTEM SUMMARY:", _strip_raw_function_definitions(system).strip(), ""])

    for turn in turns:
        role = turn["role"]
        content = turn["content"].replace("<|endoftext|>", "").strip()
        if not content:
            continue
        lines.extend([f"{role}:", content, ""])

    return "\n".join(lines).strip()


def format_report_without_gold_call(
    system: str,
    turns: list[dict[str, str]],
    gold_turn_index: int,
) -> str:
    report_turns = []
    for i, turn in enumerate(turns[: gold_turn_index + 1]):
        content = turn["content"]
        if i == gold_turn_index:
            content = strip_functioncall(content)
            if not content:
                continue
        report_turns.append({"role": turn["role"], "content": content})

    return format_report(system, report_turns)


def _ensure_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    return normalized


def _infer_schema_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {k: _infer_schema_from_value(v) for k, v in value.items()},
            "required": list(value.keys()),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item_schema = _infer_schema_from_value(value[0]) if value else {}
        return {"type": "array", "items": item_schema}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return {"type": "null"}
    return {"type": "string"}


def _build_call_schema(function_name: str, arguments_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "const": function_name},
            "arguments": arguments_schema,
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def _json_dumps(obj: Any, pretty: bool) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def convert_item(item: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    system = parse_system(item.get("system", ""))
    turns = parse_chat(item.get("chat", ""))
    functions = parse_function_definitions(system)

    records = []
    for i, turn in enumerate(turns):
        if turn["role"] != "ASSISTANT":
            continue

        call = extract_functioncall_obj(turn["content"])
        if call is None:
            continue
        if not args.allow_empty_gold and not call["arguments"]:
            continue

        function_name = call["name"]
        function_def = functions.get(function_name, {})
        function_description = (function_def.get("description") or "Not specified.").strip()
        schema = function_def.get("parameters")
        if not isinstance(schema, dict):
            schema = _infer_schema_from_value(call["arguments"])
        schema = _ensure_schema(schema)

        gold_obj: dict[str, Any]
        prompt_schema = schema
        if args.gold_format == "call":
            gold_obj = {"name": function_name, "arguments": call["arguments"]}
            prompt_schema = _build_call_schema(function_name, schema)
        else:
            gold_obj = call["arguments"]

        report = format_report_without_gold_call(system, turns, i)
        if not report or not any(t["role"] == "USER" for t in turns):
            continue

        if len(report) > args.max_report_chars:
            report = report[: args.max_report_chars].rstrip() + "\n\n[TRUNCATED]"

        records.append({
            "conversations": [
                {"from": "system", "value": args.system_prompt},
                {"from": "human", "value": USER_TEMPLATE.format(
                    function_name=function_name,
                    function_description=function_description,
                    report=report,
                    schema=_json_dumps(prompt_schema, args.pretty),
                )},
                {"from": "gpt", "value": _json_dumps(gold_obj, args.pretty)},
            ]
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="glaive-function-calling-v2 → SFT 데이터 준비")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--output", default="data/sft/glaive_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="glaiveai/glaive-function-calling-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-report-chars", type=int, default=12000)
    parser.add_argument("--gold-format", choices=["args", "call"], default="args")
    parser.add_argument("--allow-empty-gold", action="store_true")
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    print(f"{args.dataset} 로드 중...")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.shuffle(seed=args.seed)
    print(f"총 {len(ds)}개 로드 완료\n")

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="처리 중"):
            if written >= args.num_samples:
                break

            records = convert_item(item, args)
            if not records:
                skipped += 1
                continue

            for record in records:
                if written >= args.num_samples:
                    break
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"\n완료.")
    print(f"  저장:  {written}개")
    print(f"  스킵:  {skipped}개")
    print(f"  출력: {output_path}")

    dataset_name = "glaive_sft"
    print(f"\n[LLaMA-Factory 데이터셋 등록]")
    print(f"  cp {output_path} ../LLaMA-Factory/data/{output_path.name}")
    print(f"  dataset_info.json 추가:")
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


if __name__ == "__main__":
    main()
