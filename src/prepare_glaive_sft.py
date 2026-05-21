"""
glaiveai/glaive-function-calling-v2 -> LLaMA-Factory SFT 데이터 준비

<functioncall> 안에 있는 JSON만 gold assistant 응답으로 추출하고,
그 JSON 호출 부분을 제외한 나머지 대화는 report 형태의 user 입력으로 만듭니다.

출력 포맷: sharegpt jsonl
  {"conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "=== REPORT ===\n..."},
      {"from": "gpt", "value": "{\"name\": ..., \"arguments\": ...}"}
  ]}

사용법:
    python3 src/prepare_glaive_sft.py
    python3 src/prepare_glaive_sft.py --num-samples 2000 --output data/sft/glaive_sft.jsonl
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

PROJECT_ROOT = find_project_root()

DEFAULT_SYSTEM_PROMPT = (
    "You are a function-calling JSON generation assistant. "
    "Read the conversation report and output only the next function call as valid JSON. "
    "Do not include markdown, code fences, explanations, or XML-like tags."
)

USER_TEMPLATE = """Read the conversation report below and produce the JSON function call that should be made next.

=== REPORT ===
{report}

Return ONLY the valid JSON object."""


def parse_system(system_str: str) -> str:
    return re.sub(r"^SYSTEM:\s*", "", system_str, flags=re.IGNORECASE).strip()


def parse_chat(chat_str: str) -> list[dict[str, str]]:
    parts = re.split(r"\b(USER|ASSISTANT|FUNCTION RESPONSE):\s*", chat_str or "")
    turns = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i]
        content = parts[i + 1].replace("<|endoftext|>", "").strip()
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


def extract_functioncall_json(content: str) -> str | None:
    """<functioncall> 내부 JSON을 파싱 검증하고 정규화해서 반환."""
    content = (content or "").replace("<|endoftext|>", "").strip()
    if "<functioncall>" not in content:
        return None

    after_tag = content.split("<functioncall>", 1)[1]
    if "</functioncall>" in after_tag:
        after_tag = after_tag.split("</functioncall>", 1)[0]

    candidate = _extract_balanced_json(after_tag)
    if candidate is None:
        return None

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None
    return json.dumps(obj, ensure_ascii=False)


def strip_functioncall(content: str) -> str:
    """report에 gold JSON이 새지 않도록 <functioncall> 블록을 제거."""
    content = (content or "").replace("<|endoftext|>", "").strip()
    tag_start = content.find("<functioncall>")
    if tag_start < 0:
        return content

    json_start = content.find("{", tag_start)
    if json_start < 0:
        return content[:tag_start].strip()

    json_text = _extract_balanced_json(content[json_start:])
    if json_text is None:
        return content[:tag_start].strip()

    remove_end = json_start + len(json_text)
    if content[remove_end:].lstrip().startswith("</functioncall>"):
        close_start = content.find("</functioncall>", remove_end)
        remove_end = close_start + len("</functioncall>")

    return (content[:tag_start] + content[remove_end:]).strip()


def format_report(system: str, turns: list[dict[str, str]]) -> str:
    lines = []
    if system:
        lines.extend(["SYSTEM:", system.strip(), ""])

    for turn in turns:
        role = turn["role"]
        content = turn["content"].strip()
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
    for i, turn in enumerate(turns):
        content = turn["content"]
        if i == gold_turn_index:
            content = strip_functioncall(content)
            if not content:
                continue
        report_turns.append({"role": turn["role"], "content": content})

    return format_report(system, report_turns)


def convert_item(item: dict[str, Any], system_prompt: str) -> list[dict[str, Any]]:
    system = parse_system(item.get("system", ""))
    turns = parse_chat(item.get("chat", ""))

    records = []
    for i, turn in enumerate(turns):
        if turn["role"] != "ASSISTANT":
            continue

        gold_json = extract_functioncall_json(turn["content"])
        if gold_json is None:
            continue

        report = format_report_without_gold_call(system, turns, i)
        if not report or not any(t["role"] == "USER" for t in turns):
            continue

        records.append({
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": USER_TEMPLATE.format(report=report)},
                {"from": "gpt", "value": gold_json},
            ]
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="glaive-function-calling-v2 → SFT 데이터 준비")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--output", default="data/sft/glaive_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    print("glaiveai/glaive-function-calling-v2 로드 중...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    ds = ds.shuffle(seed=args.seed)
    print(f"총 {len(ds)}개 로드 완료\n")

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="처리 중"):
            if written >= args.num_samples:
                break

            records = convert_item(item, args.system_prompt)
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
