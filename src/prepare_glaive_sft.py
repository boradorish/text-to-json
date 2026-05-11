"""
glaiveai/glaive-function-calling-v2 → LLaMA-Factory SFT 데이터 준비

출력 포맷: sharegpt
  {"conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."},
      {"from": "observation", "value": "..."},  # FUNCTION RESPONSE
      ...
  ]}

사용법:
    python src/prepare_glaive_sft.py
    python src/prepare_glaive_sft.py --num-samples 2000 --output data/sft/glaive_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()


def parse_system(system_str: str) -> str:
    return re.sub(r"^SYSTEM:\s*", "", system_str, flags=re.IGNORECASE).strip()


def parse_chat(chat_str: str) -> list[dict]:
    segments = [s.strip() for s in re.split(r"\n{2,}", chat_str) if s.strip()]
    turns = []
    for seg in segments:
        if seg.startswith("USER:"):
            content = seg[5:].strip()
            if content:
                turns.append({"from": "human", "value": content})
        elif seg.startswith("ASSISTANT:"):
            content = seg[10:].strip().replace("<|endoftext|>", "").strip()
            if content:
                turns.append({"from": "gpt", "value": content})
        elif seg.startswith("FUNCTION RESPONSE:"):
            content = seg[18:].strip()
            if content:
                turns.append({"from": "observation", "value": content})
    return turns


def is_valid_conversation(turns: list[dict]) -> bool:
    if not turns:
        return False
    # 반드시 human-gpt 쌍이 하나 이상 있어야 함
    roles = [t["from"] for t in turns]
    return "human" in roles and "gpt" in roles


def main():
    parser = argparse.ArgumentParser(description="glaive-function-calling-v2 → SFT 데이터 준비")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--output", default="data/sft/glaive_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

            system_str = parse_system(item.get("system", ""))
            turns = parse_chat(item.get("chat", ""))

            if not is_valid_conversation(turns):
                skipped += 1
                continue

            conversations = []
            if system_str:
                conversations.append({"from": "system", "value": system_str})
            conversations.extend(turns)

            record = {"conversations": conversations}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n완료.")
    print(f"  저장:  {written}개")
    print(f"  스킵:  {skipped}개")
    print(f"  출력: {output_path}")

    dataset_name = "glaive_sft"
    print(f"\n[LLaMA-Factory 데이터셋 등록]")
    print(f"  cp {output_path} /workspace/LLaMA-Factory/data/{output_path.name}")
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
