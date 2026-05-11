"""
JSONSchemaBench DPO 데이터 준비 스크립트 (Baseline)

Qwen3-4B-Instruct-2507를 사용해 JSONSchemaBench 스키마별 여러 출력을 생성하고
jsonschema 검증 통과 = chosen, 실패 = rejected 로 DPO 쌍을 만듭니다.

출력 포맷: LLaMA-Factory sharegpt DPO
  {
    "conversations": [{"from": "system", ...}, {"from": "human", ...}],
    "chosen":   {"from": "gpt", "value": "..."},
    "rejected": {"from": "gpt", "value": "..."}
  }

사용법:
    python src/prepare_jsonschemabench_dpo.py
    python src/prepare_jsonschemabench_dpo.py --model Qwen/Qwen3-4B-Instruct-2507 --num-pairs 2000
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import jsonschema
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()

SYSTEM_PROMPT = (
    "You are a JSON generation assistant. "
    "Given a JSON Schema, generate a valid JSON object that strictly conforms to it. "
    "Output ONLY the raw JSON object — no explanation, no markdown, no code fences."
)

USER_TEMPLATE = "Generate a valid JSON object conforming to the following JSON Schema:\n\n{schema}"

TRIVIAL_SCHEMAS = ({}, {"type": "object"}, {"type": "object", "properties": {}})


def strip_think(text: str) -> str:
    return re.split(r"</think>", text, maxsplit=1)[-1].strip()


def try_parse_json(text: str) -> object:
    text = text.strip()
    # 코드펜스 제거
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def is_valid(text: str, schema: dict) -> tuple[bool, str]:
    clean = strip_think(text)
    try:
        obj = try_parse_json(clean)
        jsonschema.validate(instance=obj, schema=schema)
        return True, clean
    except Exception:
        return False, clean


def load_model(model_id: str):
    print(f"모델 로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("모델 로드 완료.")
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    schema_strs: list[str],
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[str]]:
    """schema_strs[i] 에 대해 num_samples개 출력 생성. 반환: outputs[i] = [s0, s1, ...]"""
    repeated = [s for s in schema_strs for _ in range(num_samples)]
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(schema=s)},
        ]
        for s in repeated
    ]
    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for msgs in messages_list
    ]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    decoded = [
        tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in out
    ]
    del inputs, out
    gc.collect()
    torch.cuda.empty_cache()

    return [decoded[i * num_samples : (i + 1) * num_samples] for i in range(len(schema_strs))]


def main():
    parser = argparse.ArgumentParser(description="JSONSchemaBench → DPO baseline 데이터 준비")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-pairs", type=int, default=2000, help="목표 DPO 쌍 수 (기본: 2000)")
    parser.add_argument("--num-samples", type=int, default=4, help="스키마당 생성 샘플 수 (기본: 4)")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기 (기본: 4)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output",
        default="data/dpo/jsonschemabench_dpo_baseline.jsonl",
    )
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미 저장된 항목 스킵
    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                if "_unique_id" in entry:
                    done_ids.add(entry["_unique_id"])
            except Exception:
                pass
    written = len(done_ids)
    print(f"이미 저장된 쌍: {written}개")

    print(f"epfl-dlab/JSONSchemaBench ({args.split} split) 로드 중...")
    ds = load_dataset("epfl-dlab/JSONSchemaBench", split=args.split)
    print(f"총 {len(ds)}개 로드 완료\n")

    model, tokenizer = load_model(args.model)

    skipped_trivial = 0
    skipped_no_pair = 0

    items = [item for item in ds if item["unique_id"] not in done_ids]

    with output_path.open("a", encoding="utf-8") as fout:
        for batch_start in tqdm(range(0, len(items), args.batch_size), desc="배치 처리"):
            if written >= args.num_pairs:
                break

            batch = items[batch_start : batch_start + args.batch_size]
            schema_strs, schema_objs, unique_ids = [], [], []

            for item in batch:
                try:
                    schema_obj = json.loads(item["json_schema"])
                    if not isinstance(schema_obj, dict) or schema_obj in TRIVIAL_SCHEMAS:
                        skipped_trivial += 1
                        continue
                except Exception:
                    skipped_trivial += 1
                    continue
                schema_strs.append(item["json_schema"])
                schema_objs.append(schema_obj)
                unique_ids.append(item["unique_id"])

            if not schema_strs:
                continue

            all_outputs = generate_batch(
                model, tokenizer, schema_strs,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            for schema_str, schema_obj, uid, outputs in zip(
                schema_strs, schema_objs, unique_ids, all_outputs
            ):
                valids, invalids = [], []
                for raw in outputs:
                    ok, clean = is_valid(raw, schema_obj)
                    (valids if ok else invalids).append(clean)

                if not valids or not invalids:
                    skipped_no_pair += 1
                    continue

                entry = {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": USER_TEMPLATE.format(schema=schema_str)},
                    ],
                    "chosen":   {"from": "gpt", "value": valids[0]},
                    "rejected": {"from": "gpt", "value": invalids[0]},
                    "_unique_id": uid,
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

                if written >= args.num_pairs:
                    break

    print(f"\n완료.")
    print(f"  저장:                {written}개")
    print(f"  스킵 (trivial):      {skipped_trivial}개")
    print(f"  스킵 (쌍 없음):      {skipped_no_pair}개")
    print(f"  출력: {output_path}")

    dataset_name = "jsonschemabench_dpo_baseline"
    print(f"\n[LLaMA-Factory 데이터셋 등록]")
    print(f"  cp {output_path} /workspace/LLaMA-Factory/data/{output_path.name}")
    print(f"  dataset_info.json 추가:")
    print(json.dumps({
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
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
