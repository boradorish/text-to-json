from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))

from utils.prompt_loader import find_project_root
from utils.parsing_answer import _extract_json_from_chunk
from utils.vllm_inference import build_chat_prompts


PROJECT_ROOT = find_project_root()

SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(
    encoding="utf-8"
)

DATA_ROOT = PROJECT_ROOT / "src" / "data"

USER_PROMPT_DIR = DATA_ROOT / "user_prompt"
JSON_DIR = DATA_ROOT / "json"
SCHEMA_DIR = DATA_ROOT / "json_schema"


def resolve_path(path_text: str):
    path = Path(path_text)

    if path.is_absolute() or path.exists():
        return path

    project_path = PROJECT_ROOT / path_text

    if project_path.exists():
        return project_path

    return path_text


def extract_schema_from_user_prompt(user_text: str) -> dict | None:
    match = re.search(r"=== JSON Schema ===\s*([\s\S]+)$", user_text, re.IGNORECASE)

    if not match:
        return None

    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def strip_think_block(text: str) -> str:
    parts = re.split(r"</think>", text, maxsplit=1)
    return parts[-1].strip()


def build_chosen_response(stem: str) -> str | None:
    json_path = JSON_DIR / f"{stem}.json"
    schema_path = SCHEMA_DIR / f"{stem}.json"

    if not json_path.exists() or not schema_path.exists():
        return None

    try:
        json_obj = json.loads(json_path.read_text(encoding="utf-8"))
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=json_obj, schema=schema_obj)
    except Exception:
        return None

    return json.dumps(json_obj, ensure_ascii=False, indent=2)


def is_valid_against_schema(raw_output: str, gold_schema: dict) -> tuple[bool, str]:
    clean = strip_think_block(raw_output)

    try:
        json_obj = _extract_json_from_chunk(clean)
    except Exception:
        return False, clean

    try:
        jsonschema.validate(instance=json_obj, schema=gold_schema)
        return True, clean
    except Exception:
        return False, clean


def build_dpo_entry(user_text: str, chosen: str, rejected: str, stem: str) -> dict:
    return {
        "_stem": stem,
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_text},
        ],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
    }


def load_items(input_path: Path):
    print(f"[DEBUG] cwd: {Path.cwd()}")
    print(f"[DEBUG] input_path: {input_path}")
    print(f"[DEBUG] absolute path: {input_path.resolve()}")
        
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.txt"))
    items = []
    skipped = 0

    for file in tqdm(files):
        user_text = file.read_text(encoding="utf-8")
        schema = extract_schema_from_user_prompt(user_text)
        chosen = build_chosen_response(file.stem)

            
        if schema is None or chosen is None:
            skipped += 1
            continue

        items.append({
            "stem": file.stem,
            "user_text": user_text,
            "schema": schema,
            "chosen": chosen,
        })
    return items, skipped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="saves/qwen3-0.6b/full/sft")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="../LLaMA-Factory/data/sunny_dpo.jsonl")

    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=4096)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)

    args = parser.parse_args()

    model_path = resolve_path(args.model)
    tokenizer_path = resolve_path(args.tokenizer) if args.tokenizer else model_path

    input_path = Path(args.input) if args.input else USER_PROMPT_DIR
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(input_path)
    items, skipped_before_generation = load_items(input_path)

    print(f"valid prompts: {len(items)}")
    print(f"skipped before generation: {skipped_before_generation}")

    if not items:
        print("no valid prompts found")
        return

    user_texts = [item["user_text"] for item in items]

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
    )

    prompts = build_chat_prompts(
        tokenizer,
        SYSTEM_PROMPT,
        user_texts,
    )

    llm_kwargs = {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    print("starting one vLLM offline generate call")
    outputs = llm.generate(prompts, sampling_params)
    print("generation done")

    total_pairs = 0
    total_no_bad_samples = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item, output in tqdm(
            zip(items, outputs),
            total=len(items),
            desc="validation and DPO pairing",
        ):
            bad_samples = []

            for completion in output.outputs:
                raw_text = completion.text
                is_valid, clean_text = is_valid_against_schema(
                    raw_text,
                    item["schema"],
                )

                if not is_valid:
                    bad_samples.append(clean_text)

            if not bad_samples:
                total_no_bad_samples += 1
                continue

            for rejected in bad_samples:
                entry = build_dpo_entry(
                    user_text=item["user_text"],
                    chosen=item["chosen"],
                    rejected=rejected,
                    stem=item["stem"],
                )

                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"done")
    print(f"dpo pairs written: {total_pairs}")
    print(f"prompts with no bad samples: {total_no_bad_samples}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()