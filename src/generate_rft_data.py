"""
Rejection Sampling Fine-Tuning (RFT) 데이터 생성 스크립트

현재 SFT 모델로 각 user_prompt에 대해 N개 샘플을 생성하고,
스키마 검증을 통과한 샘플만 골라 LLaMA-Factory용 SFT 데이터셋으로 저장합니다.

검증 기준:
  - 모델 출력이 === JSON === / === JSON_SCHEMA === 포맷을 따름
  - 출력의 JSON이 user_prompt에 포함된 gold JSON Schema를 만족함

사용법:
    python src/generate_rft_data.py --model models/qwen3-0.6b-finetuned
    python src/generate_rft_data.py --model models/qwen3-0.6b-finetuned --num-samples 8 --max-prompts 2000

출력 포맷 (LLaMA-Factory sharegpt):
    {"conversations": [{"from": "system", ...}, {"from": "human", ...}, {"from": "gpt", ...}]}

LLaMA-Factory 등록 방법:
    1. 출력 JSONL을 /LLaMA-Factory/data/ 에 복사
    2. /LLaMA-Factory/data/dataset_info.json 에 아래 항목 추가:
       "sunny_rft": {
         "file_name": "sunny_rft.jsonl",
         "formatting": "sharegpt",
         "columns": {"messages": "conversations"}
       }
    3. yaml의 dataset: sunny_rft 로 변경
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root
from utils.parsing_answer import parse_json_and_schema

PROJECT_ROOT = find_project_root()
SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "json_SYSTEM_prompt.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 모델 로드 (infer.py와 동일)
# ---------------------------------------------------------------------------

def _parse_model_path(model_path: str) -> tuple[str, str | None]:
    parts = model_path.split("/")
    if len(parts) > 2 and not model_path.startswith("/"):
        return "/".join(parts[:2]), "/".join(parts[2:])
    return model_path, None


def load_model(model_path: str | Path, tokenizer_path: str | Path | None = None):
    model_path = str(model_path)
    tokenizer_src = str(tokenizer_path) if tokenizer_path else model_path
    repo_id, subfolder = _parse_model_path(model_path)
    print(f"모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        config = AutoConfig.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
    except Exception:
        config = AutoConfig.from_pretrained(tokenizer_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder=subfolder,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"모델 로드 완료. (device: {device})")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def extract_schema_from_user_prompt(user_text: str) -> dict | None:
    """user_prompt에서 gold JSON Schema를 추출합니다."""
    m = re.search(r"=== JSON Schema ===\s*([\s\S]+)$", user_text, re.IGNORECASE)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def strip_think_block(text: str) -> str:
    """<think>...</think> 블록을 제거합니다."""
    parts = re.split(r"</think>", text, maxsplit=1)
    return parts[-1].strip()


def validate_output_against_schema(raw_output: str, gold_schema: dict) -> tuple[bool, str]:
    """
    모델 출력을 파싱하고 gold_schema에 대해 검증합니다.
    반환: (통과 여부, 정제된 출력 텍스트)
    """
    clean = strip_think_block(raw_output)
    try:
        parsed = parse_json_and_schema(clean)
    except (ValueError, Exception):
        return False, clean

    try:
        jsonschema.validate(instance=parsed["json_obj"], schema=gold_schema)
        return True, clean
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return False, clean


# ---------------------------------------------------------------------------
# 배치 생성 (num_samples개를 한 번에)
# ---------------------------------------------------------------------------

def generate_samples_batch(
    model,
    tokenizer,
    user_texts: list[str],
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[str]]:
    """
    user_texts[i]에 대해 num_samples개 샘플을 생성합니다.
    반환: outputs[i] = [sample_0, sample_1, ..., sample_{num_samples-1}]
    """
    # 각 프롬프트를 num_samples번 반복해서 배치 구성
    repeated_texts = [t for t in user_texts for _ in range(num_samples)]

    messages_list = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t}]
        for t in repeated_texts
    ]
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = [
        tokenizer.decode(out[input_len:], skip_special_tokens=True)
        for out in outputs
    ]

    # num_samples 단위로 묶어서 반환
    grouped: list[list[str]] = []
    for i in range(len(user_texts)):
        grouped.append(decoded[i * num_samples : (i + 1) * num_samples])
    return grouped


# ---------------------------------------------------------------------------
# 메인 처리
# ---------------------------------------------------------------------------

def build_sharegpt_entry(user_text: str, assistant_text: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Rejection Sampling SFT 데이터 생성")
    parser.add_argument("--model", default="models/qwen3-0.6b-finetuned")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/rft/sunny_rft.jsonl", help="출력 JSONL 경로")
    parser.add_argument("--num-samples", type=int, default=4, help="프롬프트당 생성 샘플 수 (기본: 4)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4, help="동시에 처리할 프롬프트 수 (기본: 4)")
    parser.add_argument("--max-prompts", type=int, default=None, help="처리할 최대 프롬프트 수 (기본: 전체)")
    args = parser.parse_args()

    _model_arg = Path(args.model)
    model_path = _model_arg if (_model_arg.is_absolute() or _model_arg.exists()) else (
        PROJECT_ROOT / args.model if (PROJECT_ROOT / args.model).exists() else args.model
    )
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "user_prompt"

    if input_path.is_file():
        all_files = [input_path]
    else:
        all_files = sorted(input_path.glob("*.txt"))

    if args.max_prompts:
        all_files = all_files[:args.max_prompts]

    # 이미 저장된 항목 스킵 (stem 기준)
    saved_stems: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                # stem은 별도 필드로 저장
                if "_stem" in entry:
                    saved_stems.add(entry["_stem"])
            except json.JSONDecodeError:
                pass

    files = [f for f in all_files if f.stem not in saved_stems]
    print(f"[SKIP] 이미 처리된 {len(all_files) - len(files)}개 건너뜀")
    print(f"총 {len(files)}개 프롬프트 처리 시작 (num_samples={args.num_samples})\n")

    model, tokenizer = load_model(model_path, tokenizer_path=args.tokenizer)

    total_accepted = 0
    total_rejected = 0

    with output_path.open("a", encoding="utf-8") as fout:
        for batch_start in range(0, len(files), args.batch_size):
            batch_files = files[batch_start : batch_start + args.batch_size]
            user_texts = [f.read_text(encoding="utf-8") for f in batch_files]
            gold_schemas = [extract_schema_from_user_prompt(t) for t in user_texts]

            print(f"[{batch_start + 1}~{min(batch_start + args.batch_size, len(files))}/{len(files)}] 생성 중...")
            all_samples = generate_samples_batch(
                model, tokenizer, user_texts,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            for f, user_text, schema, samples in zip(batch_files, user_texts, gold_schemas, all_samples):
                if schema is None:
                    print(f"  [SKIP] {f.stem}: user_prompt에서 JSON Schema 추출 실패")
                    continue

                accepted = 0
                for sample in samples:
                    ok, clean_output = validate_output_against_schema(sample, schema)
                    if ok:
                        entry = build_sharegpt_entry(user_text, clean_output)
                        entry["_stem"] = f.stem  # 재실행 시 중복 방지용
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        accepted += 1

                total_accepted += accepted
                total_rejected += (args.num_samples - accepted)
                status = "✓" if accepted > 0 else "✗"
                print(f"  [{status}] {f.stem}: {accepted}/{args.num_samples} 통과")

    print(f"\n완료. 총 통과: {total_accepted}, 실패: {total_rejected}")
    print(f"출력: {output_path}")
    print(f"\n[LLaMA-Factory 등록]")
    print(f'  파일을 /LLaMA-Factory/data/sunny_rft.jsonl 에 복사 후')
    print(f'  dataset_info.json 에 추가:')
    print(json.dumps({
        "sunny_rft": {
            "file_name": "sunny_rft.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
