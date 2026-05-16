"""
DPO (Direct Preference Optimization) 데이터 생성 스크립트

각 user_prompt에 대해:
  - chosen: data/json/ 의 gold JSON object
  - rejected: 현재 SFT 모델이 생성했지만 gold schema 검증에 실패한 출력

검증 기준:
  - rejected 후보: user_prompt에 포함된 gold JSON Schema를 만족하지 못하는 JSON 출력
  - gold schema가 없는 프롬프트는 스킵

사용법:
    python src/generate_dpo_data.py --model models/qwen3-0.6b-finetuned
    python src/generate_dpo_data.py --model models/qwen3-0.6b-finetuned --num-samples 8 --max-prompts 2000
    python src/generate_dpo_data.py --model models/qwen3-0.6b-finetuned --split train
    python src/generate_dpo_data.py --model models/qwen3-0.6b-finetuned --num-shards 2 --shard-index 0

출력 포맷 (LLaMA-Factory DPO sharegpt):
    {
      "conversations": [{"from": "system", ...}, {"from": "human", ...}],
      "chosen": {"from": "gpt", "value": "..."},
      "rejected": {"from": "gpt", "value": "..."}
    }

LLaMA-Factory 등록 방법:
    1. 출력 JSONL을 /LLaMA-Factory/data/ 에 복사
    2. /LLaMA-Factory/data/dataset_info.json 에 아래 항목 추가:
       "sunny_dpo": {
         "file_name": "sunny_dpo.jsonl",
         "formatting": "sharegpt",
         "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"}
       }
    3. yaml의 dataset: sunny_dpo 로 변경
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
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root
from utils.parsing_answer import _extract_json_from_chunk

PROJECT_ROOT = find_project_root()
SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "json_SYSTEM_prompt.txt").read_text(encoding="utf-8")

JSON_DIR = PROJECT_ROOT / "data" / "json"
SCHEMA_DIR = PROJECT_ROOT / "data" / "json_schema"


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

    subfolder_kwargs = {"subfolder": subfolder} if subfolder else {}
    try:
        config = AutoConfig.from_pretrained(repo_id, **subfolder_kwargs, trust_remote_code=True)
    except Exception:
        config = AutoConfig.from_pretrained(tokenizer_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        **subfolder_kwargs,
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


def build_chosen_response(stem: str) -> str | None:
    """
    data/json/{stem}.json + data/json_schema/{stem}.json 을 읽어
    모델이 출력해야 할 gold JSON 문자열을 구성합니다.
    """
    json_path = JSON_DIR / f"{stem}.json"
    schema_path = SCHEMA_DIR / f"{stem}.json"
    if not json_path.exists() or not schema_path.exists():
        return None
    try:
        json_obj = json.loads(json_path.read_text(encoding="utf-8"))
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
        # gold JSON이 실제로 gold schema를 만족하는지 확인
        jsonschema.validate(instance=json_obj, schema=schema_obj)
    except Exception:
        return None

    return json.dumps(json_obj, ensure_ascii=False, indent=2)


def is_valid_against_schema(raw_output: str, gold_schema: dict) -> tuple[bool, str]:
    """
    모델 출력이 gold_schema를 만족하는지 확인합니다.
    반환: (통과 여부, think 블록 제거된 출력)
    """
    clean = strip_think_block(raw_output)
    try:
        json_obj = _extract_json_from_chunk(clean)
    except (ValueError, Exception):
        return False, clean

    try:
        jsonschema.validate(instance=json_obj, schema=gold_schema)
        return True, clean
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return False, clean


# ---------------------------------------------------------------------------
# 배치 생성
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
    반환: outputs[i] = [sample_0, ..., sample_{num_samples-1}]
    """
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
        max_length=4096,
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
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    grouped: list[list[str]] = []
    for i in range(len(user_texts)):
        grouped.append(decoded[i * num_samples : (i + 1) * num_samples])
    return grouped


# ---------------------------------------------------------------------------
# 메인 처리
# ---------------------------------------------------------------------------

def build_dpo_entry(user_text: str, chosen: str, rejected: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_text},
        ],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
    }


def main():
    parser = argparse.ArgumentParser(description="DPO 데이터 생성 (chosen=gold, rejected=model 실패)")
    parser.add_argument("--model", default="models/qwen3-0.6b-finetuned")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/dpo/sunny_dpo.jsonl", help="출력 JSONL 경로")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train", help="data/test_stems.txt 기준 split (기본: train)")
    parser.add_argument("--num-samples", type=int, default=8, help="프롬프트당 생성 샘플 수 (기본: 8)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2, help="동시에 처리할 프롬프트 수 (기본: 2)")
    parser.add_argument("--max-prompts", type=int, default=None, help="처리할 최대 프롬프트 수 (기본: 전체)")
    parser.add_argument("--max-pairs-per-prompt", type=int, default=3, help="프롬프트당 최대 DPO 쌍 수 (기본: 3)")
    parser.add_argument("--num-shards", type=int, default=1, help="전체 파일을 나눌 shard 수 (기본: 1)")
    parser.add_argument("--shard-index", type=int, default=0, help="처리할 shard index, 0부터 시작 (기본: 0)")
    args = parser.parse_args()
    if args.num_shards < 1:
        print("[ERROR] --num-shards 는 1 이상이어야 합니다.")
        sys.exit(1)
    if not 0 <= args.shard_index < args.num_shards:
        print("[ERROR] --shard-index 는 0 이상 --num-shards 미만이어야 합니다.")
        sys.exit(1)

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

    if args.split != "all":
        test_stems_path = PROJECT_ROOT / "data" / "test_stems.txt"
        if not test_stems_path.exists():
            print(f"[ERROR] split={args.split} 이지만 test_stems.txt 없음: {test_stems_path}")
            print("        먼저 python src/test/make_test_split.py 를 실행하거나 --split all 을 명시하세요.")
            sys.exit(1)
        test_stems = set(test_stems_path.read_text(encoding="utf-8").splitlines())
        if args.split == "test":
            all_files = [f for f in all_files if f.stem in test_stems]
        else:
            all_files = [f for f in all_files if f.stem not in test_stems]
        print(f"[SPLIT] {args.split}: test_stems.txt 기준 {len(all_files)}개 파일 선택")

    # gold 데이터(json + schema)가 있는 파일만 대상
    all_files = [f for f in all_files if (JSON_DIR / f"{f.stem}.json").exists()]

    if args.max_prompts:
        all_files = all_files[:args.max_prompts]

    if args.num_shards > 1:
        before_shard = len(all_files)
        all_files = [
            f for idx, f in enumerate(all_files)
            if idx % args.num_shards == args.shard_index
        ]
        print(f"[SHARD] {args.shard_index}/{args.num_shards}: {before_shard}개 중 {len(all_files)}개 파일 선택")

    # 이미 처리된 stem 스킵
    saved_stems: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                if "_stem" in entry:
                    saved_stems.add(entry["_stem"])
            except json.JSONDecodeError:
                pass

    files = [f for f in all_files if f.stem not in saved_stems]
    print(f"[SKIP] 이미 처리된 {len(all_files) - len(files)}개 건너뜀")
    print(f"총 {len(files)}개 프롬프트 처리 시작 (num_samples={args.num_samples})\n")

    model, tokenizer = load_model(model_path, tokenizer_path=args.tokenizer)

    total_pairs = 0
    total_no_failure = 0
    total_skipped = 0
    total_batches = (len(files) + args.batch_size - 1) // args.batch_size

    with output_path.open("a", encoding="utf-8") as fout:
        progress = tqdm(
            range(0, len(files), args.batch_size),
            total=total_batches,
            desc="DPO 생성",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_start in progress:
            batch_files = files[batch_start : batch_start + args.batch_size]
            user_texts = [f.read_text(encoding="utf-8") for f in batch_files]
            gold_schemas = [extract_schema_from_user_prompt(t) for t in user_texts]

            progress.set_postfix(
                range=f"{batch_start + 1}-{min(batch_start + args.batch_size, len(files))}/{len(files)}",
                pairs=total_pairs,
                no_rejected=total_no_failure,
                skipped=total_skipped,
            )
            all_samples = generate_samples_batch(
                model, tokenizer, user_texts,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            for f, user_text, schema, samples in zip(batch_files, user_texts, gold_schemas, all_samples):
                if schema is None:
                    total_skipped += 1
                    tqdm.write(f"  [SKIP] {f.stem}: user_prompt에서 JSON Schema 추출 실패")
                    fout.write(json.dumps({"_stem": f.stem, "_skipped": True}) + "\n")
                    fout.flush()
                    continue

                chosen = build_chosen_response(f.stem)
                if chosen is None:
                    total_skipped += 1
                    tqdm.write(f"  [SKIP] {f.stem}: gold JSON/Schema 로드 실패 (스키마 불일치 포함)")
                    fout.write(json.dumps({"_stem": f.stem, "_skipped": True}) + "\n")
                    fout.flush()
                    continue

                # 검증 실패한 샘플만 rejected 후보로
                failures: list[str] = []
                for sample in samples:
                    ok, clean = is_valid_against_schema(sample, schema)
                    if not ok:
                        failures.append(clean)

                if not failures:
                    total_no_failure += 1
                    tqdm.write(f"  [--] {f.stem}: 모든 샘플 통과 (rejected 없음, 스킵)")
                    fout.write(json.dumps({"_stem": f.stem, "_skipped": True}) + "\n")
                    fout.flush()
                    continue

                # 최대 max_pairs_per_prompt 개 쌍 저장
                pairs_written = 0
                for rejected in failures[:args.max_pairs_per_prompt]:
                    entry = build_dpo_entry(user_text, chosen, rejected)
                    entry["_stem"] = f.stem
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    pairs_written += 1
                fout.flush()

                total_pairs += pairs_written
                tqdm.write(f"  [OK] {f.stem}: {pairs_written}쌍 저장 ({len(failures)} failures 중)")
            progress.set_postfix(
                range=f"{batch_start + 1}-{min(batch_start + args.batch_size, len(files))}/{len(files)}",
                pairs=total_pairs,
                no_rejected=total_no_failure,
                skipped=total_skipped,
            )

    print(f"\n완료. 총 DPO 쌍: {total_pairs}, rejected 없어 스킵: {total_no_failure}, 기타 스킵: {total_skipped}")
    print(f"출력: {output_path}")
    print(f"\n[LLaMA-Factory 등록]")
    print(f'  파일을 /LLaMA-Factory/data/sunny_dpo.jsonl 에 복사 후')
    print(f'  dataset_info.json 에 추가:')
    print(json.dumps({
        "sunny_dpo": {
            "file_name": "sunny_dpo.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
