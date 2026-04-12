"""
학습된 로컬 모델로 user_prompt → JSON 추론을 수행합니다.

사용법:
    python src/infer.py                          # data/user_prompt/ 전체 처리
    python src/infer.py --input data/user_prompt/data1.txt
    python src/infer.py --model models/qwen3-0.6b-finetuned
    python src/infer.py --batch-size 32          # 배치 크기 조정
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()
SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "json_SYSTEM_prompt.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model(model_path: str | Path, tokenizer_path: str | Path | None = None):
    model_path = str(model_path)
    tokenizer_src = str(tokenizer_path) if tokenizer_path else model_path
    print(f"모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 배치 생성 시 left padding 필요

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"모델 로드 완료. (device: {device})")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------

def build_prompt(user_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_json_from_output(text: str) -> Any | None:
    parts = re.split(r"</think>", text, maxsplit=1)
    content = parts[-1].strip()

    m = re.search(r"```json\s*([\s\S]+?)\s*```", content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m2 = re.search(r"(\{[\s\S]*\})", content)
    if m2:
        try:
            return json.loads(m2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def run_batch_inference(
    model, tokenizer, user_texts: list[str], max_new_tokens: int = 4096
) -> list[dict]:
    """
    여러 user_prompt를 배치로 추론합니다.
    """
    prompts = [build_prompt(t, tokenizer) for t in user_texts]
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
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for output in outputs:
        raw_output = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        results.append({
            "raw_output": raw_output,
            "json_obj": extract_json_from_output(raw_output),
        })
    return results


# ---------------------------------------------------------------------------
# 파일 처리
# ---------------------------------------------------------------------------

def process_batch(
    file_paths: list[Path],
    model,
    tokenizer,
    output_dir: Path,
    max_new_tokens: int = 4096,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir.parent / "json_infer_raw"

    user_texts = [p.read_text(encoding="utf-8") for p in file_paths]
    results = run_batch_inference(model, tokenizer, user_texts, max_new_tokens=max_new_tokens)

    for file_path, result in zip(file_paths, results):
        stem = file_path.stem
        if result["json_obj"] is not None:
            (output_dir / f"{stem}.json").write_text(
                json.dumps(result["json_obj"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  [OK] {stem}")
        else:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{stem}.txt").write_text(result["raw_output"], encoding="utf-8")
            print(f"  [WARN] {stem}: JSON 파싱 실패. raw 저장")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="로컬 모델로 user_prompt → JSON 배치 추론")
    parser.add_argument("--model", default="models/qwen3-0.6b-finetuned", help="모델 경로 또는 HF repo ID")
    parser.add_argument("--tokenizer", default=None, help="토크나이저 경로 또는 HF repo ID (미지정 시 --model과 동일)")
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/json_infer", help="출력 디렉토리")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기 (기본: 32)")
    parser.add_argument("--test-only", action="store_true", help="data/test_stems.txt에 있는 파일만 추론")
    args = parser.parse_args()

    _model_arg = Path(args.model)
    if _model_arg.is_absolute() or _model_arg.exists():
        model_path = _model_arg
    else:
        local_path = PROJECT_ROOT / args.model
        model_path = local_path if local_path.exists() else args.model
    output_dir = PROJECT_ROOT / args.output
    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "user_prompt"

    model, tokenizer = load_model(model_path, tokenizer_path=args.tokenizer)

    if input_path.is_file():
        all_files = [input_path]
    elif input_path.is_dir():
        all_files = sorted(input_path.glob("*.txt"))
    else:
        print(f"[ERROR] 경로를 찾을 수 없음: {input_path}")
        sys.exit(1)

    # test_stems.txt 필터링
    if args.test_only:
        test_stems_path = PROJECT_ROOT / "data" / "test_stems.txt"
        if not test_stems_path.exists():
            print(f"[ERROR] test_stems.txt 없음: {test_stems_path}")
            sys.exit(1)
        test_stems = set(test_stems_path.read_text(encoding="utf-8").splitlines())
        all_files = [f for f in all_files if f.stem in test_stems]
        print(f"[TEST] test_stems.txt 기준 {len(all_files)}개 파일 필터링")

    # 이미 처리된 파일 스킵
    files = [f for f in all_files if not (output_dir / f"{f.stem}.json").exists()]
    skipped = len(all_files) - len(files)
    if skipped:
        print(f"[SKIP] 이미 처리된 {skipped}개 파일 건너뜀")

    print(f"\n총 {len(files)}개 파일 처리 시작 (batch_size={args.batch_size})\n")

    for i in range(0, len(files), args.batch_size):
        batch = files[i:i + args.batch_size]
        print(f"[{i + 1}~{min(i + args.batch_size, len(files))}/{len(files)}] 처리 중...")
        process_batch(batch, model, tokenizer, output_dir, max_new_tokens=args.max_new_tokens)

    print("\n모든 파일 처리 완료.")


if __name__ == "__main__":
    main()
