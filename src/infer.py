"""
학습된 로컬 모델로 user_prompt → JSON 추론을 수행합니다.

사용법:
    python src/infer.py                          # data/user_prompt/ 전체 처리
    python src/infer.py --input data/user_prompt/data1.txt
    python src/infer.py --model model/qwen3-0.6b-finetuned
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

def load_model(model_path: str | Path):
    model_path = str(model_path)
    print(f"모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
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
    """user_prompt 텍스트로 chat 프롬프트를 구성합니다."""
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
    """
    모델 출력 (<think>...</think>\n{json}) 에서 JSON 객체를 파싱합니다.
    성공 시 dict/list 반환, 실패 시 None 반환.
    """
    # </think> 이후 텍스트만 사용
    parts = re.split(r"</think>", text, maxsplit=1)
    content = parts[-1].strip()

    # ```json ... ``` 코드 블록 우선 시도
    m = re.search(r"```json\s*([\s\S]+?)\s*```", content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 첫 번째 { ... } 블록 추출
    m2 = re.search(r"(\{[\s\S]*\})", content)
    if m2:
        try:
            return json.loads(m2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def run_inference(model, tokenizer, user_text: str, max_new_tokens: int = 4096) -> dict:
    """
    user_prompt 텍스트 → 모델 → { "raw_output", "json_obj" } 반환.
    json_obj: 파싱된 dict 또는 None.
    """
    prompt = build_prompt(user_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    json_obj = extract_json_from_output(raw_output)

    return {"raw_output": raw_output, "json_obj": json_obj}


# ---------------------------------------------------------------------------
# 파일 처리
# ---------------------------------------------------------------------------

def process_file(file_path: Path, model, tokenizer, output_dir: Path, max_new_tokens: int = 4096) -> None:
    stem = file_path.stem
    out_json = output_dir / f"{stem}.json"

    if out_json.exists():
        print(f"[SKIP] {stem}: 이미 존재함")
        return

    print(f"[PROCESSING] {file_path.name}")
    user_text = file_path.read_text(encoding="utf-8")
    result = run_inference(model, tokenizer, user_text, max_new_tokens=max_new_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)

    if result["json_obj"] is not None:
        out_json.write_text(
            json.dumps(result["json_obj"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [OK] {out_json}")
    else:
        raw_dir = output_dir.parent / "json_infer_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{stem}.txt").write_text(result["raw_output"], encoding="utf-8")
        print(f"  [WARN] JSON 파싱 실패. raw 저장: {raw_dir / f'{stem}.txt'}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="로컬 모델로 user_prompt → JSON 추론")
    parser.add_argument("--model", default="model/qwen3-0.6b-finetuned", help="모델 경로")
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/json_infer", help="출력 디렉토리")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    output_dir = PROJECT_ROOT / args.output
    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "user_prompt"

    model, tokenizer = load_model(model_path)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
    else:
        print(f"[ERROR] 경로를 찾을 수 없음: {input_path}")
        sys.exit(1)

    print(f"\n총 {len(files)}개 파일 처리 시작\n")
    for file_path in files:
        process_file(file_path, model, tokenizer, output_dir, max_new_tokens=args.max_new_tokens)
    print("\n모든 파일 처리 완료.")


if __name__ == "__main__":
    main()
