"""
학습된 로컬 모델로 user_prompt → JSON 추론을 수행합니다.
결과는 GT(정답 JSON, JSON Schema)와 함께 단일 Excel 파일로 저장됩니다.

사용법:
    python src/test/infer.py --test-only
    python src/test/infer.py --model models/qwen3-0.6b-finetuned
    python src/test/infer.py --batch-size 16 --output data/infer_results.xlsx
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()
SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "json_SYSTEM_prompt.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 모델 로드
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
# 추론
# ---------------------------------------------------------------------------

def build_prompt(user_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="로컬 모델로 user_prompt → JSON 배치 추론 (Excel 저장)")
    parser.add_argument("--model", default="models/qwen3-0.6b-finetuned")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/infer_results.xlsx")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-only", action="store_true", help="data/test_stems.txt 기준으로 필터링")
    args = parser.parse_args()

    _model_arg = Path(args.model)
    if _model_arg.is_absolute() or _model_arg.exists():
        model_path = _model_arg
    else:
        local_path = PROJECT_ROOT / args.model
        model_path = local_path if local_path.exists() else args.model

    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "user_prompt"
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data"
    gold_dir = data_dir / "json"
    schema_dir = data_dir / "json_schema"

    # 파일 목록 수집
    if input_path.is_file():
        all_files = [input_path]
    elif input_path.is_dir():
        all_files = sorted(input_path.glob("*.txt"))
    else:
        print(f"[ERROR] 경로를 찾을 수 없음: {input_path}")
        sys.exit(1)

    if args.test_only:
        test_stems_path = data_dir / "test_stems.txt"
        if not test_stems_path.exists():
            print(f"[ERROR] test_stems.txt 없음: {test_stems_path}")
            sys.exit(1)
        test_stems = set(test_stems_path.read_text(encoding="utf-8").splitlines())
        all_files = [f for f in all_files if f.stem in test_stems]
        print(f"[TEST] test_stems.txt 기준 {len(all_files)}개 파일 필터링")

    if not all_files:
        print("[WARN] 처리할 파일이 없습니다.")
        return

    # GT 데이터 로드
    rows = []
    for f in all_files:
        stem = f.stem
        gold_path = gold_dir / f"{stem}.json"
        schema_path = schema_dir / f"{stem}.json"
        rows.append({
            "stem": stem,
            "user_prompt": f.read_text(encoding="utf-8"),
            "gold_json": gold_path.read_text(encoding="utf-8") if gold_path.exists() else "",
            "json_schema": schema_path.read_text(encoding="utf-8") if schema_path.exists() else "",
        })

    model, tokenizer = load_model(model_path, tokenizer_path=args.tokenizer)

    # 배치 추론
    all_results: list[dict] = []
    user_texts = [r["user_prompt"] for r in rows]
    total = len(user_texts)
    for i in range(0, total, args.batch_size):
        batch = user_texts[i : i + args.batch_size]
        print(f"[{i + 1}~{min(i + args.batch_size, total)}/{total}] 추론 중...")
        all_results.extend(run_batch_inference(model, tokenizer, batch, args.max_new_tokens))

    # Excel 저장
    records = []
    for row, result in zip(rows, all_results):
        pred_json_str = (
            json.dumps(result["json_obj"], ensure_ascii=False)
            if result["json_obj"] is not None
            else ""
        )
        records.append({
            "stem": row["stem"],
            "user_prompt": row["user_prompt"],
            "gold_json": row["gold_json"],
            "json_schema": row["json_schema"],
            "raw_output": result["raw_output"],
            "pred_json": pred_json_str,
        })

    df = pd.DataFrame(records)
    df.to_excel(output_path, index=False)

    parsed = sum(1 for r in records if r["pred_json"])
    print(f"\n완료. 총 {total}개 / JSON 파싱 성공 {parsed}개 / 실패 {total - parsed}개")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
