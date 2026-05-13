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


def _is_lora_adapter(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def _get_base_model_id(adapter_path: str) -> str:
    cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    return cfg["base_model_name_or_path"]


def load_model(model_path: str | Path, tokenizer_path: str | Path | None = None):
    model_path = str(model_path)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if _is_lora_adapter(model_path):
        # LoRA 어댑터: adapter_config.json에서 베이스 모델 읽어서 로드 후 어댑터 병합
        from peft import PeftModel
        base_id = _get_base_model_id(model_path)
        print(f"LoRA 어댑터 감지. 베이스 모델: {base_id}")
        tokenizer_src = str(tokenizer_path) if tokenizer_path else base_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 추론 속도를 위해 가중치 병합
    else:
        # 풀 모델 (merge된 체크포인트 또는 HF repo)
        repo_id, subfolder = _parse_model_path(model_path)
        tokenizer_src = str(tokenizer_path) if tokenizer_path else model_path
        print(f"모델 로드 중: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        subfolder_kwargs = {"subfolder": subfolder} if subfolder else {}
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            **subfolder_kwargs,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

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

_EXCEL_TRUNCATE = 32000  # Excel 셀 한도(32767)보다 약간 짧게
_DISPLAY_COLS = ("user_prompt", "raw_output")  # Excel에서 잘라도 평가에 영향 없는 컬럼


def _save(records: list[dict], jsonl_path: Path, xlsx_path: Path) -> None:
    # JSONL — 원본 전체 저장 (평가용)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Excel — 긴 display 컬럼만 잘라서 저장 (열람용)
    display_records = []
    for r in records:
        row = dict(r)
        for col in _DISPLAY_COLS:
            if col in row and isinstance(row[col], str) and len(row[col]) > _EXCEL_TRUNCATE:
                row[col] = row[col][:_EXCEL_TRUNCATE] + "…"
        display_records.append(row)
    pd.DataFrame(display_records).to_excel(xlsx_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="로컬 모델로 user_prompt → JSON 배치 추론")
    parser.add_argument("--model", default="models/qwen3-0.6b-finetuned")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/infer_results", help="출력 경로 (확장자 제외, .jsonl/.xlsx 자동 생성)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-only", action="store_true", help="data/test_stems.txt 기준으로 필터링")
    args = parser.parse_args()

    _model_arg = Path(args.model)
    if _model_arg.is_absolute() or _model_arg.exists():
        model_path = _model_arg
    else:
        local_path = PROJECT_ROOT / args.model
        model_path = local_path if local_path.exists() else args.model

    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "user_prompt"
    base_path = PROJECT_ROOT / args.output
    jsonl_path = base_path.with_suffix(".jsonl")
    xlsx_path = base_path.with_suffix(".xlsx")
    base_path.parent.mkdir(parents=True, exist_ok=True)

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

    # 기존 JSONL에서 이어 처리
    saved_records: list[dict] = []
    done_stems: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    saved_records.append(r)
                    done_stems.add(str(r["stem"]))
        print(f"[RESUME] 기존 결과 {len(done_stems)}개 발견, 이어서 처리합니다.")

    all_files = [f for f in all_files if f.stem not in done_stems]

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

    # 배치 추론 — 배치마다 저장
    total = len(rows)
    for i in range(0, total, args.batch_size):
        batch_rows = rows[i : i + args.batch_size]
        batch_texts = [r["user_prompt"] for r in batch_rows]
        print(f"[{i + 1}~{min(i + args.batch_size, total)}/{total}] 추론 중...")
        batch_results = run_batch_inference(model, tokenizer, batch_texts, args.max_new_tokens)

        for row, result in zip(batch_rows, batch_results):
            saved_records.append({
                "stem": row["stem"],
                "user_prompt": row["user_prompt"],
                "gold_json": row["gold_json"],
                "json_schema": row["json_schema"],
                "raw_output": result["raw_output"],
                "pred_json": (
                    json.dumps(result["json_obj"], ensure_ascii=False)
                    if result["json_obj"] is not None
                    else ""
                ),
            })

        _save(saved_records, jsonl_path, xlsx_path)
        parsed = sum(1 for r in saved_records if r["pred_json"])
        print(f"  → 저장 ({len(saved_records)}개 누적 / 파싱 성공 {parsed}개)")

    print(f"\n완료. 총 {len(saved_records)}개")
    print(f"  JSONL (평가용): {jsonl_path}")
    print(f"  Excel (열람용): {xlsx_path}")


if __name__ == "__main__":
    main()
