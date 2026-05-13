"""
prepare_dataset.ipynb 실행 후 생성된 custom-reasoning.json 검수 스크립트

사용법:
    python src/train/validate_dataset.py
    python src/train/validate_dataset.py --data /LLaMA-Factory/data/custom-reasoning.json
    python src/train/validate_dataset.py --cutoff 4096
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"


def check_format(record: dict, idx: int) -> list[str]:
    errors = []
    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append(f"[{idx}] messages 필드 없음 또는 list 아님")
        return errors

    roles = [m.get("role") for m in messages]
    for required in ("system", "user", "assistant"):
        if required not in roles:
            errors.append(f"[{idx}] '{required}' role 없음")

    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not content or not content.strip():
            errors.append(f"[{idx}] '{role}' content 비어 있음")

    return errors


def check_assistant_json(record: dict, idx: int) -> str | None:
    messages = record.get("messages", [])
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if not assistant:
        return None
    content = assistant.get("content", "")
    try:
        json.loads(content)
        return None
    except json.JSONDecodeError as e:
        return f"[{idx}] assistant JSON 파싱 실패: {e}"


def main():
    parser = argparse.ArgumentParser(description="LLaMA-Factory 학습 데이터 검수")
    parser.add_argument("--data", default="/LLaMA-Factory/data/custom-reasoning.json")
    parser.add_argument("--cutoff", type=int, default=8192, help="학습 yaml의 cutoff_len")
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--no-tokenize", action="store_true", help="토큰 길이 측정 생략 (빠른 검수)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 파일 없음: {data_path}")
        sys.exit(1)

    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)
    print(f"로드: {data_path} ({len(records)}개)\n")

    # ── 1. 포맷 검수 ──────────────────────────────────────────────
    format_errors: list[str] = []
    json_errors: list[str] = []
    user_contents: list[str] = []

    for i, rec in enumerate(tqdm(records, desc="포맷/JSON 검수")):
        format_errors.extend(check_format(rec, i))
        err = check_assistant_json(rec, i)
        if err:
            json_errors.append(err)
        # user content 수집 (중복 검사용)
        messages = rec.get("messages", [])
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        user_contents.append(user)

    print(f"\n{'='*55}")
    print("1. 포맷 검수")
    print(f"{'='*55}")
    if format_errors:
        print(f"  오류 {len(format_errors)}건:")
        for e in format_errors[:10]:
            print(f"    {e}")
        if len(format_errors) > 10:
            print(f"    ... 외 {len(format_errors)-10}건")
    else:
        print("  이상 없음 ✓")

    print(f"\n{'='*55}")
    print("2. assistant JSON 유효성")
    print(f"{'='*55}")
    if json_errors:
        print(f"  파싱 실패 {len(json_errors)}건:")
        for e in json_errors[:10]:
            print(f"    {e}")
        if len(json_errors) > 10:
            print(f"    ... 외 {len(json_errors)-10}건")
    else:
        print("  전체 유효한 JSON ✓")

    # ── 2. 중복 검사 ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("3. user_prompt 중복")
    print(f"{'='*55}")
    unique = len(set(user_contents))
    dup = len(user_contents) - unique
    if dup:
        print(f"  중복 {dup}건 발견")
    else:
        print(f"  중복 없음 ✓  (총 {unique}개 고유)")

    # ── 3. 토큰 길이 분포 ─────────────────────────────────────────
    if args.no_tokenize:
        print("\n[토큰 길이 측정 생략]")
        return

    print(f"\n{'='*55}")
    print(f"4. 토큰 길이 분포  (cutoff={args.cutoff})")
    print(f"{'='*55}")
    print(f"토크나이저 로드: {args.tokenizer}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    lengths = []
    for rec in tqdm(records, desc="토큰화"):
        # 전체 대화를 chat template으로 변환해서 실제 학습 시 길이와 동일하게 측정
        messages = rec.get("messages", [])
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            length = len(tok(text, add_special_tokens=False)["input_ids"])
        except Exception:
            length = 0
        lengths.append(length)

    lengths = np.array(lengths)
    print(f"  평균:        {lengths.mean():>8,.0f} tokens")
    print(f"  중간값(50%): {np.percentile(lengths, 50):>8,.0f} tokens")
    print(f"  75%:         {np.percentile(lengths, 75):>8,.0f} tokens")
    print(f"  90%:         {np.percentile(lengths, 90):>8,.0f} tokens")
    print(f"  95%:         {np.percentile(lengths, 95):>8,.0f} tokens")
    print(f"  최대:        {lengths.max():>8,.0f} tokens")
    print()

    over = (lengths > args.cutoff).sum()
    print(f"  cutoff={args.cutoff} 초과: {over}개 ({over/len(lengths)*100:.1f}%)  → 학습 시 잘림")
    print(f"  cutoff={args.cutoff} 이하: {len(lengths)-over}개 ({(len(lengths)-over)/len(lengths)*100:.1f}%)  → 정상 학습")

    # 권장 cutoff
    for pct in (90, 95, 99):
        rec_cutoff = int(np.percentile(lengths, pct))
        print(f"  {pct}% 커버하려면 cutoff_len ≥ {rec_cutoff}")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()
