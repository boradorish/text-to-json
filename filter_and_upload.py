"""
user_prompt 토큰 길이가 max_tokens를 초과하는 샘플을 삭제하고 HF에 재업로드합니다.

사용법:
    HF_TOKEN=hf_xxx python filter_and_upload.py
    HF_TOKEN=hf_xxx python filter_and_upload.py --max-tokens 8192
    HF_TOKEN=hf_xxx python filter_and_upload.py --max-tokens 8192 --dry-run  # 삭제 없이 미리 확인
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
REPO_ID = "boradorish/text-to-json-data"
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"

SUBDIRS = {
    "user_prompt":          ("txt", DATA_DIR / "user_prompt"),
    "json":                 ("json", DATA_DIR / "json"),
    "json_schema":          ("json", DATA_DIR / "json_schema"),
    "report":               ("txt", DATA_DIR / "report"),
    "user_prompt_question": ("txt", DATA_DIR / "user_prompt_question"),
}


def load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="실제 삭제 없이 대상 파일만 출력")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("[ERROR] HF_TOKEN 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 전체 stem 목록
    user_prompt_dir = DATA_DIR / "user_prompt"
    stems = sorted(p.stem for p in user_prompt_dir.glob("*.txt"))
    print(f"전체 샘플: {len(stems)}개")

    # 토크나이저로 길이 측정
    print(f"\n토크나이저 로드 중: {TOKENIZER_ID}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

    over_limit = []
    for stem in tqdm(stems, desc="토큰 길이 측정"):
        text = load_text(user_prompt_dir / f"{stem}.txt") or ""
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n > args.max_tokens:
            over_limit.append(stem)

    print(f"\n{args.max_tokens} 토큰 초과: {len(over_limit)}개 / {len(stems)}개 ({len(over_limit)/len(stems)*100:.1f}%)")
    keep = len(stems) - len(over_limit)
    print(f"남길 샘플: {keep}개")

    if not over_limit:
        print("삭제할 파일이 없습니다.")
        return

    if args.dry_run:
        print("\n[DRY RUN] 삭제될 stem 목록 (최대 20개 출력):")
        for s in over_limit[:20]:
            print(f"  {s}")
        if len(over_limit) > 20:
            print(f"  ... 외 {len(over_limit) - 20}개")
        return

    # 파일 삭제
    print("\n파일 삭제 중...")
    for stem in tqdm(over_limit, desc="삭제"):
        for _, (ext, d) in SUBDIRS.items():
            p = d / f"{stem}.{ext}"
            if p.exists():
                p.unlink()

    # 남은 파일로 HF 업로드
    remaining = sorted(p.stem for p in user_prompt_dir.glob("*.txt"))
    print(f"\n남은 샘플: {len(remaining)}개 — HF 업로드 준비 중...")

    records = []
    for stem in tqdm(remaining, desc="레코드 구성"):
        records.append({
            "id": stem,
            "user_prompt":          load_text(DATA_DIR / "user_prompt"          / f"{stem}.txt"),
            "json":                 load_text(DATA_DIR / "json"                  / f"{stem}.json"),
            "json_schema":          load_text(DATA_DIR / "json_schema"           / f"{stem}.json"),
            "report":               load_text(DATA_DIR / "report"                / f"{stem}.txt"),
            "user_prompt_question": load_text(DATA_DIR / "user_prompt_question"  / f"{stem}.txt"),
        })

    dataset = Dataset.from_list(records)
    print(f"\nHugging Face 업로드 중: {REPO_ID}")
    dataset.push_to_hub(REPO_ID, token=token, private=False)
    print(f"\n완료: https://huggingface.co/datasets/{REPO_ID}")
    print(f"  삭제: {len(over_limit)}개 / 업로드: {len(remaining)}개")


if __name__ == "__main__":
    main()
