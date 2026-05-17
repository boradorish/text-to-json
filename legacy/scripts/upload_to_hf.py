"""
데이터를 Hugging Face Dataset으로 업로드합니다.

사용법:
    HF_TOKEN=hf_xxx python upload_to_hf.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

REPO_ID = "boradorish/text-to-json-data"
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"


def load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def load_json(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def build_records() -> list[dict]:
    user_prompt_dir = DATA_DIR / "user_prompt"
    json_dir = DATA_DIR / "json"
    json_schema_dir = DATA_DIR / "json_schema"
    report_dir = DATA_DIR / "report"
    user_prompt_question_dir = DATA_DIR / "user_prompt_question"

    stems = sorted(p.stem for p in user_prompt_dir.glob("*.txt"))
    print(f"총 {len(stems)}개 샘플 로드 중...")

    records = []
    for i, stem in enumerate(stems):
        if i % 1000 == 0:
            print(f"  {i}/{len(stems)}")

        record = {
            "id": stem,
            "user_prompt": load_text(user_prompt_dir / f"{stem}.txt"),
            "json": load_json(json_dir / f"{stem}.json"),
            "json_schema": load_json(json_schema_dir / f"{stem}.json"),
            "report": load_text(report_dir / f"{stem}.txt"),
            "user_prompt_question": load_text(user_prompt_question_dir / f"{stem}.txt"),
        }
        records.append(record)

    return records


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    records = build_records()
    print(f"\n데이터셋 생성 중... ({len(records)}개 샘플)")
    dataset = Dataset.from_list(records)

    print(f"\nHugging Face 업로드 중: {REPO_ID}")
    dataset.push_to_hub(
        REPO_ID,
        token=token,
        private=False,
    )
    print(f"\n완료: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
