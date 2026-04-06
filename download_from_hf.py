"""
Hugging Face Dataset에서 데이터를 다운로드합니다.

사용법:
    python download_from_hf.py
    HF_TOKEN=hf_xxx python download_from_hf.py  # private 레포인 경우
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ID = "boradorish/text-to-json-data"
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    token = os.environ.get("HF_TOKEN") or None

    print(f"다운로드 중: {REPO_ID}")
    dataset = load_dataset(REPO_ID, token=token, split="train")
    print(f"총 {len(dataset)}개 샘플")

    dirs = {
        "user_prompt": DATA_DIR / "user_prompt",
        "json": DATA_DIR / "json",
        "json_schema": DATA_DIR / "json_schema",
        "report": DATA_DIR / "report",
        "user_prompt_question": DATA_DIR / "user_prompt_question",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  {i}/{len(dataset)}")

        stem = row["id"]

        if row["user_prompt"]:
            (dirs["user_prompt"] / f"{stem}.txt").write_text(row["user_prompt"], encoding="utf-8")
        if row["json"]:
            (dirs["json"] / f"{stem}.json").write_text(row["json"], encoding="utf-8")
        if row["json_schema"]:
            (dirs["json_schema"] / f"{stem}.json").write_text(row["json_schema"], encoding="utf-8")
        if row["report"]:
            (dirs["report"] / f"{stem}.txt").write_text(row["report"], encoding="utf-8")
        if row["user_prompt_question"]:
            (dirs["user_prompt_question"] / f"{stem}.txt").write_text(row["user_prompt_question"], encoding="utf-8")

    print(f"\n완료: {DATA_DIR}")


if __name__ == "__main__":
    main()
