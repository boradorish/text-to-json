"""
prepare_dataset.ipynb과 동일한 random.seed(42) + 90/10 분리 로직으로
data/test_stems.txt 를 생성합니다.

사용법:
    python src/test/make_test_split.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()


def main() -> None:
    data_dir = PROJECT_ROOT / "data"

    stems = []
    for user_file in sorted(data_dir.glob("user_prompt/*.txt")):
        name = user_file.stem
        if (data_dir / "report" / f"{name}.txt").exists() and (data_dir / "json" / f"{name}.json").exists():
            stems.append(name)

    print(f"총 {len(stems)}개 샘플 발견")

    random.seed(42)
    random.shuffle(stems)

    split = int(len(stems) * 0.9)
    test_stems = stems[split:]

    test_stems_path = data_dir / "test_stems.txt"
    test_stems_path.write_text("\n".join(test_stems), encoding="utf-8")

    print(f"train: {split}개 / test: {len(test_stems)}개")
    print(f"test_stems.txt 저장: {test_stems_path}")


if __name__ == "__main__":
    main()
