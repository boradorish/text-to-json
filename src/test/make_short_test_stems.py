"""
Gold JSON token length가 짧은 샘플만 data/test_stems.txt로 저장합니다.

사용법:
    python src/test/make_short_test_stems.py --input data/gold_token_lengths.csv --count 1000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root


PROJECT_ROOT = find_project_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="짧은 gold JSON 샘플만 test_stems.txt로 저장")
    parser.add_argument("--input", default="data/gold_token_lengths.csv", help="analyze_gold_tokens.py 출력 CSV")
    parser.add_argument("--count", type=int, default=1000, help="선택할 샘플 수")
    parser.add_argument("--output", default="data/test_stems.txt", help="저장할 stem 목록")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"CSV 파일 없음: {input_path}")

    rows = []
    with input_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "stem": row["stem"],
                "tokens": int(row["tokens"]),
                "chars": int(row["chars"]),
            })

    rows.sort(key=lambda r: (r["tokens"], r["stem"]))
    selected = rows[: args.count]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(r["stem"] for r in selected) + "\n", encoding="utf-8")

    print(f"입력 샘플: {len(rows)}")
    print(f"선택 샘플: {len(selected)}")
    if selected:
        tokens = [r["tokens"] for r in selected]
        print(f"선택 token min/p50/max: {min(tokens)} / {tokens[len(tokens)//2]} / {max(tokens)}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
