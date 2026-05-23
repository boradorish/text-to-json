"""
Gold JSON token length 기준으로 data/test_stems.txt를 저장합니다.

사용법:
    python src/test/make_short_test_stems.py --input data/gold_token_lengths.csv --count 1000
    python src/test/make_short_test_stems.py --input data/gold_token_lengths.csv --count 1000 --max-tokens 512 --strategy random --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root


PROJECT_ROOT = find_project_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold JSON token length 기준으로 test_stems.txt 저장")
    parser.add_argument("--input", default="data/gold_token_lengths.csv", help="analyze_gold_tokens.py 출력 CSV")
    parser.add_argument("--count", type=int, default=1000, help="선택할 샘플 수")
    parser.add_argument("--max-tokens", type=int, default=None, help="이 token 수 이하인 샘플만 후보로 사용")
    parser.add_argument("--strategy", choices=["shortest", "random"], default="shortest", help="선택 방식")
    parser.add_argument("--seed", type=int, default=42, help="strategy=random일 때 사용할 random seed")
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

    candidates = rows
    if args.max_tokens is not None:
        candidates = [row for row in rows if row["tokens"] <= args.max_tokens]

    if args.strategy == "random":
        rng = random.Random(args.seed)
        candidates = sorted(candidates, key=lambda r: r["stem"])
        selected = rng.sample(candidates, k=min(args.count, len(candidates)))
        selected.sort(key=lambda r: r["stem"])
    else:
        candidates = sorted(candidates, key=lambda r: (r["tokens"], r["stem"]))
        selected = candidates[: args.count]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(r["stem"] for r in selected) + "\n", encoding="utf-8")

    print(f"입력 샘플: {len(rows)}")
    if args.max_tokens is not None:
        print(f"후보 샘플(max_tokens<={args.max_tokens}): {len(candidates)}")
    print(f"선택 방식: {args.strategy}")
    if args.strategy == "random":
        print(f"random seed: {args.seed}")
    print(f"선택 샘플: {len(selected)}")
    if selected:
        tokens = [r["tokens"] for r in selected]
        print(f"선택 token min/p50/max: {min(tokens)} / {tokens[len(tokens)//2]} / {max(tokens)}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
