"""
user_prompt 토큰 길이 분포 분석

사용법:
    python src/test/analyze_token_lengths.py
    python src/test/analyze_token_lengths.py --tokenizer Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--input", default="data/user_prompt", help="user_prompt 디렉토리")
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"[ERROR] txt 파일 없음: {input_dir}")
        sys.exit(1)

    print(f"토크나이저 로드 중: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    lengths = []
    for f in tqdm(files, desc="토큰화 중"):
        text = f.read_text(encoding="utf-8")
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))

    lengths = np.array(lengths)

    print(f"\n{'='*50}")
    print(f"총 샘플 수:   {len(lengths):,}")
    print(f"{'='*50}")
    print(f"평균:         {lengths.mean():,.0f} tokens")
    print(f"중간값(50%):  {np.percentile(lengths, 50):,.0f} tokens")
    print(f"75%:          {np.percentile(lengths, 75):,.0f} tokens")
    print(f"90%:          {np.percentile(lengths, 90):,.0f} tokens")
    print(f"95%:          {np.percentile(lengths, 95):,.0f} tokens")
    print(f"99%:          {np.percentile(lengths, 99):,.0f} tokens")
    print(f"최대:         {lengths.max():,.0f} tokens")
    print(f"{'='*50}")

    for cutoff in [2048, 4096, 8192, 16384]:
        over = (lengths > cutoff).sum()
        print(f"cutoff={cutoff:>6}  초과 샘플: {over:>5}개 ({over/len(lengths)*100:.1f}%)")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
