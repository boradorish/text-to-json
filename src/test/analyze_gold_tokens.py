"""
Gold JSON token length distribution을 측정합니다.

사용법:
    python src/test/analyze_gold_tokens.py --model saves/qwen3-0.6b/full/sft --test-only
    python src/test/analyze_gold_tokens.py --tokenizer Qwen/Qwen3-0.6B --test-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_loader import find_project_root


PROJECT_ROOT = find_project_root()


def _parse_model_path(model_path: str) -> tuple[str, str | None]:
    parts = model_path.split("/")
    if len(parts) > 2 and not model_path.startswith("/"):
        return "/".join(parts[:2]), "/".join(parts[2:])
    return model_path, None


def _resolve_tokenizer_source(model: str | None, tokenizer: str | None) -> tuple[str, str | None]:
    src = tokenizer or model
    if not src:
        raise ValueError("--model 또는 --tokenizer 중 하나는 필요합니다.")

    p = Path(src)
    if p.is_absolute() or p.exists():
        return str(p), None

    local_path = PROJECT_ROOT / src
    if local_path.exists():
        return str(local_path), None

    return _parse_model_path(src)


def _load_json_text(path: Path) -> str | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold JSON token length 분석")
    parser.add_argument("--model", default=None, help="모델/체크포인트 경로 또는 HF repo id")
    parser.add_argument("--tokenizer", default=None, help="별도 tokenizer 경로 또는 HF repo id")
    parser.add_argument("--test-only", action="store_true", help="data/test_stems.txt 기준으로 분석")
    parser.add_argument("--output", default="data/gold_token_lengths.csv")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer_src, subfolder = _resolve_tokenizer_source(args.model, args.tokenizer)
    kwargs = {"subfolder": subfolder} if subfolder else {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True, **kwargs)

    data_dir = PROJECT_ROOT / "data"
    json_dir = data_dir / "json"

    files = sorted(json_dir.glob("*.json"))
    if args.test_only:
        test_stems_path = data_dir / "test_stems.txt"
        if not test_stems_path.exists():
            print(f"[ERROR] test_stems.txt 없음: {test_stems_path}")
            sys.exit(1)
        test_stems = set(test_stems_path.read_text(encoding="utf-8").splitlines())
        files = [p for p in files if p.stem in test_stems]

    rows = []
    skipped = 0
    for path in files:
        text = _load_json_text(path)
        if text is None:
            skipped += 1
            continue
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        rows.append({
            "stem": path.stem,
            "chars": len(text),
            "tokens": token_count,
        })

    if not rows:
        print("[WARN] 분석할 gold JSON이 없습니다.")
        return

    rows.sort(key=lambda r: r["tokens"], reverse=True)
    out = (PROJECT_ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".xlsx":
        import pandas as pd
        pd.DataFrame(rows).to_excel(out, index=False)
    else:
        import csv
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "chars", "tokens"])
            writer.writeheader()
            writer.writerows(rows)

    tokens = [r["tokens"] for r in rows]
    print(f"샘플 수: {len(rows)}")
    print(f"파싱 실패/스킵: {skipped}")
    print(f"min: {min(tokens)}")
    print(f"p50: {_quantile(tokens, 0.5):.0f}")
    print(f"p90: {_quantile(tokens, 0.9):.0f}")
    print(f"p95: {_quantile(tokens, 0.95):.0f}")
    print(f"p99: {_quantile(tokens, 0.99):.0f}")
    print(f"max: {max(tokens)}")
    print("\n상위 10개:")
    for row in rows[:10]:
        print(f"{row['stem']}\tchars={row['chars']}\ttokens={row['tokens']}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
