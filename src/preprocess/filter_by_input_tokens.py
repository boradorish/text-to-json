from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .common import delete_related_files, read_json, resolve_path, stem_to_related_paths, write_json
except ImportError:  # pragma: no cover - supports `python src/preprocess/filter_by_input_tokens.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocess.common import delete_related_files, read_json, resolve_path, stem_to_related_paths, write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B-Base"


@dataclass
class TokenRow:
    stem: str
    input_tokens: int
    user_tokens: int
    assistant_tokens: int
    total_tokens: int
    status: str
    user_preview: str


def load_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install transformers first: pip install transformers") from exc
    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chat_text(tokenizer: Any, messages: list[dict], *, add_generation_prompt: bool = False) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return "\n".join(str(message.get("content", "")) for message in messages)


def split_existing_user_prompt(text: str) -> tuple[str, str, str]:
    question, sep, rest = text.partition("=== Report ===")
    if not sep:
        return text.strip(), "", ""
    report, sep, schema = rest.partition("=== JSON Schema ===")
    return question.strip(), report.strip(), schema.strip()


def build_user_prompt(
    stem: str,
    *,
    data_dir: str | Path = "data",
    input_mode: str = "user_prompt",
) -> str | None:
    related = stem_to_related_paths(stem, data_dir=data_dir, include_missing=True)

    if input_mode == "user_prompt":
        path = related["user_prompt"]
        return path.read_text(encoding="utf-8") if path.is_file() else None

    question_path = related["user_prompt_question"]
    report_path = related["report"]
    schema_path = related["json_schema"]

    question = question_path.read_text(encoding="utf-8").strip() if question_path.is_file() else None
    if question is None and related["user_prompt"].is_file():
        question, _, _ = split_existing_user_prompt(related["user_prompt"].read_text(encoding="utf-8"))

    if question is None or not report_path.is_file() or not schema_path.is_file():
        return None

    schema_text = json.dumps(read_json(schema_path), ensure_ascii=False, indent=2)
    report_text = report_path.read_text(encoding="utf-8").strip()
    return f"{question}\n\n=== Report ===\n{report_text}\n\n=== JSON Schema ===\n{schema_text}"


def iter_dataset_stems(data_dir: str | Path = "data") -> list[str]:
    json_dir = resolve_path(data_dir) / "json"
    return sorted(path.stem for path in json_dir.glob("*.json") if path.is_file())


def measure_rows(
    *,
    tokenizer: Any,
    data_dir: str | Path = "data",
    system_prompt_path: str | Path = "prompt/infer_SYSTEM_prompt.txt",
    input_mode: str = "user_prompt",
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[TokenRow]:
    system_prompt = resolve_path(system_prompt_path).read_text(encoding="utf-8")
    rows: list[TokenRow] = []

    stems = iter_dataset_stems(data_dir)
    iterator = stems
    if tqdm is not None:
        iterator = tqdm(stems, desc="measure input tokens", unit="sample")

    for stem in iterator:
        user_prompt = build_user_prompt(stem, data_dir=data_dir, input_mode=input_mode)
        json_path = resolve_path(data_dir) / "json" / f"{stem}.json"
        if user_prompt is None or not json_path.is_file():
            rows.append(
                TokenRow(
                    stem=stem,
                    input_tokens=0,
                    user_tokens=0,
                    assistant_tokens=0,
                    total_tokens=0,
                    status="missing_input",
                    user_preview="",
                )
            )
            continue

        assistant = json.dumps(read_json(json_path), ensure_ascii=False, indent=2)
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_messages = [*input_messages, {"role": "assistant", "content": assistant}]

        input_tokens = token_len(tokenizer, chat_text(tokenizer, input_messages, add_generation_prompt=True))
        user_tokens = token_len(tokenizer, user_prompt)
        assistant_tokens = token_len(tokenizer, assistant)
        total_tokens = token_len(tokenizer, chat_text(tokenizer, full_messages, add_generation_prompt=False))

        reasons: list[str] = []
        if min_tokens is not None and input_tokens < min_tokens:
            reasons.append("below_min")
        if max_tokens is not None and input_tokens > max_tokens:
            reasons.append("above_max")

        rows.append(
            TokenRow(
                stem=stem,
                input_tokens=input_tokens,
                user_tokens=user_tokens,
                assistant_tokens=assistant_tokens,
                total_tokens=total_tokens,
                status="keep" if not reasons else ",".join(reasons),
                user_preview=" ".join(user_prompt.split())[:240],
            )
        )

    return rows


def write_filtered_dataset(
    rows: list[TokenRow],
    *,
    output: str | Path,
    data_dir: str | Path = "data",
    system_prompt_path: str | Path = "prompt/infer_SYSTEM_prompt.txt",
    input_mode: str = "user_prompt",
) -> None:
    system_prompt = resolve_path(system_prompt_path).read_text(encoding="utf-8")
    records: list[dict] = []
    kept_stems = {row.stem for row in rows if row.status == "keep"}
    kept_sorted = sorted(kept_stems)
    iterator = kept_sorted
    if tqdm is not None:
        iterator = tqdm(kept_sorted, desc="write filtered dataset", unit="sample")

    for stem in iterator:
        user_prompt = build_user_prompt(stem, data_dir=data_dir, input_mode=input_mode)
        if user_prompt is None:
            continue
        assistant = json.dumps(read_json(resolve_path(data_dir) / "json" / f"{stem}.json"), ensure_ascii=False, indent=2)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )

    write_json(resolve_path(output), records)


def print_extremes(rows: list[TokenRow], count: int) -> None:
    measured = [row for row in rows if row.status != "missing_input"]
    shortest = sorted(measured, key=lambda row: row.input_tokens)[:count]
    longest = sorted(measured, key=lambda row: row.input_tokens, reverse=True)[:count]

    print(f"\nShortest {count}")
    for row in shortest:
        print(f"  {row.stem:<16} input={row.input_tokens:>7,} total={row.total_tokens:>7,} status={row.status} :: {row.user_preview}")

    print(f"\nLongest {count}")
    for row in longest:
        print(f"  {row.stem:<16} input={row.input_tokens:>7,} total={row.total_tokens:>7,} status={row.status} :: {row.user_preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure model input token length and keep only records inside an inclusive min/max range."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--system-prompt", default="prompt/infer_SYSTEM_prompt.txt")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--input-mode", choices=["user_prompt", "composed"], default="user_prompt")
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--extremes", type=int, default=5)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--output-dataset", default=None, help="Write kept records in LLaMA-Factory sharegpt JSON format.")
    parser.add_argument("--delete-dropped", action="store_true", help="Delete related source files for out-of-range records.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Without this, deletion is dry-run.")
    args = parser.parse_args()

    print(f"tokenizer: {args.tokenizer}")
    print(f"Paths")
    print(f"  data_dir:       {resolve_path(args.data_dir)}")
    print(f"  system_prompt:  {resolve_path(args.system_prompt)}")
    tokenizer = load_tokenizer(args.tokenizer)
    rows = measure_rows(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        system_prompt_path=args.system_prompt,
        input_mode=args.input_mode,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    counts = Counter(row.status for row in rows)
    measured = [row for row in rows if row.status != "missing_input"]
    kept = [row for row in rows if row.status == "keep"]
    print("Input token filter")
    print(f"  total:          {len(rows):,}")
    print(f"  measured:       {len(measured):,}")
    print(f"  keep:           {len(kept):,}")
    for status, count in counts.most_common():
        if status != "keep":
            print(f"  {status:<14} {count:,}")

    if measured:
        values = sorted(row.input_tokens for row in measured)
        print(f"  min/median/max: {values[0]:,} / {values[len(values)//2]:,} / {values[-1]:,}")

    print_extremes(rows, args.extremes)

    if args.report_out:
        write_json(resolve_path(args.report_out), [asdict(row) for row in rows])
        print(f"\nreport: {resolve_path(args.report_out)}")

    if args.output_dataset:
        write_filtered_dataset(
            rows,
            output=args.output_dataset,
            data_dir=args.data_dir,
            system_prompt_path=args.system_prompt,
            input_mode=args.input_mode,
        )
        print(f"filtered dataset: {resolve_path(args.output_dataset)}")

    if args.delete_dropped:
        dropped = [row.stem for row in rows if row.status not in {"keep", "missing_input"}]
        delete_result = delete_related_files(dropped, data_dir=args.data_dir, dry_run=not args.execute)
        mode = "deleted" if args.execute else "would delete"
        print(f"{mode}: {len(delete_result.deleted):,} related files for {len(set(dropped)):,} stems")
        if delete_result.failed:
            print(f"delete failures: {len(delete_result.failed):,}")


if __name__ == "__main__":
    main()
