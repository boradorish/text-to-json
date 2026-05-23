#!/usr/bin/env python3
"""
Analyze text/token length distributions for Hugging Face or local tabular datasets.

Example usage:
    python analyze_dataset_lengths.py \
      --source hf \
      --dataset boradorish/text-to-json-data \
      --preset text-to-json \
      --split train \
      --output_prefix ours_text_to_json

    python analyze_dataset_lengths.py \
      --source xlsx \
      --input ../DeepJSONEval/data.xlsx \
      --text_column prompt \
      --output_prefix deepjsoneval
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


LOGGER = logging.getLogger("dataset_lengths")
DEFAULT_TEXT_TO_JSON_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
DEFAULT_SYSTEM_PROMPT = "prompt/infer_SYSTEM_prompt.txt"

TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "document",
    "report",
    "input",
    "prompt",
    "user_prompt",
    "instruction",
    "question",
    "source_text",
    "body",
)

ID_COLUMN_CANDIDATES = (
    "id",
    "stem",
    "filename",
    "file_name",
    "doc_id",
    "document_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sample character, word, and token length statistics."
    )
    parser.add_argument(
        "--source",
        choices=("hf", "csv", "jsonl", "xlsx"),
        required=True,
        help="Input source type.",
    )
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset repo id for --source hf.")
    parser.add_argument("--split", default=None, help="Hugging Face split. Defaults to first available split.")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming mode.")
    parser.add_argument("--input", default=None, help="Local CSV/JSONL/XLSX path.")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or zero-based sheet index.")
    parser.add_argument(
        "--preset",
        choices=("generic", "text-to-json"),
        default="generic",
        help="Use text-to-json prompt construction for boradorish/text-to-json-data.",
    )
    parser.add_argument("--text_column", default=None, help="Single text column to measure.")
    parser.add_argument(
        "--text_columns",
        nargs="+",
        default=None,
        help="Multiple columns to concatenate before measuring.",
    )
    parser.add_argument("--id_column", default=None, help="Optional row id column.")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum rows to analyze.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument("--output_prefix", default=None, help="Output file prefix.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Optional Hugging Face tokenizer. If omitted, text-to-json preset defaults to "
            f"{DEFAULT_TEXT_TO_JSON_TOKENIZER}; generic mode uses tiktoken/fallback."
        ),
    )
    parser.add_argument("--tiktoken_encoding", default="cl100k_base")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.source == "hf" and not args.dataset:
        raise ValueError("--dataset is required when --source hf")
    if args.source != "hf" and not args.input:
        raise ValueError("--input is required for local sources")
    if args.text_column and args.text_columns:
        raise ValueError("Use either --text_column or --text_columns, not both")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def load_length_tokenizer(args: argparse.Namespace):
    tokenizer_name = args.tokenizer
    if args.preset == "text-to-json" and tokenizer_name is None:
        tokenizer_name = DEFAULT_TEXT_TO_JSON_TOKENIZER

    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to use --tokenizer.") from exc
        LOGGER.info("Loading Hugging Face tokenizer: %s", tokenizer_name)
        return ("transformers", AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True), tokenizer_name)

    try:
        import tiktoken

        LOGGER.info("Using tiktoken encoding: %s", args.tiktoken_encoding)
        return ("tiktoken", tiktoken.get_encoding(args.tiktoken_encoding), args.tiktoken_encoding)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("tiktoken unavailable (%s); using ceil(word_count * 1.3).", exc)
        return ("fallback", None, "ceil(word_count * 1.3)")


def token_len(tokenizer_bundle: tuple[str, Any, str], text: str, word_count: int | None = None) -> int:
    kind, tokenizer, _ = tokenizer_bundle
    if kind == "transformers":
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if kind == "tiktoken":
        return len(tokenizer.encode(text))
    if word_count is None:
        word_count = len(text.split())
    return int(math.ceil(word_count * 1.3))


def chat_text(tokenizer_bundle: tuple[str, Any, str], messages: list[dict], *, add_generation_prompt: bool) -> str:
    kind, tokenizer, _ = tokenizer_bundle
    if kind == "transformers":
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)


def infer_split(dataset_name: str, requested_split: str | None) -> str | None:
    if requested_split:
        return requested_split
    try:
        from datasets import get_dataset_split_names

        splits = get_dataset_split_names(dataset_name)
        if splits:
            LOGGER.info("Available splits: %s", ", ".join(splits))
            LOGGER.info("No --split provided; using first split: %s", splits[0])
            return splits[0]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not inspect split names: %s", exc)
    return None


def iter_hf_rows(args: argparse.Namespace) -> tuple[list[str], Iterator[dict[str, Any]], str | None]:
    split = infer_split(args.dataset, args.split)
    token = os.environ.get("HF_TOKEN") or None
    LOGGER.info("Loading HF dataset=%s split=%s streaming=%s", args.dataset, split or "<default>", args.streaming)
    if split is None:
        loaded = load_dataset(args.dataset, streaming=args.streaming, token=token)
    else:
        loaded = load_dataset(args.dataset, split=split, streaming=args.streaming, token=token)
    if isinstance(loaded, dict):
        first_split = next(iter(loaded.keys()))
        LOGGER.info("Loaded split dictionary; using first split: %s", first_split)
        split = first_split
        loaded = loaded[first_split]

    iterator = iter(loaded)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Dataset split is empty.") from exc
    columns = list(first.keys())
    LOGGER.info("Available columns: %s", ", ".join(columns))
    return columns, itertools.chain([first], iterator), split


def read_jsonl_rows(path: Path) -> tuple[list[str], Iterator[dict[str, Any]]]:
    def iterator() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                yield row

    rows = iterator()
    try:
        first = next(rows)
    except StopIteration as exc:
        raise ValueError(f"JSONL file is empty: {path}") from exc
    columns = list(first.keys())
    LOGGER.info("Available columns: %s", ", ".join(columns))
    return columns, itertools.chain([first], rows)


def read_csv_rows(path: Path, chunksize: int) -> tuple[list[str], Iterator[dict[str, Any]]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    LOGGER.info("Available columns: %s", ", ".join(columns))

    def iterator() -> Iterator[dict[str, Any]]:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            for row in chunk.to_dict(orient="records"):
                yield row

    return columns, iterator()


def parse_sheet(sheet: str) -> str | int:
    if re.fullmatch(r"\d+", str(sheet)):
        return int(sheet)
    return sheet


def read_xlsx_rows(path: Path, sheet: str) -> tuple[list[str], Iterator[dict[str, Any]]]:
    frame = pd.read_excel(path, sheet_name=parse_sheet(sheet))
    columns = [str(column) for column in frame.columns]
    frame.columns = columns
    LOGGER.info("Available columns: %s", ", ".join(columns))
    return columns, iter(frame.to_dict(orient="records"))


def score_column(column: str, candidates: tuple[str, ...]) -> tuple[int, int]:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    exact = 100 if normalized in candidates else 0
    partial = max((50 + len(item) for item in candidates if item in normalized), default=0)
    return exact + partial, -len(normalized)


def choose_id_column(columns: list[str], override: str | None) -> str | None:
    if override:
        if override not in columns:
            raise ValueError(f"--id_column {override!r} not found. Available: {columns}")
        return override
    ranked = sorted(columns, key=lambda col: score_column(col, ID_COLUMN_CANDIDATES), reverse=True)
    return ranked[0] if ranked and score_column(ranked[0], ID_COLUMN_CANDIDATES)[0] > 0 else None


def choose_text_columns(columns: list[str], args: argparse.Namespace) -> list[str]:
    if args.preset == "text-to-json":
        return []
    if args.text_columns:
        missing = [column for column in args.text_columns if column not in columns]
        if missing:
            raise ValueError(f"--text_columns not found: {missing}. Available: {columns}")
        return args.text_columns
    if args.text_column:
        if args.text_column not in columns:
            raise ValueError(f"--text_column {args.text_column!r} not found. Available: {columns}")
        return [args.text_column]

    ranked = sorted(columns, key=lambda col: score_column(col, TEXT_COLUMN_CANDIDATES), reverse=True)
    if ranked and score_column(ranked[0], TEXT_COLUMN_CANDIDATES)[0] > 0:
        LOGGER.info("Automatically selected text column: %s", ranked[0])
        return [ranked[0]]
    raise ValueError(f"Could not infer text column. Available: {columns}. Pass --text_column.")


def load_system_prompt(path_text: str | Path) -> str:
    path = resolve_path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def maybe_json_text(value: Any) -> str:
    text = coerce_text(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def build_text_to_json_measurement(
    row: dict[str, Any],
    tokenizer_bundle: tuple[str, Any, str],
    system_prompt: str,
) -> tuple[str, dict[str, int]]:
    user_prompt = coerce_text(row.get("user_prompt"))
    if not user_prompt:
        question = coerce_text(row.get("user_prompt_question")).strip()
        report = coerce_text(row.get("report")).strip()
        schema = maybe_json_text(row.get("json_schema")).strip()
        parts = []
        if question:
            parts.append(question)
        if report:
            parts.append(f"=== Report ===\n{report}")
        if schema:
            parts.append(f"=== JSON Schema ===\n{schema}")
        user_prompt = "\n\n".join(parts)

    assistant = maybe_json_text(row.get("json"))
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = [*input_messages, {"role": "assistant", "content": assistant}]
    input_text = chat_text(tokenizer_bundle, input_messages, add_generation_prompt=True)
    full_text = chat_text(tokenizer_bundle, full_messages, add_generation_prompt=False)
    return input_text, {
        "user_tokens": token_len(tokenizer_bundle, user_prompt),
        "assistant_tokens": token_len(tokenizer_bundle, assistant),
        "total_tokens": token_len(tokenizer_bundle, full_text),
    }


def build_generic_text(row: dict[str, Any], text_columns: list[str]) -> str:
    return "\n\n".join(coerce_text(row.get(column)) for column in text_columns if coerce_text(row.get(column)))


def compute_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            key: None
            for key in ("mean", "median", "std", "min", "max", "p5", "p25", "p50", "p75", "p90", "p95", "p99")
        } | {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    p5, p25, p50, p75, p90, p95, p99 = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def highly_skewed(values: list[int]) -> bool:
    positive = np.asarray([value for value in values if value > 0], dtype=np.float64)
    if positive.size < 2:
        return False
    p50, p95 = np.percentile(positive, [50, 95])
    return bool(p50 > 0 and p95 / p50 >= 5)


def save_plot(token_counts: list[int], output_path: Path, title: str) -> None:
    positive = np.asarray([value for value in token_counts if value > 0], dtype=np.float64)
    if positive.size == 0:
        LOGGER.warning("No positive token counts; skipping plot.")
        return
    use_log = highly_skewed(token_counts)
    plt.figure(figsize=(10, 6))
    if use_log and positive.min() != positive.max():
        bins = np.logspace(np.log10(max(1, positive.min())), np.log10(positive.max()), 60)
        plt.hist(positive, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
        plt.xscale("log")
        plt.xlabel("Token count (log scale)")
    else:
        plt.hist(positive, bins=60, color="#4C78A8", alpha=0.85, edgecolor="white")
        plt.xlabel("Token count")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.grid(True, which="major", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    if args.dataset:
        return args.dataset.split("/")[-1].lower().replace("-", "_")
    return resolve_path(args.input).stem.lower().replace("-", "_")


def source_rows(args: argparse.Namespace) -> tuple[list[str], Iterator[dict[str, Any]], str | None]:
    if args.source == "hf":
        return iter_hf_rows(args)
    path = resolve_path(args.input)
    if args.source == "csv":
        columns, rows = read_csv_rows(path, args.chunksize)
    elif args.source == "jsonl":
        columns, rows = read_jsonl_rows(path)
    elif args.source == "xlsx":
        columns, rows = read_xlsx_rows(path, args.sheet)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source: {args.source}")
    return columns, rows, None


def analyze(args: argparse.Namespace) -> None:
    columns, rows_iter, split = source_rows(args)
    id_column = choose_id_column(columns, args.id_column)
    text_columns = choose_text_columns(columns, args)
    tokenizer_bundle = load_length_tokenizer(args)
    system_prompt = load_system_prompt(args.system_prompt) if args.preset == "text-to-json" else ""

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(args)
    csv_path = out_dir / f"{prefix}_lengths.csv"
    jsonl_path = out_dir / f"{prefix}_lengths.jsonl"
    summary_path = out_dir / f"{prefix}_length_summary.json"
    plot_path = out_dir / f"{prefix}_token_length_distribution.png"

    fieldnames = [
        "row_index",
        "source_id",
        "char_length",
        "word_count",
        "token_count",
        "user_tokens",
        "assistant_tokens",
        "total_tokens",
        "preset",
        "text_columns",
        "split",
    ]
    char_lengths: list[int] = []
    word_counts: list[int] = []
    token_counts: list[int] = []
    missing_text = 0

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        progress = tqdm(rows_iter, total=args.max_samples, desc="Analyzing lengths", unit="row")
        for row_index, row in enumerate(progress):
            if row_index >= args.max_samples:
                break
            if args.preset == "text-to-json":
                text, extra = build_text_to_json_measurement(row, tokenizer_bundle, system_prompt)
            else:
                text = build_generic_text(row, text_columns)
                extra = {"user_tokens": None, "assistant_tokens": None, "total_tokens": None}
            if not text:
                missing_text += 1
            char_length = len(text)
            word_count = len(text.split())
            token_count = token_len(tokenizer_bundle, text, word_count)
            source_id = coerce_text(row.get(id_column)) if id_column else str(row_index)
            result = {
                "row_index": row_index,
                "source_id": source_id or str(row_index),
                "char_length": char_length,
                "word_count": word_count,
                "token_count": token_count,
                "user_tokens": extra["user_tokens"],
                "assistant_tokens": extra["assistant_tokens"],
                "total_tokens": extra["total_tokens"],
                "preset": args.preset,
                "text_columns": ",".join(text_columns),
                "split": split or "",
            }
            writer.writerow(result)
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            char_lengths.append(char_length)
            word_counts.append(word_count)
            token_counts.append(token_count)

    kind, _, tokenizer_name = tokenizer_bundle
    summary = {
        "source": args.source,
        "dataset": args.dataset,
        "input": str(resolve_path(args.input)) if args.input else None,
        "split": split,
        "preset": args.preset,
        "text_columns": text_columns,
        "id_column": id_column,
        "num_documents": len(token_counts),
        "missing_text": missing_text,
        "tokenizer_kind": kind,
        "tokenizer": tokenizer_name,
        "statistics": {
            "char_length": compute_summary(char_lengths),
            "word_count": compute_summary(word_counts),
            "token_count": compute_summary(token_counts),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    save_plot(token_counts, plot_path, f"{prefix} Token Length Distribution")

    LOGGER.info("Saved CSV: %s", csv_path)
    LOGGER.info("Saved JSONL: %s", jsonl_path)
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved plot: %s", plot_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        validate_args(args)
        analyze(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
