#!/usr/bin/env python3
"""
Analyze SEC 10-K document length distributions from Hugging Face.

Example usage:
    python analyze_sec_10k_lengths.py --max_samples 5000 --streaming
    python analyze_sec_10k_lengths.py --max_samples 10000
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm


LOGGER = logging.getLogger("sec_10k_lengths")

TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "document",
    "filing",
    "body",
    "report",
    "article",
    "html",
    "markdown",
    "plain_text",
    "full_text",
    "raw_text",
)

ID_COLUMN_CANDIDATES = (
    "id",
    "accession_number",
    "accession",
    "filename",
    "file_name",
    "url",
    "cik",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/sample PleIAs/SEC from Hugging Face and compute per-document "
            "character, word, and approximate token length statistics."
        )
    )
    parser.add_argument("--dataset", default="PleIAs/SEC", help="Hugging Face dataset name.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to first available split.")
    parser.add_argument("--text_column", default=None, help="Override the automatically selected text column.")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum documents to analyze.")
    parser.add_argument("--streaming", action="store_true", help="Use Hugging Face streaming mode.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for CSV/JSONL/summary/plot outputs.")
    parser.add_argument("--output_prefix", default="sec_10k", help="Prefix for output file names.")
    parser.add_argument(
        "--tiktoken_encoding",
        default="cl100k_base",
        help="tiktoken encoding name used for approximate token counts.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be a positive integer")


def load_token_encoder(encoding_name: str):
    try:
        import tiktoken
    except ImportError:
        LOGGER.warning("tiktoken is not installed; falling back to ceil(word_count * 1.3).")
        return None

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001 - keep CLI robust across local tiktoken installs.
        LOGGER.warning(
            "Could not load tiktoken encoding %r (%s); falling back to ceil(word_count * 1.3).",
            encoding_name,
            exc,
        )
        return None


def infer_split(dataset_name: str, requested_split: str | None, streaming: bool) -> str | None:
    if requested_split:
        return requested_split

    try:
        from datasets import get_dataset_split_names

        splits = get_dataset_split_names(dataset_name)
        if splits:
            LOGGER.info("Available splits: %s", ", ".join(splits))
            LOGGER.info("No --split provided; using first available split: %s", splits[0])
            return splits[0]
    except Exception as exc:  # noqa: BLE001 - load_dataset fallback below is more useful to users.
        LOGGER.warning("Could not inspect split names before loading dataset: %s", exc)

    LOGGER.info("No split could be inferred; letting datasets.load_dataset choose the default.")
    return None if streaming else "train"


def load_hf_dataset(dataset_name: str, split: str | None, streaming: bool) -> Dataset | IterableDataset:
    LOGGER.info("Loading dataset=%s split=%s streaming=%s", dataset_name, split or "<default>", streaming)
    try:
        if split is None:
            loaded = load_dataset(dataset_name, streaming=streaming)
        else:
            loaded = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset {dataset_name!r}. Check your network, Hugging Face access, "
            f"and the requested split {split!r}."
        ) from exc

    if isinstance(loaded, dict):
        first_split = next(iter(loaded.keys()))
        LOGGER.info("Loaded split dictionary; using first split: %s", first_split)
        return loaded[first_split]
    return loaded


def peek_dataset(dataset: Dataset | IterableDataset) -> tuple[list[str], Iterator[dict[str, Any]]]:
    iterator = iter(dataset)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Dataset split is empty.") from exc

    if not isinstance(first, dict):
        raise TypeError(f"Expected dataset rows to be dictionaries, got {type(first).__name__}.")

    columns = list(first.keys())
    LOGGER.info("Available columns: %s", ", ".join(columns))
    return columns, itertools.chain([first], iterator)


def score_text_column(column: str) -> tuple[int, int]:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    exact_score = 100 if normalized in TEXT_COLUMN_CANDIDATES else 0
    partial_score = 0
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            partial_score = max(partial_score, 50 + len(candidate))
    return exact_score + partial_score, -len(normalized)


def choose_text_column(columns: list[str], override: str | None) -> str:
    if override:
        if override not in columns:
            raise ValueError(f"--text_column {override!r} was not found. Available columns: {columns}")
        LOGGER.info("Using text column override: %s", override)
        return override

    ranked = sorted(columns, key=score_text_column, reverse=True)
    if ranked and score_text_column(ranked[0])[0] > 0:
        LOGGER.info("Automatically selected text column: %s", ranked[0])
        return ranked[0]

    raise ValueError(
        "Could not automatically identify a text column. "
        f"Available columns: {columns}. Re-run with --text_column COLUMN_NAME."
    )


def choose_id_column(columns: list[str]) -> str | None:
    normalized_to_original = {
        re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_"): column for column in columns
    }
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    return None


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def whitespace_word_count(text: str) -> int:
    return len(text.split())


def approximate_token_count(text: str, word_count: int, encoder: Any | None) -> int:
    if encoder is None:
        return int(math.ceil(word_count * 1.3))
    try:
        return len(encoder.encode(text))
    except Exception as exc:  # noqa: BLE001 - one pathological document should not stop the run.
        LOGGER.warning("tiktoken failed on one document (%s); using word-count fallback.", exc)
        return int(math.ceil(word_count * 1.3))


def compute_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def is_highly_skewed(values: list[int]) -> bool:
    positive = [value for value in values if value > 0]
    if len(positive) < 2:
        return False
    p50, p95 = np.percentile(np.asarray(positive, dtype=np.float64), [50, 95])
    return bool(p50 > 0 and p95 / p50 >= 5)


def save_token_plot(token_counts: list[int], output_path: Path) -> None:
    if not token_counts:
        LOGGER.warning("No token counts available; skipping plot.")
        return

    positive = np.asarray([value for value in token_counts if value > 0], dtype=np.float64)
    if positive.size == 0:
        LOGGER.warning("All token counts are zero; skipping plot.")
        return

    use_log_x = is_highly_skewed(token_counts)
    plt.figure(figsize=(10, 6))

    if use_log_x:
        if positive.min() == positive.max():
            bins = 10
        else:
            bins = np.logspace(np.log10(max(1, positive.min())), np.log10(positive.max()), 60)
        plt.hist(positive, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
        plt.xscale("log")
        plt.xlabel("Approximate token count (log scale)")
    else:
        plt.hist(positive, bins=60, color="#4C78A8", alpha=0.85, edgecolor="white")
        plt.xlabel("Approximate token count")

    plt.ylabel("Number of documents")
    plt.title("SEC 10-K Token Length Distribution")
    plt.grid(True, which="major", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    LOGGER.info("Saved token length distribution plot: %s", output_path)


def analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{args.output_prefix}_lengths.csv"
    jsonl_path = output_dir / f"{args.output_prefix}_lengths.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_length_summary.json"
    plot_path = output_dir / f"{args.output_prefix}_token_length_distribution.png"

    split = infer_split(args.dataset, args.split, args.streaming)
    dataset = load_hf_dataset(args.dataset, split, args.streaming)
    columns, rows_iter = peek_dataset(dataset)
    text_column = choose_text_column(columns, args.text_column)
    id_column = choose_id_column(columns)

    encoder = load_token_encoder(args.tiktoken_encoding)
    char_lengths: list[int] = []
    word_counts: list[int] = []
    token_counts: list[int] = []
    skipped_missing_text = 0

    fieldnames = [
        "row_index",
        "source_id",
        "char_length",
        "word_count",
        "token_count",
        "text_column",
        "split",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        progress = tqdm(
            rows_iter,
            total=args.max_samples,
            desc="Analyzing SEC documents",
            unit="doc",
        )
        for row_index, row in enumerate(progress):
            if row_index >= args.max_samples:
                break
            raw_text = row.get(text_column)
            text = coerce_text(raw_text)
            if not text:
                skipped_missing_text += 1

            char_length = len(text)
            word_count = whitespace_word_count(text)
            token_count = approximate_token_count(text, word_count, encoder)
            source_id = str(row.get(id_column, row_index)) if id_column else str(row_index)

            result = {
                "row_index": row_index,
                "source_id": source_id,
                "char_length": char_length,
                "word_count": word_count,
                "token_count": token_count,
                "text_column": text_column,
                "split": split or "",
            }
            writer.writerow(result)
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            char_lengths.append(char_length)
            word_counts.append(word_count)
            token_counts.append(token_count)

    summary = {
        "dataset": args.dataset,
        "split": split,
        "text_column": text_column,
        "id_column": id_column,
        "streaming": bool(args.streaming),
        "max_samples": int(args.max_samples),
        "num_documents": len(token_counts),
        "skipped_missing_text": skipped_missing_text,
        "token_count_method": f"tiktoken:{args.tiktoken_encoding}" if encoder is not None else "ceil(word_count * 1.3)",
        "statistics": {
            "char_length": compute_summary(char_lengths),
            "word_count": compute_summary(word_counts),
            "token_count": compute_summary(token_counts),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    save_token_plot(token_counts, plot_path)

    LOGGER.info("Saved per-document CSV: %s", csv_path)
    LOGGER.info("Saved per-document JSONL: %s", jsonl_path)
    LOGGER.info("Saved summary JSON: %s", summary_path)
    LOGGER.info("Analyzed %s documents; missing/empty text rows: %s", len(token_counts), skipped_missing_text)


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
