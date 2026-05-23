#!/usr/bin/env python3
"""
Analyze SEC 10-K token lengths by standard Item section.

Example usage:
    python analyze_sec_10k_sections.py --max_samples 1000 --streaming
    python analyze_sec_10k_sections.py --input_csv outputs/sec_10k_text_sample.csv --text_column text
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


LOGGER = logging.getLogger("sec_10k_sections")

SECTION_ORDER = [
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "9C",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
]

SECTION_LABELS = {
    "1": "Item 1 Business",
    "1A": "Item 1A Risk Factors",
    "1B": "Item 1B Unresolved Comments",
    "1C": "Item 1C Cybersecurity",
    "2": "Item 2 Properties",
    "3": "Item 3 Legal Proceedings",
    "4": "Item 4 Mine Safety",
    "5": "Item 5 Market",
    "6": "Item 6 Selected Data",
    "7": "Item 7 MD&A",
    "7A": "Item 7A Market Risk",
    "8": "Item 8 Financial Statements",
    "9": "Item 9 Accounting Changes",
    "9A": "Item 9A Controls",
    "9B": "Item 9B Other Information",
    "9C": "Item 9C Foreign Jurisdictions",
    "10": "Item 10 Directors",
    "11": "Item 11 Compensation",
    "12": "Item 12 Security Ownership",
    "13": "Item 13 Related Transactions",
    "14": "Item 14 Accountant Fees",
    "15": "Item 15 Exhibits",
}

TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "document",
    "filing",
    "body",
    "report",
    "full_text",
    "raw_text",
)

ID_COLUMN_CANDIDATES = ("id", "accession_number", "accession", "filename", "file_name", "url", "cik")

SECTION_RE = re.compile(
    r"(?im)^[ \t]*(?:part[ \t]+[ivx]+[ \t]*)?"
    r"item[ \t]+"
    r"(1a|1b|1c|7a|9a|9b|9c|1[0-5]|[1-9])"
    r"[ \t]*[.\-:)]?[ \t]*"
    r"([^\n\r]{0,160})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure SEC 10-K section-level token lengths.")
    parser.add_argument("--dataset", default="PleIAs/SEC", help="HF dataset name.")
    parser.add_argument("--split", default=None, help="HF split. Defaults to first available split.")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming.")
    parser.add_argument("--input_csv", default=None, help="Optional local CSV containing full filing text.")
    parser.add_argument("--text_column", default=None, help="Text column override.")
    parser.add_argument("--id_column", default=None, help="ID column override.")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--min_section_chars", type=int, default=500, help="Discard shorter section spans as likely TOC noise.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--output_prefix", default="sec_10k_sections")
    parser.add_argument("--tiktoken_encoding", default="cl100k_base")
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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


def load_token_encoder(encoding_name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("tiktoken unavailable (%s); using ceil(word_count * 1.3).", exc)
        return None


def token_count(text: str, encoder: Any | None) -> int:
    if encoder is None:
        return int(math.ceil(len(text.split()) * 1.3))
    try:
        return len(encoder.encode(text))
    except Exception:  # noqa: BLE001
        return int(math.ceil(len(text.split()) * 1.3))


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
    LOGGER.info("Loading dataset=%s split=%s streaming=%s", args.dataset, split or "<default>", args.streaming)
    loaded = load_dataset(args.dataset, split=split, streaming=args.streaming) if split else load_dataset(args.dataset, streaming=args.streaming)
    if isinstance(loaded, dict):
        split = next(iter(loaded.keys()))
        loaded = loaded[split]
    iterator = iter(loaded)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Dataset is empty.") from exc
    columns = list(first.keys())
    LOGGER.info("Available columns: %s", ", ".join(columns))
    return columns, itertools.chain([first], iterator), split


def iter_csv_rows(path: Path) -> tuple[list[str], Iterator[dict[str, Any]], None]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
    LOGGER.info("Available columns: %s", ", ".join(columns))

    def iterator() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as f_in:
            yield from csv.DictReader(f_in)

    return columns, iterator(), None


def score_column(column: str, candidates: tuple[str, ...]) -> tuple[int, int]:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    exact = 100 if normalized in candidates else 0
    partial = max((50 + len(item) for item in candidates if item in normalized), default=0)
    return exact + partial, -len(normalized)


def choose_column(columns: list[str], override: str | None, candidates: tuple[str, ...], label: str) -> str | None:
    if override:
        if override not in columns:
            raise ValueError(f"--{label}_column {override!r} not found. Available: {columns}")
        return override
    ranked = sorted(columns, key=lambda column: score_column(column, candidates), reverse=True)
    if ranked and score_column(ranked[0], candidates)[0] > 0:
        return ranked[0]
    return None


def normalize_section(raw: str) -> str:
    return raw.upper()


def normalize_for_headings(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:div|p|tr|table|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    return text


def find_section_spans(text: str, min_section_chars: int) -> list[dict[str, Any]]:
    normalized = normalize_for_headings(text)
    matches = []
    for match in SECTION_RE.finditer(normalized):
        section = normalize_section(match.group(1))
        if section not in SECTION_ORDER:
            continue
        heading = " ".join(match.group(0).split())
        matches.append({"section": section, "heading": heading, "start": match.start(), "end": match.end()})

    spans_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, match in enumerate(matches):
        next_start = matches[index + 1]["start"] if index + 1 < len(matches) else len(normalized)
        section_text = normalized[match["end"] : next_start].strip()
        if len(section_text) < min_section_chars:
            continue
        spans_by_section[match["section"]].append(
            {
                "section": match["section"],
                "heading": match["heading"],
                "start": match["start"],
                "end": next_start,
                "text": section_text,
            }
        )

    selected: list[dict[str, Any]] = []
    for section in SECTION_ORDER:
        candidates = spans_by_section.get(section, [])
        if not candidates:
            continue
        # Prefer the longest span per section; this usually drops table-of-contents duplicates.
        selected.append(max(candidates, key=lambda item: len(item["text"])))
    return selected


def compute_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    p25, p50, p75, p90, p95, p99 = np.percentile(arr, [25, 50, 75, 90, 95, 99])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def save_section_plot(section_tokens: dict[str, list[int]], output_path: Path) -> None:
    sections = [section for section in SECTION_ORDER if section_tokens.get(section)]
    if not sections:
        LOGGER.warning("No sections found; skipping plot.")
        return
    data = [section_tokens[section] for section in sections]
    labels = [section for section in sections]
    plt.figure(figsize=(13, 6))
    plt.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    plt.yscale("log")
    plt.xlabel("10-K Item section")
    plt.ylabel("Token count (log scale)")
    plt.title("SEC 10-K Section-Level Token Lengths")
    plt.grid(True, which="major", axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def analyze(args: argparse.Namespace) -> None:
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.min_section_chars < 0:
        raise ValueError("--min_section_chars must be non-negative")

    if args.input_csv:
        columns, rows_iter, split = iter_csv_rows(Path(args.input_csv))
    else:
        columns, rows_iter, split = iter_hf_rows(args)
    text_column = choose_column(columns, args.text_column, TEXT_COLUMN_CANDIDATES, "text")
    if text_column is None:
        raise ValueError(f"Could not infer text column. Available: {columns}. Pass --text_column.")
    id_column = choose_column(columns, args.id_column, ID_COLUMN_CANDIDATES, "id")
    LOGGER.info("Using text_column=%s id_column=%s", text_column, id_column or "<row_index>")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}_lengths.csv"
    jsonl_path = output_dir / f"{args.output_prefix}_lengths.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    plot_path = output_dir / f"{args.output_prefix}_boxplot.png"

    encoder = load_token_encoder(args.tiktoken_encoding)
    section_tokens: dict[str, list[int]] = defaultdict(list)
    docs_with_sections = 0
    total_docs = 0
    fieldnames = [
        "row_index",
        "source_id",
        "section",
        "section_label",
        "heading",
        "char_length",
        "word_count",
        "token_count",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(tqdm(rows_iter, total=args.max_samples, desc="Analyzing sections", unit="doc")):
            if row_index >= args.max_samples:
                break
            total_docs += 1
            source_id = coerce_text(row.get(id_column)) if id_column else str(row_index)
            text = coerce_text(row.get(text_column))
            spans = find_section_spans(text, args.min_section_chars)
            if spans:
                docs_with_sections += 1
            for span in spans:
                section_text = span["text"]
                word_count = len(section_text.split())
                row_out = {
                    "row_index": row_index,
                    "source_id": source_id or str(row_index),
                    "section": span["section"],
                    "section_label": SECTION_LABELS.get(span["section"], f"Item {span['section']}"),
                    "heading": span["heading"],
                    "char_length": len(section_text),
                    "word_count": word_count,
                    "token_count": token_count(section_text, encoder),
                }
                writer.writerow(row_out)
                jsonl_file.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                section_tokens[span["section"]].append(int(row_out["token_count"]))

    summary = {
        "dataset": args.dataset if not args.input_csv else None,
        "input_csv": args.input_csv,
        "split": split,
        "text_column": text_column,
        "id_column": id_column,
        "max_samples": args.max_samples,
        "num_documents": total_docs,
        "documents_with_any_section": docs_with_sections,
        "section_detection_note": "Regex-based Item heading extraction; short spans are filtered to reduce table-of-contents noise.",
        "sections": {
            section: {
                "label": SECTION_LABELS.get(section, f"Item {section}"),
                "statistics": compute_summary(section_tokens.get(section, [])),
            }
            for section in SECTION_ORDER
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    save_section_plot(section_tokens, plot_path)

    LOGGER.info("Saved section CSV: %s", csv_path)
    LOGGER.info("Saved section JSONL: %s", jsonl_path)
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved plot: %s", plot_path)
    LOGGER.info("Documents with detected sections: %s/%s", docs_with_sections, total_docs)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        analyze(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
