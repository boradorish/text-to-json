#!/usr/bin/env python3
"""
Analyze ContextualAI/extract-bench PDF-to-JSON input lengths.

Example usage:
    git clone --depth 1 https://github.com/ContextualAI/extract-bench.git /tmp/extract-bench
    python analyze_extract_bench_lengths.py --repo_dir /tmp/extract-bench
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "text_to_json_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
from tqdm import tqdm


LOGGER = logging.getLogger("extract_bench_lengths")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ExtractBench PDF/schema input lengths.")
    parser.add_argument("--repo_dir", required=True, help="Path to cloned ContextualAI/extract-bench repo.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--output_prefix", default="extract_bench")
    parser.add_argument("--tiktoken_encoding", default="cl100k_base")
    parser.add_argument("--include_schema", action="store_true", default=True)
    parser.add_argument("--pdf_only", action="store_true", help="Measure only extracted PDF text as token_count.")
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_encoder(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
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


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def find_tasks(repo_dir: Path) -> list[dict[str, Path | str]]:
    dataset_dir = repo_dir / "dataset"
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset directory not found: {dataset_dir}")

    tasks: list[dict[str, Path | str]] = []
    for schema_path in sorted(dataset_dir.glob("*/*/*-schema.json")):
        task_dir = schema_path.parent
        pdf_gold_dir = task_dir / "pdf+gold"
        if not pdf_gold_dir.is_dir():
            continue
        domain = task_dir.parent.name
        schema_name = task_dir.name
        for pdf_path in sorted(pdf_gold_dir.glob("*.pdf")):
            gold_path = pdf_path.with_suffix(".gold.json")
            tasks.append(
                {
                    "domain": domain,
                    "schema_name": schema_name,
                    "schema_path": schema_path,
                    "pdf_path": pdf_path,
                    "gold_path": gold_path,
                }
            )
    return tasks


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


def save_plot(values: list[int], output_path: Path) -> None:
    positive = np.asarray([value for value in values if value > 0], dtype=np.float64)
    if positive.size == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(positive, bins=30, color="#4C78A8", alpha=0.85, edgecolor="white")
    plt.xlabel("Token count")
    plt.ylabel("Number of documents")
    plt.title("ExtractBench Input Token Length Distribution")
    plt.grid(True, which="major", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def analyze(args: argparse.Namespace) -> None:
    repo_dir = Path(args.repo_dir).resolve()
    tasks = find_tasks(repo_dir)
    if not tasks:
        raise ValueError(f"No ExtractBench PDF tasks found under {repo_dir}")
    LOGGER.info("Found %s PDF tasks", len(tasks))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}_lengths.csv"
    jsonl_path = output_dir / f"{args.output_prefix}_lengths.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_length_summary.json"
    plot_path = output_dir / f"{args.output_prefix}_token_length_distribution.png"
    encoder = load_encoder(args.tiktoken_encoding)

    rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    pdf_token_counts: list[int] = []
    schema_token_counts: list[int] = []

    for idx, task in enumerate(tqdm(tasks, desc="ExtractBench PDFs", unit="pdf")):
        pdf_path = Path(task["pdf_path"])
        schema_path = Path(task["schema_path"])
        pdf_text = extract_pdf_text(pdf_path)
        schema_text = json.dumps(json.loads(schema_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        measured_text = pdf_text if args.pdf_only else f"{pdf_text}\n\n=== JSON Schema ===\n{schema_text}"

        pdf_tokens = token_count(pdf_text, encoder)
        schema_tokens = token_count(schema_text, encoder)
        measured_tokens = token_count(measured_text, encoder)
        row = {
            "row_index": idx,
            "source_id": pdf_path.stem,
            "domain": task["domain"],
            "schema_name": task["schema_name"],
            "pdf_path": str(pdf_path),
            "schema_path": str(schema_path),
            "gold_path": str(task["gold_path"]),
            "char_length": len(measured_text),
            "word_count": len(measured_text.split()),
            "token_count": measured_tokens,
            "pdf_token_count": pdf_tokens,
            "schema_token_count": schema_tokens,
            "measurement": "pdf_only" if args.pdf_only else "pdf_text_plus_schema",
        }
        rows.append(row)
        token_counts.append(measured_tokens)
        pdf_token_counts.append(pdf_tokens)
        schema_token_counts.append(schema_tokens)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": "ContextualAI/extract-bench",
        "repo_dir": str(repo_dir),
        "measurement": "pdf_only" if args.pdf_only else "pdf_text_plus_schema",
        "num_documents": len(rows),
        "statistics": {
            "token_count": compute_summary(token_counts),
            "pdf_token_count": compute_summary(pdf_token_counts),
            "schema_token_count": compute_summary(schema_token_counts),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    save_plot(token_counts, plot_path)

    LOGGER.info("Saved CSV: %s", csv_path)
    LOGGER.info("Saved JSONL: %s", jsonl_path)
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved plot: %s", plot_path)


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
