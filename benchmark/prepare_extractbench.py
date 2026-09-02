"""Export digital-text ExtractBench documents to STAGE-compatible JSONL.

The Hugging Face dataset stores PDF paths and the task schema/output in its
Arrow splits.  PDFs are fetched lazily from the same dataset repository.  Rows
whose PDFs contain no extractable text are recorded as skipped, preventing an
OCR capability confound in the zero-shot evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PREFIX = """Extract the requested structured information from the document according to the JSON Schema.

=== Report ===
"""


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_pdf_text(pdf_path: str | Path) -> str:
    # PyMuPDF is substantially faster than pypdf on the very large PDFs in
    # ExtractBench while still yielding native text only (no OCR fallback).
    import pymupdf

    with pymupdf.open(str(pdf_path)) as document:
        return "\n\n".join(page.get_text("text").strip() for page in document).strip()


def make_record(row: dict[str, Any], pdf_text: str) -> dict[str, Any]:
    schema = json.loads(row["data_schema"])
    gold = json.loads(row["expected_output"])
    return {
        "stem": str(row["id"]).replace("/", "__"),
        "user_prompt": f"{PROMPT_PREFIX}{pdf_text}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}",
        "gold_json": json.dumps(gold, ensure_ascii=False),
        "json_schema": json.dumps(schema, ensure_ascii=False),
        "source_split": f"ExtractBench/{row['category']}",
        "source_pdf": str(row["pdf"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export digital-text ExtractBench PDFs to STAGE JSONL.")
    parser.add_argument("--dataset", default="llamaindex/ExtractBench")
    parser.add_argument("--splits", default="short,medium,long")
    parser.add_argument("--limit", type=int, default=None, help="Optional global row cap.")
    parser.add_argument("--output", default="benchmark/data/extractbench_digital.jsonl")
    parser.add_argument("--skipped-output", default="outputs/extractbench/skipped.jsonl")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    wanted_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    output = resolve_path(args.output)
    skipped_output = resolve_path(args.skipped_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped_output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as out_handle, skipped_output.open("w", encoding="utf-8") as skip_handle:
        for split in wanted_splits:
            dataset = load_dataset(args.dataset, split=split)
            for row in dataset:
                if args.limit is not None and written >= args.limit:
                    break
                row = dict(row)
                try:
                    pdf_path = hf_hub_download(args.dataset, row["pdf"], repo_type="dataset")
                    pdf_text = extract_pdf_text(pdf_path)
                    if not pdf_text:
                        raise ValueError("no digital text extracted")
                    record = make_record(row, pdf_text)
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:  # retain a reproducible exclusion record
                    skip_handle.write(
                        json.dumps(
                            {
                                "id": row.get("id"),
                                "category": row.get("category"),
                                "pdf": row.get("pdf"),
                                "reason": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    skipped += 1
            if args.limit is not None and written >= args.limit:
                break
    print(f"exported {written} digital-text rows: {output}")
    print(f"skipped {skipped} rows: {skipped_output}")


if __name__ == "__main__":
    main()
