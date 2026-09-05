"""OCR the ExtractBench PDFs that have no digital text layer and export them to STAGE JSONL.

Complements prepare_extractbench.py (digital-text rows). Pages are rasterised with
``pdftoppm`` (200 dpi) and read with ``tesseract`` (eng, psm 3); page texts are joined in
order. Rows come from the HF Arrow splits; PDFs from the HF cache. Documents whose OCR
text exceeds --max-chars are skipped so prompts fit a 131k-token YaRN context.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from prepare_extractbench import make_record  # noqa: E402  (same directory)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ocr_pdf(pdf_path: str, dpi: int = 200) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", pdf_path, f"{tmp}/p"], check=True, capture_output=True)
        pages = sorted(Path(tmp).glob("p-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        texts = []
        for png in pages:
            out = subprocess.run(["tesseract", str(png), "stdout", "--psm", "3", "-l", "eng"], capture_output=True, text=True)
            texts.append(out.stdout.strip())
    return "\n\n".join(t for t in texts if t)


def work(args):
    row, pdf_path = args
    try:
        return row, ocr_pdf(pdf_path), None
    except Exception as exc:  # noqa: BLE001
        return row, "", str(exc)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="llamaindex/ExtractBench")
    ap.add_argument("--digital", default="benchmark/data/extractbench_digital.jsonl", help="rows already covered by digital text (skipped here)")
    ap.add_argument("--output", default="benchmark/data/extractbench_ocr.jsonl")
    ap.add_argument("--max-chars", type=int, default=400000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    done = {json.loads(l)["stem"] for l in open(PROJECT_ROOT / a.digital, encoding="utf-8") if l.strip()}
    jobs = []
    for split in ("short", "medium", "long"):
        for row in load_dataset(a.dataset, "extract-bench", split=split):
            stem = str(row["id"]).replace("/", "__")
            if stem in done:
                continue
            jobs.append((row, hf_hub_download(a.dataset, row["pdf"], repo_type="dataset")))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"OCR jobs: {len(jobs)}", flush=True)
    out = PROJECT_ROOT / a.output; out.parent.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, []
    with out.open("w", encoding="utf-8") as fh, ProcessPoolExecutor(a.workers) as ex:
        for i, (row, text, err) in enumerate(ex.map(work, jobs)):
            if err or len(text) < 200:
                skipped.append({"id": row["id"], "reason": err or "empty OCR"}); continue
            if len(text) > a.max_chars:
                skipped.append({"id": row["id"], "reason": f"ocr text {len(text)} chars > max"}); continue
            rec = make_record(row, text); rec["text_source"] = "tesseract-4.1.1-200dpi"; rec["ocr_chars"] = len(text)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            if (i + 1) % 10 == 0:
                print(f"progress {i + 1}/{len(jobs)} written={n}", flush=True)
    (out.with_suffix(".skipped.json")).write_text(json.dumps(skipped, indent=2))
    print(f"OCR_DONE wrote {n} rows to {out}; skipped {len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
