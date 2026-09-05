"""OCR the SROIE receipts mirrored on the Hub (rajistics/sroie: image + Donut target string) and
export them to STAGE JSONL with the four SROIE fields (company, date, address, total).

Gold comes from the Donut sequence ``<s_total>..</s_total><s_date>..`` in the ``text`` column;
the report is tesseract text (psm 6) of the receipt image, so this is a real-world scanned
receipt task with OCR noise, comparable to CORD but with English receipts.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["company", "date", "address", "total"]
DESC = {"company": "Name of the store or company issuing the receipt, as printed", "date": "Date on the receipt, as printed",
        "address": "Address of the store, as printed", "total": "Total amount paid, as printed"}
PROMPT_PREFIX = "Extract the receipt fields from the OCR text according to the JSON Schema. Copy values as they appear. Return exactly one JSON object.\n\n"


def parse_target(t: str) -> dict:
    out = {}
    for k in FIELDS:
        m = re.search(rf"<s_{k}>(.*?)</s_{k}>", t, re.S)
        if m:
            out[k] = " ".join(m.group(1).split())
    return out


def ocr(img_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        from PIL import Image
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(f.name)
        return subprocess.run(["tesseract", f.name, "stdout", "--psm", "6", "-l", "eng"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="rajistics/sroie")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "sroie_ocr.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    p = snapshot_download(a.repo, repo_type="dataset", cache_dir=a.cache_dir)
    rows = []
    for f in sorted(glob.glob(p + "/**/*.parquet", recursive=True)):
        rows += pq.read_table(f).to_pylist()
    schema = {"type": "object", "additionalProperties": False, "required": FIELDS, "properties": {k: {"type": "string", "description": DESC[k]} for k in FIELDS}}
    with ThreadPoolExecutor(a.workers) as ex:
        texts = list(ex.map(lambda r: ocr(r["image"]["bytes"]), rows))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.output.open("w", encoding="utf-8") as fh:
        for i, (r, text) in enumerate(zip(rows, texts)):
            gold = parse_target(r["text"])
            if len(gold) < 4 or len(text) < 50:
                continue
            prompt = f"{PROMPT_PREFIX}=== Report ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(schema, indent=2)}"
            fh.write(json.dumps({"stem": f"sroie_{i:04d}", "dataset": "sroie_ocr", "user_prompt": prompt, "gold_json": json.dumps(gold, ensure_ascii=False),
                                 "json_schema": json.dumps(schema), "ocr_chars": len(text)}, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} receipts (of {len(rows)}) to {a.output}")


if __name__ == "__main__":
    main()
