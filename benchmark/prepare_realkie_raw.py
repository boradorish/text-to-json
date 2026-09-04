"""Export the raw RealKIE releases (Indico, Wasabi bucket ``project-fruitfly``) to STAGE JSONL.

Subsets: ``charities`` (UK charity annual reports, 28 classes), ``nda`` (3 classes),
``resource_contracts`` (23 classes, 50k+ tokens), ``s1_pages`` (SEC S-1 pages, 24 classes),
``fcc_invoices`` (11 classes; the verified HF version is used instead in prepare_realkie.py).
Every class becomes an array-of-strings field holding the unique verbatim spans (order of
first appearance); classes absent from a document get an empty array. Labels are character
spans over the ``text`` column (offsets verified to match ``text``). Input: ``<src>/<subset>_<split>.csv``
downloaded from https://s3.us-east-2.wasabisys.com/project-fruitfly/<subset>/<split>.csv.
Score with benchmark/score_cuad.py (generic array-of-span scorer).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(10**9)
DOC_KIND = {"charities": "charity annual report", "nda": "non-disclosure agreement", "resource_contracts": "natural-resource contract",
            "s1_pages": "page of an SEC Form S-1 registration statement", "fcc_invoices": "FCC political-advertising invoice"}
PROMPT = ("Extract the labeled fields from this {kind} according to the JSON Schema. For every field, copy the exact text of "
          "each span in the document that matches the field name (one array element per distinct span); use an empty array when "
          "the document has no such span. Return exactly one JSON object.\n\n")


def key_of(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/mnt/nvme/cache/interns/realkie_raw/csv"))
    ap.add_argument("--subset", required=True, choices=list(DOC_KIND))
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = a.output or ROOT / "benchmark" / "data" / "realworld" / f"realkie_{a.subset}_{a.split}.jsonl"
    rows = list(csv.DictReader((a.src / f"{a.subset}_{a.split}.csv").open(encoding="utf-8")))
    classes: list[str] = []
    for r in rows:
        for l in json.loads(r["labels"]):
            if l["label"] not in classes:
                classes.append(l["label"])
    classes.sort()
    if a.max_docs and a.max_docs < len(rows):
        rows = random.Random(a.seed).sample(rows, a.max_docs)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    props = {key_of(c): {"type": "array", "description": f"{c}: every distinct span labeled '{c}', verbatim", "items": {"type": "string"}} for c in classes}
    schema = {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}
    out.parent.mkdir(parents=True, exist_ok=True)
    lengths, n_spans, bad = [], 0, 0
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            text = r["text"]
            gold = {k: [] for k in props}
            for l in json.loads(r["labels"]):
                span = " ".join(l["text"].split())
                if text[l["start"]:l["end"]] != l["text"]:
                    bad += 1
                k = key_of(l["label"])
                if span and span not in gold[k]:
                    gold[k].append(span); n_spans += 1
            prompt = f"{PROMPT.format(kind=DOC_KIND[a.subset])}=== Report ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            rec = {"stem": f"realkie_{a.subset}_{i:04d}", "dataset": f"realkie_{a.subset}_{a.split}", "source_id": r.get("original_filename") or r.get("document_path"),
                   "user_prompt": prompt, "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False),
                   "text_chars": len(text), "n_labels": len(json.loads(r["labels"]))}
            if tok is not None:
                n = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = n; lengths.append(n)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} docs, {len(classes)} classes, {n_spans} unique gold spans ({bad} offset mismatches) to {out}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}; >32768: {sum(l>32768 for l in lengths)}; >36864: {sum(l>36864 for l in lengths)}; >126976: {sum(l>126976 for l in lengths)}")


if __name__ == "__main__":
    main()
