"""Export SWDE (hazyresearch/based-swde, validation) to STAGE benchmark JSONL.

Real web pages (movie / university sites) flattened to text, with verbatim gold attribute
values (one row per doc x key in the source). Rows are grouped per document into one
schema whose properties are that document's annotated keys (all required, string).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PREFIX = {"based-swde": "Extract the requested attributes from the web page text according to the JSON Schema. Copy values exactly as they appear in the page.\n\n",
                 "based-fda": "Extract the requested fields from this excerpt of an FDA 510(k) decision summary according to the JSON Schema. Copy values exactly as they appear in the text.\n\n"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hazyresearch/based-swde")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    tag = a.repo.split("/")[-1].replace("based-", "")
    if a.output is None:
        a.output = ROOT / "benchmark" / "data" / "realworld" / f"{tag}_validation.jsonl"
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    p = snapshot_download(a.repo, repo_type="dataset", cache_dir=a.cache_dir)
    rows = []
    for f in sorted(glob.glob(p + "/**/*.parquet", recursive=True)):
        rows += pq.read_table(f).to_pylist()
    # doc_id / file_name are NOT unique across sites in based-swde (e.g. id0484 is both a movie page and a
    # university page), so documents are grouped by their page text.
    import hashlib
    docs: dict[str, dict] = collections.OrderedDict()
    for r in rows:
        h = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:12]
        d = docs.setdefault(h, {"text": r["text"], "file_name": r["file_name"], "doc_id": r["doc_id"], "kv": collections.OrderedDict()})
        d["kv"][r["key"]] = r["value"]
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with a.output.open("w", encoding="utf-8") as fh:
        for i, (doc_id, d) in enumerate(docs.items()):
            schema = {"type": "object", "additionalProperties": False,
                      "properties": {k: {"type": "string", "description": f"The {k} as printed"} for k in d["kv"]},
                      "required": list(d["kv"])}
            text = d["text"].lstrip("﻿")
            prompt = f"{PROMPT_PREFIX.get(a.repo.split('/')[-1], PROMPT_PREFIX['based-swde'])}=== Report ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            rec = {"stem": f"{tag}_{i:04d}", "dataset": f"{tag}_validation", "source_id": f"{d['doc_id']}:{doc_id}", "user_prompt": prompt,
                   "gold_json": json.dumps(dict(d["kv"]), ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False)}
            if tok is not None:
                n = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = n; lengths.append(n)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} docs ({len(rows)} gold values) to {a.output}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}")


if __name__ == "__main__":
    main()
