"""Export RealKIE-FCC-Verified (amazon-agi) to the STAGE benchmark JSONL format.

75 multi-page FCC political-ad invoices with OCR text, one shared JSON Schema
(header fields + nested LineItems), and a verified gold JSON. No OCR engine is
needed: the ``text`` column is used as the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PREFIX = "Extract the invoice fields from the document text according to the JSON Schema.\n\n"


def fix_types(node):
    """RealKIE uses the non-standard JSON Schema type "float"; map it to "number" (and "int" to "integer")."""
    if isinstance(node, dict):
        if node.get("type") == "float":
            node["type"] = "number"
        elif node.get("type") == "int":
            node["type"] = "integer"
        for v in node.values():
            fix_types(v)
    elif isinstance(node, list):
        for v in node:
            fix_types(v)
    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="amazon-agi/RealKIE-FCC-Verified")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "realkie_fcc_verified.jsonl")
    ap.add_argument("--tokenizer", default=None, help="Optional tokenizer to record prompt token lengths")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    p = hf_hub_download(a.repo, "data/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(p).to_pylist()
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with a.output.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            schema = json.loads(r["json_schema"]) if isinstance(r["json_schema"], str) else r["json_schema"]
            schema = fix_types(schema)
            gold = json.loads(r["json_response"]) if isinstance(r["json_response"], str) else r["json_response"]
            prompt = f"{PROMPT_PREFIX}=== Report ===\n{r['text']}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            rec = {"stem": f"realkie_fcc_{i:03d}", "source_id": r["id"], "user_prompt": prompt,
                   "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False),
                   "source_split": "RealKIE-FCC-Verified/test", "pages": len(r["image_files"]) if isinstance(r["image_files"], list) else None}
            if tok is not None:
                n = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = n; lengths.append(n)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {a.output}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}; >8192: {sum(l>8192 for l in lengths)}, >32768: {sum(l>32768 for l in lengths)}")


if __name__ == "__main__":
    main()
