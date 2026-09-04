"""Export CUAD (Hendrycks et al., 2021) contracts to STAGE benchmark JSONL.

Each contract becomes one document with a 41-field schema (one array-of-verbatim-clause
field per CUAD category, empty array when the category is absent). Gold spans are the
CUAD answer texts (multiple spans per category are kept). Input is the SQuAD-style
``test.json`` (102 contracts) or ``CUADv1.json`` (510 contracts) from the CUAD release.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PREFIX = (
    "Extract the clauses of this contract according to the JSON Schema. For every field, copy the exact text "
    "of every clause in the contract that matches the field description (one array element per clause); "
    "use an empty array when the contract has no such clause. Return exactly one JSON object.\n\n"
)


def key_of(cat: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="CUAD test.json or CUADv1.json")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "cuad_test.jsonl")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    a = ap.parse_args()
    data = json.load(a.src.open(encoding="utf-8"))["data"]
    if a.max_docs:
        data = data[: a.max_docs]
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lengths, n_spans = [], 0
    with a.output.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(data):
            p = c["paragraphs"][0]
            props, gold = {}, {}
            for q in p["qas"]:
                m = re.search(r'related to "([^"]+)"', q["question"])
                cat = m.group(1) if m else q["question"]
                desc = q["question"].split("Details:", 1)[1].strip() if "Details:" in q["question"] else cat
                k = key_of(cat)
                props[k] = {"type": "array", "description": f"{cat}: {desc}", "items": {"type": "string"}}
                spans = []
                for ans in q["answers"]:
                    t = " ".join(ans["text"].split())
                    if t and t not in spans:
                        spans.append(t)
                gold[k] = spans; n_spans += len(spans)
            schema = {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}
            text = p["context"]
            prompt = f"{PROMPT_PREFIX}=== Report ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            rec = {"stem": f"cuad_{i:03d}", "dataset": "cuad_test", "source_id": c["title"], "user_prompt": prompt,
                   "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False),
                   "context_chars": len(text)}
            if tok is not None:
                n = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = n; lengths.append(n)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(data)} contracts ({n_spans} gold spans) to {a.output}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}; >32768: {sum(l>32768 for l in lengths)}; >131072-4096: {sum(l>131072-4096 for l in lengths)}")


if __name__ == "__main__":
    main()
