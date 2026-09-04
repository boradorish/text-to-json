"""Export Kleister Charity dev-0 (applicaai/kleister-charity) to STAGE benchmark JSONL.

Long UK charity annual reports (OCR text, median ~7k tokens, tail > 60k) with up to
eight gold fields per document. Values keep Kleister's canonical form (spaces as
underscores, ISO dates, plain decimals) so the gold is comparable across models.
Inputs are the ``dev-0/in.tsv`` (text_best column) and ``dev-0/expected.tsv`` files
from the GitHub repository; PDFs are not needed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESC = {
    "address__post_town": "Post town of the charity's address, upper case as printed",
    "address__postcode": "UK postcode of the charity's address; write the space as an underscore",
    "address__street_line": "Street line of the charity's address; write spaces as underscores",
    "charity_name": "Registered name of the charity; write spaces as underscores",
    "charity_number": "Registered charity number (digits only)",
    "income_annually_in_british_pounds": "Total annual income in GBP as a plain decimal, e.g. 10348000.00",
    "report_date": "End date of the reporting period in ISO format YYYY-MM-DD",
    "spending_annually_in_british_pounds": "Total annual expenditure in GBP as a plain decimal, e.g. 9415000.00",
}
PROMPT_PREFIX = (
    "Extract only information supported by the source document according to the JSON Schema.\n"
    "Return exactly one JSON object and no explanation.\n\n"
)


def parse_expected(line: str) -> dict[str, str]:
    out = {}
    for tok in line.strip().split(" "):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/mnt/nvme/cache/interns/kleister_charity"))
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "kleister_charity_dev-0.jsonl")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    csv.field_size_limit(10**9)
    hdr = (a.src / "in-header.tsv").read_text().strip().split("\t")
    rows = list(csv.reader((a.src / "in.tsv").open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    gold_lines = (a.src / "expected.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(gold_lines), (len(rows), len(gold_lines))
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with a.output.open("w", encoding="utf-8") as fh:
        for i, (r, g) in enumerate(zip(rows, gold_lines)):
            rec = dict(zip(hdr, r))
            keys = rec["keys"].split(" ")
            gold = parse_expected(g)
            schema = {"type": "object", "additionalProperties": False,
                      "properties": {k: {"type": "string", "description": DESC[k]} for k in keys},
                      "required": [k for k in keys if k in gold]}
            text = rec["text_best"]
            prompt = f"{PROMPT_PREFIX}=== Source document ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            out = {"stem": f"kleister_charity_{i:03d}", "dataset": "kleister_charity_dev-0", "source_id": rec["filename"],
                   "user_prompt": prompt, "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False)}
            if tok is not None:
                n = len(tok(prompt)["input_ids"]); out["prompt_tokens"] = n; lengths.append(n)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {a.output}")
    if lengths:
        lengths.sort()
        print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}; >8192: {sum(l>8192 for l in lengths)}; fit 40960-3100: {sum(l+3100<=40960 for l in lengths)}")


if __name__ == "__main__":
    main()
