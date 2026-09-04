"""Export VRDU (Google Research, 2023) ad-buy-form / registration-form to STAGE benchmark JSONL.

Real-world scanned forms with OCR text and verbatim gold spans. ``ad-buy-form`` (DeepForm,
641 FCC political-ad invoices) has nine header entities and repeated ``line_item`` groups
(channel, program_desc, program_start_date, program_end_date, sub_amount); ``registration-form``
(FARA, 1,915 docs) has six flat entities. Every gold entity may appear several times in the
document; all occurrences are kept in ``gold_alts`` so the scorer accepts any of them.
Input: ``<src>/<subset>/dataset.jsonl`` + ``meta.json`` (gunzipped from the GitHub release).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE_KEYS = ["channel", "program_desc", "program_start_date", "program_end_date", "sub_amount"]
DESC = {
    "advertiser": "Name of the advertiser (the political committee or company buying the ads), as printed",
    "agency": "Name of the advertising agency placing the order, as printed",
    "contract_num": "Contract / order / invoice number, as printed",
    "flight_from": "Start date of the advertising flight, as printed",
    "flight_to": "End date of the advertising flight, as printed",
    "gross_amount": "Gross (total) amount of the order, as printed (e.g. $12,345.00)",
    "product": "Product or campaign being advertised, as printed",
    "tv_address": "Postal address of the TV station or its remit address, as printed",
    "property": "TV station / property call sign, as printed",
    "channel": "Channel or station on which the spot airs, as printed",
    "program_desc": "Program or daypart description of the spot, as printed",
    "program_start_date": "Start date of the spot's run, as printed",
    "program_end_date": "End date of the spot's run, as printed",
    "sub_amount": "Amount charged for this line item, as printed",
    "registration_num": "FARA registration number, as printed",
    "registrant_name": "Name of the registrant, as printed",
    "file_date": "Filing date of the form, as printed",
    "foreign_principle_name": "Name of the foreign principal, as printed",
    "signer_name": "Name of the person signing the form, as printed",
    "signer_title": "Title of the person signing the form, as printed",
}
PROMPT_PREFIX = {
    "ad-buy-form": "Extract the invoice fields from the document text according to the JSON Schema. Copy values exactly as they appear in the document.\n\n",
    "registration-form": "Extract the registration form fields from the document text according to the JSON Schema. Copy values exactly as they appear in the document.\n\n",
}


def clean(s: str) -> str:
    return " ".join(s.split())


def build(row: dict, subset: str, meta: dict):
    header: dict[str, str] = {}
    alts: dict[str, list[str]] = {}
    items: list[dict] = []
    for ent, occ in row["annotations"]:
        if isinstance(ent, str):
            vals = [clean(o[0]) for o in occ if clean(o[0])]
            if not vals:
                continue
            header.setdefault(ent, vals[0])
            bucket = alts.setdefault(ent, [])
            for v in vals:
                if v not in bucket:
                    bucket.append(v)
        else:  # line item: (keys, [[(text, bbox, seg) per key]])
            for group in occ:
                item = {k: clean(v[0]) for k, v in zip(ent, group) if clean(v[0])}
                if item:
                    items.append({k: item[k] for k in LINE_KEYS if k in item})
    header_keys = [k for k in meta["entity_name_to_match_func"] if meta.get("entity_appearance_pattern", {}).get(k, "unrepeated") != "line_item"]
    props = {k: {"type": "string", "description": DESC[k]} for k in header_keys}
    gold = {k: header[k] for k in header_keys if k in header}
    if subset == "ad-buy-form":
        props["line_items"] = {"type": "array", "description": "One entry per advertising line item (spot / program row) in the order they appear",
                               "items": {"type": "object", "additionalProperties": False,
                                         "properties": {k: {"type": "string", "description": DESC[k]} for k in LINE_KEYS}}}
        gold["line_items"] = items
    schema = {"type": "object", "additionalProperties": False, "properties": props,
              "required": [k for k in props if k in gold]}
    return schema, gold, alts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/mnt/nvme/cache/interns/vrdu"))
    ap.add_argument("--subset", choices=["ad-buy-form", "registration-form"], required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tag = {"ad-buy-form": "vrdu_adbuy", "registration-form": "vrdu_registration"}[a.subset]
    out = a.output or ROOT / "benchmark" / "data" / "realworld" / f"{tag}.jsonl"
    meta = json.load((a.src / a.subset / "meta.json").open())
    rows = [json.loads(l) for l in (a.src / a.subset / "dataset.jsonl").open(encoding="utf-8")]
    if a.max_docs and a.max_docs < len(rows):
        import random
        rows = random.Random(a.seed).sample(rows, a.max_docs)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    lengths, n_items = [], 0
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            schema, gold, alts = build(r, a.subset, meta)
            n_items += len(gold.get("line_items", []))
            prompt = f"{PROMPT_PREFIX[a.subset]}=== Report ===\n{r['ocr']['text']}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            rec = {"stem": f"{tag}_{i:04d}", "dataset": tag, "source_id": r["filename"], "user_prompt": prompt,
                   "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False),
                   "gold_alts": json.dumps(alts, ensure_ascii=False), "pages": len(r["ocr"]["pages"]),
                   "match_func": meta["entity_name_to_match_func"]}
            if tok is not None:
                n = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = n; lengths.append(n)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({n_items} line items) to {out}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}; >8192: {sum(l>8192 for l in lengths)}; >32768: {sum(l>32768 for l in lengths)}")


if __name__ == "__main__":
    main()
