"""Transparent value-set F1 for Kleister-NDA JSON predictions.

Kleister's public gold uses canonical underscores and ISO dates.  This scorer
normalizes only superficial formatting (case, punctuation, common date forms)
for both prediction and gold, then computes micro precision/recall/F1 over
the four official fields.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


FIELDS = ("effective_date", "jurisdiction", "party", "term")


def normalize(value: object, field: str) -> str:
    text = str(value).strip()
    if field == "effective_date":
        for form in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(text, form).strftime("%Y-%m-%d")
            except ValueError:
                pass
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    tp = fp = fn = 0
    per_field = {field: [0, 0, 0] for field in FIELDS}
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    for row in rows:
        gold = json.loads(row["gold_json"])
        pred = json.loads(row["pred_json"]) if row.get("pred_json") else {}
        for field in FIELDS:
            target = {normalize(value, field) for value in gold.get(field, [])}
            actual = {normalize(value, field) for value in pred.get(field, [])}
            a, b, c = len(target & actual), len(actual - target), len(target - actual)
            tp += a; fp += b; fn += c
            per_field[field][0] += a; per_field[field][1] += b; per_field[field][2] += c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    payload = {"documents": len(rows), "precision": 100 * precision, "recall": 100 * recall, "F1": 100 * (2 * precision * recall / (precision + recall) if precision + recall else 0.0)}
    payload["per_field"] = {
        field: {"tp": a, "fp": b, "fn": c, "F1": 100 * (2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0)}
        for field, (a, b, c) in per_field.items()
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
