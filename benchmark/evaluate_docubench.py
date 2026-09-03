"""Score inference JSONL with DocuBench's official array-aware field scorer."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_scorer(root: Path):
    spec = importlib.util.spec_from_file_location("docubench_scorer", root / "scorer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load DocuBench scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--docubench-root", type=Path, required=True)
    args = parser.parse_args()
    scorer = load_scorer(args.docubench_root)
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    scores = []
    for row in rows:
        # benchmark/inference.py intentionally persists a stable ``stem`` but
        # not arbitrary source metadata.  The converter namespaces it here.
        doc_id = row.get("source_id") or row["stem"].removeprefix("docubench_")
        prediction = json.loads(row["pred_json"]) if row.get("pred_json") else {}
        schema = json.loads((args.docubench_root / "schemas" / f"{doc_id}.json").read_text())
        label = json.loads((args.docubench_root / "labels" / f"{doc_id}.json").read_text())
        scores.append({"doc_id": doc_id, "score": scorer.score_standardization(prediction, schema, label)["final"]})
    print(json.dumps({"documents": len(scores), "macro_field_accuracy": 100 * sum(x["score"] for x in scores) / len(scores), "per_document": scores}, indent=2))


if __name__ == "__main__":
    main()
