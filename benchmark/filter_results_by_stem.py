"""Keep inference records that share the evaluable stems of a reference run.

This is used when constrained decoding cannot compile a subset of schemas: the
unconstrained counterpart must be scored on the same population rather than on
the larger original benchmark population.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import resolve_path


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL to filter")
    parser.add_argument("--reference", required=True, help="JSONL defining the shared, evaluable stems")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = load_records(resolve_path(args.reference))
    stems = {str(row["stem"]) for row in reference if not row.get("skip_reason")}
    candidates = load_records(resolve_path(args.input))
    kept = [row for row in candidates if str(row.get("stem")) in stems]
    found = {str(row.get("stem")) for row in kept}
    missing = stems - found
    if missing:
        raise SystemExit(f"{len(missing)} reference stems are absent from input (first: {sorted(missing)[0]})")
    if len(kept) != len(stems):
        raise SystemExit(f"Expected {len(stems)} records but kept {len(kept)}; duplicate stems in input?")
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} shared records: {output}")


if __name__ == "__main__":
    main()
