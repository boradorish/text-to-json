"""Extract flat spreadsheet-row records from STAGE reports (markdown tables).

Each record = one table row paired with its header row, i.e. a set of
(column -> cell) pairs grounded in the original spreadsheet. Records feed the
dialogue-state data generator (STAGE-Dialog)."""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def ascii_ratio(s: str) -> float:
    return sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))

def parse_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    tables, cur = [], []
    for line in text.splitlines():
        if re.match(r"^\s*\|.*\|\s*$", line):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        else:
            if len(cur) >= 3: tables.append(cur)
            cur = []
    if len(cur) >= 3: tables.append(cur)
    out = []
    for t in tables:
        header = t[0]
        rows = [r for r in t[1:] if not all(re.fullmatch(r":?-+:?", c or "-") for c in r)]
        out.append((header, [r for r in rows if len(r) == len(header)]))
    return out

def good_header(h: str) -> bool:
    return bool(h) and not h.lower().startswith("unnamed") and 2 <= len(h) <= 40 and ascii_ratio(h) > 0.95

def good_cell(v: str) -> bool:
    return bool(v) and v.lower() not in ("nan", "none", "null", "-", "") and 2 <= len(v) <= 60 and ascii_ratio(v) > 0.9

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=ROOT / "data" / "report")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "stage_dialog_records.jsonl")
    ap.add_argument("--min-cols", type=int, default=4)
    ap.add_argument("--max-cols", type=int, default=10)
    ap.add_argument("--per-table", type=int, default=2, help="rows sampled per table")
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); rng = random.Random(a.seed)
    records = []; stats = {"reports": 0, "tables": 0, "kept_tables": 0}
    for path in sorted(a.reports.glob("*.txt")):
        stats["reports"] += 1
        for header, rows in parse_tables(path.read_text(encoding="utf-8", errors="ignore")):
            stats["tables"] += 1
            cols = [i for i, h in enumerate(header) if good_header(h)]
            if not (a.min_cols <= len(cols) <= a.max_cols): continue
            cand = []
            for r in rows:
                cells = {header[i]: r[i] for i in cols if good_cell(r[i])}
                if len(cells) >= a.min_cols and len(set(cells.values())) == len(cells):
                    cand.append(cells)
            if not cand: continue
            stats["kept_tables"] += 1
            for cells in rng.sample(cand, min(a.per_table, len(cand))):
                records.append({"stem": path.stem, "columns": list(cells), "record": cells})
    rng.shuffle(records); records = records[: a.limit]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as fh:
        for r in records: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["records"] = len(records)
    print(json.dumps(stats)); print("saved", a.output)
    for r in records[:3]: print(json.dumps(r, ensure_ascii=False)[:300])

if __name__ == "__main__":
    main()
