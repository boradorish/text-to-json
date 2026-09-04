"""Score VRDU inference outputs (see prepare_vrdu.py) with type-aware normalized matching.

Metrics per run: parse success (PFR), schema-valid rate (SCR), header VA (gold header
fields recovered; any gold occurrence counts, matched with the dataset's match function:
general string / numeric / date / price / address), header hallucination rate (non-empty
prediction for an absent gold field), line-item field recall (greedy item alignment by
overlapping fields), line-item precision (predicted items aligned to a gold item), and a
prompt-length breakdown. Usage:
  python benchmark/score_vrdu.py --benchmark benchmark/data/realworld/vrdu_adbuy.jsonl \
      base=outputs/.../base.jsonl sft=outputs/.../sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_row  # noqa: E402

EDGES = [2048, 4096, 8192, 16384, 32768]
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def bucket(n: int | None) -> str:
    if n is None:
        return "?"
    lo = 0
    for e in EDGES:
        if n <= e:
            return f"{lo//1024}-{e//1024}k"
        lo = e
    return f">{lo//1024}k"


def norm_general(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def norm_num(s: str) -> str:
    return re.sub(r"\D", "", str(s))


def norm_price(s: str):
    m = re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(" ", ""))
    if not m:
        return norm_general(s)
    try:
        return round(float(m[0].replace(",", "")), 2)
    except ValueError:
        return norm_general(s)


def norm_date(s: str):
    t = str(s).strip().lower()
    m = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", t)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (mo, d, y % 100)
    m = re.search(r"([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{2,4})", t)
    if m and m.group(1) in MONTHS:
        return (MONTHS[m.group(1)], int(m.group(2)), int(m.group(3)) % 100)
    m = re.search(r"(\d{1,2})\s+([a-z]{3})[a-z]*\.?,?\s+(\d{2,4})", t)
    if m and m.group(2) in MONTHS:
        return (MONTHS[m.group(2)], int(m.group(1)), int(m.group(3)) % 100)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return (int(m.group(2)), int(m.group(3)), int(m.group(1)) % 100)
    return norm_general(s)


NORM = {"GeneralStringMatch": norm_general, "AddressMatch": norm_general, "NumericalStringMatch": norm_num,
        "DateMatch": norm_date, "PriceMatch": norm_price}


def match(pred, golds: list[str], func: str) -> bool:
    if pred is None or pred == "" or isinstance(pred, (dict, list)):
        return False
    f = NORM.get(func, norm_general)
    p = f(pred)
    return any(p == f(g) for g in golds) and p not in ("", 0)


def score(bench: dict[str, dict], runs: dict[str, Path]) -> dict:
    report = {}
    for name, path in runs.items():
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        agg = {"n": 0, "parse": 0, "valid": 0, "hdr_gold": 0, "hdr_hit": 0, "hdr_absent": 0, "hdr_halluc": 0,
               "item_fields": 0, "item_hits": 0, "pred_items": 0, "pred_items_aligned": 0, "gold_items": 0}
        per_bucket: dict[str, dict] = {}
        for r in rows:
            b = bench[r["stem"]]
            mf = b["match_func"]; alts = json.loads(b["gold_alts"]); gold = json.loads(b["gold_json"])
            schema = json.loads(b["json_schema"])
            ev = evaluate_row({**r, "gold_json": b["gold_json"], "json_schema": b["json_schema"]})
            try:
                pred = json.loads(r.get("pred_json") or "")
            except Exception:
                pred = None
            if not isinstance(pred, dict):
                pred = {}
            bk = bucket(b.get("prompt_tokens"))
            for d in (agg, per_bucket.setdefault(bk, {k: 0 for k in agg})):
                d["n"] += 1; d["parse"] += not ev["no_output"]; d["valid"] += ev["schema_valid"]
                for k in schema["properties"]:
                    if k == "line_items":
                        continue
                    if k in gold:
                        d["hdr_gold"] += 1; d["hdr_hit"] += match(pred.get(k), alts.get(k, [gold[k]]), mf[k])
                    else:
                        d["hdr_absent"] += 1; d["hdr_halluc"] += bool(pred.get(k))
                gitems = gold.get("line_items", [])
                pitems = [x for x in (pred.get("line_items") or []) if isinstance(x, dict)] if isinstance(pred.get("line_items"), list) else []
                d["gold_items"] += len(gitems); d["pred_items"] += len(pitems)
                d["item_fields"] += sum(len(g) for g in gitems)
                used = set()
                for g in gitems:
                    best, bi = 0, None
                    for j, p in enumerate(pitems):
                        if j in used:
                            continue
                        h = sum(match(p.get(k), [v], mf[k]) for k, v in g.items())
                        if h > best:
                            best, bi = h, j
                    if bi is not None:
                        used.add(bi); d["item_hits"] += best
                d["pred_items_aligned"] += len(used)

        def fin(d):
            return {"n": d["n"], "PFR": d["parse"] / d["n"], "SCR": d["valid"] / d["n"],
                    "header_VA": d["hdr_hit"] / max(1, d["hdr_gold"]), "header_halluc": d["hdr_halluc"] / max(1, d["hdr_absent"]),
                    "item_field_recall": d["item_hits"] / max(1, d["item_fields"]), "item_precision": d["pred_items_aligned"] / max(1, d["pred_items"]),
                    "gold_items": d["gold_items"], "pred_items": d["pred_items"]}
        report[name] = {"all": fin(agg), "buckets": {k: fin(v) for k, v in sorted(per_bucket.items(), key=lambda kv: (kv[0].startswith(">"), kv[0]))}}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("runs", nargs="+", help="label=path.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    bench = {}
    for l in a.benchmark.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l); bench[r["stem"]] = r
    runs = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in a.runs}
    rep = score(bench, runs)
    cols = ["PFR", "SCR", "header_VA", "header_halluc", "item_field_recall", "item_precision"]
    print(f"{'run':<14}{'n':>5}" + "".join(f"{c:>18}" for c in cols))
    for name, r in rep.items():
        print(f"{name:<14}{r['all']['n']:>5}" + "".join(f"{100*r['all'][c]:>18.1f}" for c in cols))
    print("\n-- by prompt-token bucket (header_VA / item_field_recall) --")
    buckets = sorted({b for r in rep.values() for b in r["buckets"]}, key=lambda k: (k.startswith(">"), k))
    print(f"{'bucket':<10}{'n':>5}" + "".join(f"{name:>22}" for name in rep))
    for b in buckets:
        n = next((r["buckets"][b]["n"] for r in rep.values() if b in r["buckets"]), 0)
        print(f"{b:<10}{n:>5}" + "".join(f"{100*r['buckets'][b]['header_VA']:>11.1f}/{100*r['buckets'][b]['item_field_recall']:<10.1f}" if b in r["buckets"] else " " * 22 for r in rep.values()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
