"""Score PMC article-metadata outputs (see prepare_pmc.py).

Scalar fields (title, journal, year) and keywords use normalized string equality; authors and
references are aligned greedily to gold records by the number of matching fields, giving
record recall (gold records with an aligned prediction that matches at least the name / title),
field recall over gold record fields, and record precision. Prompt-length buckets included.
Usage: python benchmark/score_pmc.py --benchmark benchmark/data/realworld/pmc_oa_2024.jsonl base=... sft=...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_row  # noqa: E402

EDGES = [8192, 16384, 32768]


def bucket(n):
    if n is None:
        return "?"
    lo = 0
    for e in EDGES:
        if n <= e:
            return f"{lo//1024}-{e//1024}k"
        lo = e
    return f">{lo//1024}k"


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def eq(a, b) -> bool:
    return isinstance(a, str) and norm(a) != "" and norm(a) == norm(b)


def align(gold: list[dict], pred: list, keys: list[str], anchor: str):
    pred = [p for p in pred if isinstance(p, dict)] if isinstance(pred, list) else []
    used, rec_hit, field_hit, field_tot = set(), 0, 0, 0
    for g in gold:
        field_tot += sum(1 for k in keys if g.get(k) not in (None, "", []))
        best, bi = 0, None
        for j, p in enumerate(pred):
            if j in used:
                continue
            h = sum(eq(p.get(k), g.get(k)) for k in keys if isinstance(g.get(k), str))
            if h > best:
                best, bi = h, j
        if bi is not None:
            used.add(bi); p = pred[bi]
            rec_hit += eq(p.get(anchor), g.get(anchor))
            for k in keys:
                gv = g.get(k)
                if isinstance(gv, str) and gv:
                    field_hit += eq(p.get(k), gv)
                elif isinstance(gv, list) and gv:
                    pv = p.get(k) if isinstance(p.get(k), list) else []
                    field_hit += all(any(eq(x, y) for x in pv) for y in gv)
    return rec_hit, field_hit, field_tot, len(used), len(pred)


def new():
    return {"n": 0, "parse": 0, "valid": 0, "scalar_hit": 0, "scalar_tot": 0, "kw_hit": 0, "kw_tot": 0,
            "auth_rec": 0, "auth_field": 0, "auth_field_tot": 0, "auth_gold": 0, "auth_used": 0, "auth_pred": 0,
            "ref_rec": 0, "ref_field": 0, "ref_field_tot": 0, "ref_gold": 0, "ref_used": 0, "ref_pred": 0}


def fin(d):
    n = max(1, d["n"])
    return {"n": d["n"], "PFR": d["parse"] / n, "SCR": d["valid"] / n, "scalar_VA": d["scalar_hit"] / max(1, d["scalar_tot"]),
            "keyword_recall": d["kw_hit"] / max(1, d["kw_tot"]), "author_record_recall": d["auth_rec"] / max(1, d["auth_gold"]),
            "author_field_recall": d["auth_field"] / max(1, d["auth_field_tot"]), "author_precision": d["auth_used"] / max(1, d["auth_pred"]),
            "ref_record_recall": d["ref_rec"] / max(1, d["ref_gold"]), "ref_field_recall": d["ref_field"] / max(1, d["ref_field_tot"]),
            "ref_precision": d["ref_used"] / max(1, d["ref_pred"]), "ref_count_ratio": d["ref_pred"] / max(1, d["ref_gold"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    bench = {json.loads(l)["stem"]: json.loads(l) for l in a.benchmark.open(encoding="utf-8") if l.strip()}
    rep = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
        agg, per = new(), {}
        for r in rows:
            b = bench[r["stem"]]; g = json.loads(b["gold_json"])
            ev = evaluate_row({**r, "gold_json": b["gold_json"], "json_schema": b["json_schema"]})
            try:
                p = json.loads(r.get("pred_json") or "")
            except Exception:
                p = None
            if not isinstance(p, dict):
                p = {}
            for d in (agg, per.setdefault(bucket(b.get("prompt_tokens")), new())):
                d["n"] += 1; d["parse"] += not ev["no_output"]; d["valid"] += ev["schema_valid"]
                for k in ("title", "journal", "year"):
                    if g.get(k):
                        d["scalar_tot"] += 1; d["scalar_hit"] += eq(p.get(k), g[k])
                pk = p.get("keywords") if isinstance(p.get("keywords"), list) else []
                d["kw_tot"] += len(g["keywords"]); d["kw_hit"] += sum(any(eq(x, y) for x in pk) for y in g["keywords"])
                ar, af, aft, au, ap_ = align(g["authors"], p.get("authors"), ["surname", "given_names", "affiliations"], "surname")
                d["auth_rec"] += ar; d["auth_field"] += af; d["auth_field_tot"] += aft; d["auth_gold"] += len(g["authors"]); d["auth_used"] += au; d["auth_pred"] += ap_
                rr, rf, rft, ru, rp = align(g["references"], p.get("references"), ["first_author_surname", "year", "title"], "title")
                d["ref_rec"] += rr; d["ref_field"] += rf; d["ref_field_tot"] += rft; d["ref_gold"] += len(g["references"]); d["ref_used"] += ru; d["ref_pred"] += rp
        rep[name] = {"all": fin(agg), "buckets": {k: fin(v) for k, v in sorted(per.items(), key=lambda kv: (kv[0].startswith(">"), kv[0]))}}
    cols = ["PFR", "SCR", "scalar_VA", "keyword_recall", "author_record_recall", "author_field_recall", "author_precision", "ref_record_recall", "ref_field_recall", "ref_precision", "ref_count_ratio"]
    print(f"{'run':<10}{'n':>4}" + "".join(f"{c[:12]:>13}" for c in cols))
    for name, r in rep.items():
        print(f"{name:<10}{r['all']['n']:>4}" + "".join(f"{100*r['all'][c]:>13.1f}" for c in cols))
    print("\n-- by prompt-token bucket: author_field_recall / ref_record_recall --")
    bks = sorted({b for r in rep.values() for b in r["buckets"]}, key=lambda k: (k.startswith(">"), k))
    print(f"{'bucket':<10}{'n':>5}" + "".join(f"{n:>22}" for n in rep))
    for b in bks:
        n = next((r["buckets"][b]["n"] for r in rep.values() if b in r["buckets"]), 0)
        print(f"{b:<10}{n:>5}" + "".join(f"{100*r['buckets'][b]['author_field_recall']:>11.1f}/{100*r['buckets'][b]['ref_record_recall']:<10.1f}" if b in r["buckets"] else " " * 22 for r in rep.values()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rep, indent=2) + "\n"); print(f"saved {a.out}")


if __name__ == "__main__":
    main()
