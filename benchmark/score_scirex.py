"""Score SciREX salient-entity outputs (see prepare_scirex.py).

Entity recall: a gold salient entity counts as found when any predicted string of the same
type matches any of its mention forms (normalized equality, containment with length ratio
>= 0.5, or token Jaccard >= 0.5). Strict precision: predicted strings matching a salient
entity; lenient precision: predicted strings matching any annotated mention of that type
(the paper names it, but annotators did not deem it salient). Prompt-length buckets included.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_row  # noqa: E402

EDGES = [8192, 12288, 16384]


def bucket(n):
    if n is None:
        return "?"
    lo = 0
    for e in EDGES:
        if n <= e:
            return f"{lo//1024}-{e//1024}k"
        lo = e
    return f">{lo//1024}k"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def match(p, g):
    a, b = norm(p), norm(g)
    if not a or not b:
        return False
    if a == b or ((a in b or b in a) and min(len(a), len(b)) >= 0.5 * max(len(a), len(b))):
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) >= 0.5


def new():
    return {"n": 0, "parse": 0, "valid": 0, "gold": 0, "hit": 0, "pred": 0, "pred_strict": 0, "pred_lenient": 0}


def fin(d):
    rec = d["hit"] / max(1, d["gold"]); ps = d["pred_strict"] / max(1, d["pred"]); pl = d["pred_lenient"] / max(1, d["pred"])
    return {"n": d["n"], "PFR": d["parse"] / max(1, d["n"]), "SCR": d["valid"] / max(1, d["n"]), "entity_recall": rec,
            "precision_strict": ps, "precision_lenient": pl, "F1_strict": 2 * ps * rec / max(1e-9, ps + rec), "F1_lenient": 2 * pl * rec / max(1e-9, pl + rec),
            "gold": d["gold"], "pred": d["pred"]}


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
        agg, per, per_type = new(), {}, {}
        for r in rows:
            b = bench[r["stem"]]; gold = json.loads(b["gold_json"]); alts = json.loads(b["gold_alts"]); allm = json.loads(b["all_mentions"])
            ev = evaluate_row({**r, "gold_json": b["gold_json"], "json_schema": b["json_schema"]})
            try:
                p = json.loads(r.get("pred_json") or "")
            except Exception:
                p = None
            if not isinstance(p, dict):
                p = {}
            targets = [agg, per.setdefault(bucket(b.get("prompt_tokens")), new())]
            for d in targets:
                d["n"] += 1; d["parse"] += not ev["no_output"]; d["valid"] += ev["schema_valid"]
            for k in gold:
                pv = p.get(k); ps = [x for x in pv if isinstance(x, str) and x.strip()] if isinstance(pv, list) else []
                hit = sum(any(match(x, f) for x in ps for f in forms) for forms in alts[k])
                strict = sum(any(match(x, f) for forms in alts[k] for f in forms) for x in ps)
                lenient = sum(any(match(x, m) for m in allm[k]) or any(match(x, f) for forms in alts[k] for f in forms) for x in ps)
                for d in targets + [per_type.setdefault(k, new())]:
                    d["gold"] += len(gold[k]); d["hit"] += hit; d["pred"] += len(ps); d["pred_strict"] += strict; d["pred_lenient"] += lenient
        rep[name] = {"all": fin(agg), "buckets": {k: fin(v) for k, v in sorted(per.items(), key=lambda kv: (kv[0].startswith(">"), kv[0]))}, "types": {k: fin(v) for k, v in per_type.items()}}
    cols = ["PFR", "SCR", "entity_recall", "precision_strict", "precision_lenient", "F1_strict", "F1_lenient"]
    print(f"{'run':<10}{'n':>4}" + "".join(f"{c[:14]:>16}" for c in cols))
    for name, r in rep.items():
        print(f"{name:<10}{r['all']['n']:>4}" + "".join(f"{100*r['all'][c]:>16.1f}" for c in cols) + f"   pred={r['all']['pred']} gold={r['all']['gold']}")
    print("\n-- entity recall by type --")
    for name, r in rep.items():
        print(f"  {name:<10}", {k: round(100 * v["entity_recall"], 1) for k, v in r["types"].items()})
    print("\n-- by prompt-token bucket: entity_recall / precision_strict --")
    bks = sorted({b for r in rep.values() for b in r["buckets"]}, key=lambda k: (k.startswith(">"), k))
    print(f"{'bucket':<10}{'n':>5}" + "".join(f"{n:>22}" for n in rep))
    for b in bks:
        n = next((r["buckets"][b]["n"] for r in rep.values() if b in r["buckets"]), 0)
        print(f"{b:<10}{n:>5}" + "".join(f"{100*r['buckets'][b]['entity_recall']:>11.1f}/{100*r['buckets'][b]['precision_strict']:<10.1f}" if b in r["buckets"] else " " * 22 for r in rep.values()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rep, indent=2) + "\n"); print(f"saved {a.out}")


if __name__ == "__main__":
    main()
