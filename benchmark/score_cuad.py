"""Score CUAD inference outputs (see prepare_cuad.py).

A predicted clause matches a gold span of the same field when their normalized token
sets have Jaccard >= 0.5, or one normalized string contains the other and the shorter
one is at least half the longer one's length (CUAD's own metric uses Jaccard >= 0.5 on
character spans). Reports span precision / recall / F1 (micro), field-level presence
accuracy (empty vs non-empty agreement), hallucinated-fill rate (non-empty prediction for
an empty gold field), parse / schema-valid rates, and a prompt-length breakdown.
Usage: python benchmark/score_cuad.py --benchmark benchmark/data/realworld/cuad_test.jsonl base=... sft=...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_row  # noqa: E402

EDGES = [4096, 8192, 16384, 32768, 65536]


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


def span_match(p: str, g: str) -> bool:
    a, b = norm(p), norm(g)
    if not a or not b:
        return False
    if a == b:
        return True
    if (a in b or b in a) and min(len(a), len(b)) >= 0.5 * max(len(a), len(b)):
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) >= 0.5


def new_agg():
    return {"n": 0, "parse": 0, "valid": 0, "gold_spans": 0, "pred_spans": 0, "tp_gold": 0, "tp_pred": 0,
            "fields": 0, "presence_ok": 0, "gold_empty": 0, "halluc_fill": 0}


def fin(d):
    prec = d["tp_pred"] / max(1, d["pred_spans"]); rec = d["tp_gold"] / max(1, d["gold_spans"])
    return {"n": d["n"], "PFR": d["parse"] / max(1, d["n"]), "SCR": d["valid"] / max(1, d["n"]), "span_precision": prec, "span_recall": rec,
            "span_F1": 2 * prec * rec / max(1e-9, prec + rec), "presence_acc": d["presence_ok"] / max(1, d["fields"]),
            "halluc_fill": d["halluc_fill"] / max(1, d["gold_empty"]), "gold_spans": d["gold_spans"], "pred_spans": d["pred_spans"]}


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
        agg, per_b, per_field = new_agg(), {}, {}
        for r in rows:
            b = bench[r["stem"]]; gold = json.loads(b["gold_json"])
            ev = evaluate_row({**r, "gold_json": b["gold_json"], "json_schema": b["json_schema"]})
            try:
                pred = json.loads(r.get("pred_json") or "")
            except Exception:
                pred = None
            if not isinstance(pred, dict):
                pred = {}
            targets = [agg, per_b.setdefault(bucket(b.get("prompt_tokens")), new_agg())]
            for d in targets:
                d["n"] += 1; d["parse"] += not ev["no_output"]; d["valid"] += ev["schema_valid"]
            for k, gs in gold.items():
                pv = pred.get(k)
                ps = [x for x in pv if isinstance(x, str)] if isinstance(pv, list) else ([pv] if isinstance(pv, str) and pv.strip() else [])
                hit_g = sum(any(span_match(p, g) for p in ps) for g in gs)
                hit_p = sum(any(span_match(p, g) for g in gs) for p in ps)
                pf = per_field.setdefault(k, new_agg())
                for d in targets + [pf]:
                    d["gold_spans"] += len(gs); d["pred_spans"] += len(ps); d["tp_gold"] += hit_g; d["tp_pred"] += hit_p
                    d["fields"] += 1; d["presence_ok"] += (bool(gs) == bool(ps))
                    if not gs:
                        d["gold_empty"] += 1; d["halluc_fill"] += bool(ps)
        rep[name] = {"all": fin(agg), "buckets": {k: fin(v) for k, v in sorted(per_b.items(), key=lambda kv: (kv[0].startswith(">"), kv[0]))},
                     "fields": {k: fin(v) for k, v in per_field.items()}}
    cols = ["PFR", "SCR", "span_precision", "span_recall", "span_F1", "presence_acc", "halluc_fill"]
    print(f"{'run':<14}{'n':>5}" + "".join(f"{c:>16}" for c in cols))
    for name, r in rep.items():
        print(f"{name:<14}{r['all']['n']:>5}" + "".join(f"{100*r['all'][c]:>16.1f}" for c in cols) + f"   pred_spans={r['all']['pred_spans']} gold={r['all']['gold_spans']}")
    print("\n-- by prompt-token bucket: span_recall / span_precision --")
    buckets = sorted({b for r in rep.values() for b in r["buckets"]}, key=lambda k: (k.startswith(">"), k))
    print(f"{'bucket':<10}{'n':>5}" + "".join(f"{name:>22}" for name in rep))
    for b in buckets:
        n = next((r["buckets"][b]["n"] for r in rep.values() if b in r["buckets"]), 0)
        print(f"{b:<10}{n:>5}" + "".join(f"{100*r['buckets'][b]['span_recall']:>11.1f}/{100*r['buckets'][b]['span_precision']:<10.1f}" if b in r["buckets"] else " " * 22 for r in rep.values()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rep, indent=2) + "\n"); print(f"saved {a.out}")


if __name__ == "__main__":
    main()
