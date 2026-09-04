"""Per-length-bucket comparison of inference runs on one benchmark file.

For every run (label=path.jsonl) computes parse success, schema validity and rule VA with
``evaluate.evaluate_row`` and groups rows by prompt-token bucket. Prompt tokens come from
the benchmark file (``prompt_tokens``) or are counted with ``--tokenizer`` on
``user_prompt``. Prints a table and optionally writes JSON for the paper data file.
Usage: python benchmark/length_bucket_analysis.py --benchmark <bench.jsonl> [--tokenizer PATH] base=... sft=... --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_row, extract_leaves  # noqa: E402
import re


def norm_leaf(v):
    """Lenient normalization: underscores as spaces, currency/number strings as 2-decimal floats, alnum lowercase."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}"
    s = str(v).strip().replace("_", " ")
    m = re.fullmatch(r"[£$€]?\s*([-\d][\d,]*\.?\d*)", s)
    if m:
        try:
            return f"{float(m.group(1).replace(',', '')):.2f}"
        except ValueError:
            pass
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def normalized_va(pred_json: str | None, gold_json: str) -> float:
    try:
        pred = json.loads(pred_json or "")
    except Exception:
        pred = None
    gold = json.loads(gold_json)
    gl = extract_leaves(gold)
    if not gl:
        return 1.0
    pl = extract_leaves(pred) if pred is not None else {}
    return sum(1 for k, v in gl.items() if k in pl and norm_leaf(pl[k]) == norm_leaf(v)) / len(gl)


def make_bucket(edges):
    def bucket(n):
        lo = 0
        for e in edges:
            if n <= e:
                return f"{lo//1024}-{e//1024}k" if lo else f"<={e//1024}k"
            lo = e
        return f">{lo//1024}k"
    return bucket


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--edges", default="2048,4096,8192,16384,32768")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--normalized", action="store_true", help="Print lenient normalized VA instead of strict rule VA")
    a = ap.parse_args()
    edges = [int(x) for x in a.edges.split(",")]
    bucket = make_bucket(edges)
    bench = {}
    for l in a.benchmark.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l); bench[str(r.get("stem") or r.get("id"))] = r
    tok = None
    if any("prompt_tokens" not in r for r in bench.values()):
        if not a.tokenizer:
            raise SystemExit("benchmark rows lack prompt_tokens; pass --tokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        for r in bench.values():
            if "prompt_tokens" not in r:
                r["prompt_tokens"] = len(tok(r["user_prompt"])["input_ids"])
    rep = {"benchmark": str(a.benchmark), "edges": edges, "runs": {}}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
        rows = [r for r in rows if not r.get("skip_reason")]
        per = {}
        for r in rows:
            b = bench.get(str(r.get("stem")))
            if b is None:
                continue
            ev = evaluate_row({**r, "gold_json": r.get("gold_json") or b["gold_json"], "json_schema": r.get("json_schema") or b["json_schema"]})
            d = per.setdefault(bucket(b["prompt_tokens"]), {"n": 0, "parse": 0, "valid": 0, "em": 0, "va": 0.0, "van": 0.0})
            d["n"] += 1; d["parse"] += not ev["no_output"]; d["valid"] += ev["schema_valid"]; d["em"] += ev["exact_match"]; d["va"] += ev["value_match"]
            d["van"] += normalized_va(r.get("pred_json"), r.get("gold_json") or b["gold_json"])
        rep["runs"][name] = {k: {"n": v["n"], "PFR": v["parse"] / v["n"], "SCR": v["valid"] / v["n"], "EMR": v["em"] / v["n"], "VA": v["va"] / v["n"], "VA_norm": v["van"] / v["n"]} for k, v in per.items()}
    order = lambda k: (k.startswith(">"), 0 if k.startswith("<=") else 1, int(k.split("-")[0].rstrip("k").lstrip("<=>")) if k[0].isdigit() or k.startswith("<=") or k.startswith(">") else 0)
    buckets = sorted({b for r in rep["runs"].values() for b in r}, key=order)
    names = list(rep["runs"])
    key = "VA_norm" if a.normalized else "VA"
    print(f"{'bucket':<10}{'n':>5}" + "".join(f"{n:>26}" for n in names) + f"   (PFR/SCR/{key})")
    for b in buckets:
        n = next((rep["runs"][x][b]["n"] for x in names if b in rep["runs"][x]), 0)
        cells = []
        for x in names:
            v = rep["runs"][x].get(b)
            cells.append(f"{100*v['PFR']:>8.1f}/{100*v['SCR']:>6.1f}/{100*v[key]:>6.1f}" if v else " " * 26)
        print(f"{b:<10}{n:>5}" + "".join(f"{c:>26}" for c in cells))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rep, indent=2) + "\n"); print(f"saved {a.out}")


if __name__ == "__main__":
    main()
