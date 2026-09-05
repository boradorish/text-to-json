"""Mean/std over seeds for the three-seed real-world runs (RealKIE 74, ExtractBench 32k/131k).

Reads outputs/realkie_sampling3/<cond>_s<seed>.jsonl and outputs/extractbench_sampling3/<cond>_s<seed>.jsonl,
scores each seed (RealKIE: header / line-item accuracy via score_realkie.score plus PFR/SCR; ExtractBench: rule metrics),
buckets by prompt tokens, and writes one JSON with per-seed values and mean/std.
"""
from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_row  # noqa: E402
import re


# --- RealKIE scoring (copied from score_realkie.py, which is a script and cannot be imported)
def _norm(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}"
    s = str(v).strip()
    m = re.fullmatch(r"[$]?\s*([-\d][\d,]*\.?\d*)", s)
    if m:
        try:
            return f"{float(m.group(1).replace(',', '')):.2f}"
        except ValueError:
            pass
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


HDR = ["Agency", "Advertiser", "GrossTotal", "PaymentTerms", "AgencyCommission", "NetAmountDue"]


def _J(v):
    try:
        return json.loads(v) if isinstance(v, str) else v
    except Exception:
        return None


def _item_sim(a, b):
    keys = set(a) | set(b)
    return sum(1 for k in keys if _norm(a.get(k)) == _norm(b.get(k))) / max(1, len(keys))


def rk_score(rows):
    hdr = [0, 0]; li_field = [0, 0]; li_recall = [0, 0]; cnt_ok = 0
    for r in rows:
        g = _J(r["gold_json"]); p = _J(r.get("pred_json")) or {}
        for k in HDR:
            if k in g:
                hdr[1] += 1; hdr[0] += _norm(p.get(k, "<M>")) == _norm(g[k])
        gi = g.get("LineItems") or []; pi = (p.get("LineItems") if isinstance(p, dict) else None) or []
        pi = [x for x in pi if isinstance(x, dict)]
        cnt_ok += len(pi) == len(gi)
        used = set()
        for gitem in gi:
            best, bi = -1, None
            for j, pitem in enumerate(pi):
                if j in used:
                    continue
                sc = _item_sim(gitem, pitem)
                if sc > best:
                    best, bi = sc, j
            li_recall[1] += 1
            if bi is not None and best >= 0.5:
                used.add(bi); li_recall[0] += 1
            for k, v in gitem.items():
                li_field[1] += 1
                if bi is not None and _norm(pi[bi].get(k, "<M>")) == _norm(v):
                    li_field[0] += 1
    return 100 * hdr[0] / max(1, hdr[1]), 100 * li_field[0] / max(1, li_field[1]), 100 * li_recall[0] / max(1, li_recall[1]), 100 * cnt_ok / max(1, len(rows))

SEEDS = (42, 43, 44)


def bucket(n, edges):
    lo = 0
    for e in edges:
        if n <= e:
            return f"{lo//1024}-{e//1024}k" if lo else f"<={e//1024}k"
        lo = e
    return f">{lo//1024}k"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ms(vals):
    return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "per_seed": vals}


def rule_agg(rows):
    m = [evaluate_row(r) for r in rows]; k = max(1, len(m))
    return {"n": len(m), "PFR": 100 * sum(not x["no_output"] for x in m) / k, "SCR": 100 * sum(x["schema_valid"] for x in m) / k, "VA": 100 * sum(x["value_match"] for x in m) / k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/root/work/sunghee/models/Qwen3-4B")
    ap.add_argument("--out", default="outputs/realworld_sampling3_summary.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = {"protocol": "temperature 0.6, top-p 1.0, seeds 42/43/44, max_new_tokens 4096, thinking disabled for the untrained model"}
    # ---- RealKIE
    B = {r["stem"]: r for r in load("benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl")}
    ntok = {s: len(tok(r["user_prompt"])["input_ids"]) for s, r in B.items()}
    edges = [4096, 8192, 16384]
    rk = {}
    for cond in ["base_nothink", "base_nothink_xgrammar", "sft", "sft_xgrammar", "sft_dialog_v2"]:
        per = {}
        for s in SEEDS:
            p = f"outputs/realkie_sampling3/{cond}_s{s}.jsonl"
            if not Path(p).exists():
                continue
            rows = load(p); h, lf, lr, c = rk_score(rows); ra = rule_agg(rows)
            d = {"header_va": h, "item_field_va": lf, "item_recall": lr, "count_ok": c, "PFR": ra["PFR"], "SCR": ra["SCR"], "buckets": {}}
            for b in sorted({bucket(ntok[r["stem"]], edges) for r in rows}):
                sub = [r for r in rows if bucket(ntok[r["stem"]], edges) == b]
                hb, lfb, _, _ = rk_score(sub); d["buckets"][b] = {"n": len(sub), "header_va": hb, "item_field_va": lfb}
            per[s] = d
        if per:
            seeds = sorted(per)
            agg = {k: ms([per[s][k] for s in seeds]) for k in ("header_va", "item_field_va", "item_recall", "count_ok", "PFR", "SCR")}
            agg["buckets"] = {b: {"n": per[seeds[0]]["buckets"][b]["n"], "header_va": ms([per[s]["buckets"][b]["header_va"] for s in seeds]), "item_field_va": ms([per[s]["buckets"][b]["item_field_va"] for s in seeds])} for b in per[seeds[0]]["buckets"]}
            agg["seeds"] = seeds; rk[cond] = agg
    out["realkie_74"] = rk
    # ---- ExtractBench 131k buckets
    B131 = {r["stem"]: r for r in load("benchmark/data/extractbench_context131072.jsonl")}
    ntok131 = {s: len(tok(r["user_prompt"])["input_ids"]) for s, r in B131.items()}
    edges131 = [4096, 8192, 16384, 32768, 65536]
    eb = {}
    for cond in ["base_nothink_yarn", "sft_yarn"]:
        per = {}
        for s in SEEDS:
            p = f"outputs/extractbench_sampling3/{cond}_s{s}.jsonl"
            if not Path(p).exists():
                continue
            rows = load(p); d = {"all": rule_agg(rows), "buckets": {}}
            for b in sorted({bucket(ntok131[r["stem"]], edges131) for r in rows}):
                d["buckets"][b] = rule_agg([r for r in rows if bucket(ntok131[r["stem"]], edges131) == b])
            per[s] = d
        if per:
            seeds = sorted(per); agg = {"seeds": seeds, "all": {k: ms([per[s]["all"][k] for s in seeds]) for k in ("PFR", "SCR", "VA")}, "buckets": {}}
            for b in per[seeds[0]]["buckets"]:
                agg["buckets"][b] = {"n": per[seeds[0]]["buckets"][b]["n"], **{k: ms([per[s]["buckets"][b][k] for s in seeds]) for k in ("PFR", "SCR", "VA")}}
            eb[cond] = agg
    out["extractbench_131k_237"] = eb
    # ---- ExtractBench 32k, 194 compat across seeds and conditions
    B32 = {r["stem"]: r for r in load("benchmark/data/extractbench_context32768.jsonl")}
    conds = ["base_nothink_free", "base_nothink_xgrammar", "sft_free", "sft_xgrammar"]
    runs = {c: {s: {r["stem"]: r for r in load(f"outputs/extractbench_sampling3/{c}_s{s}.jsonl")} for s in SEEDS if Path(f"outputs/extractbench_sampling3/{c}_s{s}.jsonl").exists()} for c in conds}
    if all(runs[c] for c in conds):
        skipped = {st for c in conds if "xgrammar" in c for s in runs[c] for st, r in runs[c][s].items() if r.get("skip_reason")}
        stems = [st for st in next(iter(runs["sft_free"].values())) if st not in skipped]
        eb194 = {"compat_n": len(stems)}
        for c in conds:
            seeds = sorted(runs[c]); per = {}
            for s in seeds:
                rows = [runs[c][s][st] for st in stems]
                per[s] = {"all194": rule_agg(rows), "short": rule_agg([r for r in rows if B32[r["stem"]]["source_split"].endswith("short")]), "medium": rule_agg([r for r in rows if not B32[r["stem"]]["source_split"].endswith("short")])}
            eb194[c] = {"seeds": seeds, **{sub: {k: ms([per[s][sub][k] for s in seeds]) for k in ("PFR", "SCR", "VA")} for sub in ("all194", "short", "medium")}}
        out["extractbench_194"] = eb194
    Path(a.out).write_text(json.dumps(out, indent=2))
    for cond, v in rk.items():
        print("RealKIE", cond, {k: round(v[k]["mean"], 1) for k in ("header_va", "item_field_va", "SCR")})
    for cond, v in eb.items():
        print("EB131k", cond, {b: round(x["PFR"]["mean"], 1) for b, x in v["buckets"].items()})
    print("saved", a.out)


if __name__ == "__main__":
    main()
