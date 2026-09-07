"""RealKIE-FCC 74: 3,100-token vs 16,384-token generation budget, base vs STAGE, 3 seeds.
Per bucket: header VA, line-item field VA, item recall, SCR, truncation share. Writes benchmark/paper_figures/data/realkie_budget_summary.json"""
import json, re, statistics, sys
from pathlib import Path
sys.path.insert(0, "benchmark")
from score_seeds_realworld import rk_score, _norm, _J, HDR, load  # reuse the paper's RealKIE scorer
from evaluate import evaluate_row
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/work/sunghee/models/Qwen3-4B")
def bucket(n):
    for e, l in [(4096, "<=4k"), (8192, "4-8k"), (16384, "8-16k")]:
        if n <= e: return l
    return ">16k"
def ms(v): return {"mean": statistics.mean(v), "std": statistics.pstdev(v), "per_seed": v}
RUNS = {"3100": {"base": "outputs/realkie_sampling3/base_nothink_s{s}.jsonl", "stage": "outputs/realkie_sampling3/sft_s{s}.jsonl"},
        "16384": {"base": "outputs/realkie_longout/base_nothink_s{s}.jsonl", "stage": "outputs/realkie_longout/sft_s{s}.jsonl",
                  "q25_base": "outputs/realkie_longout/q25_base_s{s}.jsonl", "q25_stage": "outputs/realkie_longout/q25_sft_s{s}.jsonl"}}
out = {"protocol": "temperature 0.6, top-p 1.0, seeds 42/43/44, thinking disabled for the untrained model; budgets 3,100 vs 16,384 new tokens"}
ntok = None
for budget, arms in RUNS.items():
    out[budget] = {}
    for arm, pat in arms.items():
        per = {}
        for s in (42, 43, 44):
            p = pat.format(s=s)
            if not Path(p).exists(): continue
            rows = load(p)
            if ntok is None: ntok = {r["stem"]: len(tok(r["user_prompt"])["input_ids"]) for r in rows}
            cap = 3000 if budget == "3100" else 16000
            d = {"all": {}, "buckets": {}}
            def stats(rs):
                h, li, rec, cnt = rk_score(rs)
                m = [evaluate_row(r) for r in rs]
                trunc = 100 * sum(len(tok(r["raw_output"] or "")["input_ids"]) >= cap for r in rs) / len(rs)
                otok = statistics.mean(len(tok(r["raw_output"] or "")["input_ids"]) for r in rs)
                return {"n": len(rs), "header_va": h, "item_field_va": li, "item_recall": rec, "SCR": 100 * sum(x["schema_valid"] for x in m) / len(m), "PFR": 100 * sum(not x["no_output"] for x in m) / len(m), "truncated": trunc, "out_tokens": otok}
            d["all"] = stats(rows)
            for b in ["<=4k", "4-8k", "8-16k", ">16k"]:
                d["buckets"][b] = stats([r for r in rows if bucket(ntok[r["stem"]]) == b])
            per[s] = d
        if not per: continue
        seeds = sorted(per); keys = ["header_va", "item_field_va", "item_recall", "SCR", "PFR", "truncated", "out_tokens"]
        agg = {"seeds": seeds, "all": {k: ms([per[s]["all"][k] for s in seeds]) for k in keys}, "buckets": {}}
        for b in per[seeds[0]]["buckets"]:
            agg["buckets"][b] = {"n": per[seeds[0]]["buckets"][b]["n"], **{k: ms([per[s]["buckets"][b][k] for s in seeds]) for k in keys}}
        out[budget][arm] = agg
Path("benchmark/paper_figures/data").mkdir(parents=True, exist_ok=True)
Path("benchmark/paper_figures/data/realkie_budget_summary.json").write_text(json.dumps(out, indent=2))
for budget in RUNS:
    for arm, a in out[budget].items():
        print(f"budget {budget:>5s} {arm:5s} seeds={a['seeds']} all: hdr {a['all']['header_va']['mean']:.1f}±{a['all']['header_va']['std']:.1f} item {a['all']['item_field_va']['mean']:.1f} recall {a['all']['item_recall']['mean']:.1f} SCR {a['all']['SCR']['mean']:.1f} trunc {a['all']['truncated']['mean']:.0f}% out {a['all']['out_tokens']['mean']:.0f}")
        for b, v in a["buckets"].items():
            print(f"      {b:6s} n={v['n']:2d} hdr {v['header_va']['mean']:5.1f}±{v['header_va']['std']:4.1f} item {v['item_field_va']['mean']:5.1f} recall {v['item_recall']['mean']:5.1f} SCR {v['SCR']['mean']:5.1f} trunc {v['truncated']['mean']:3.0f}% out {v['out_tokens']['mean']:5.0f}")
