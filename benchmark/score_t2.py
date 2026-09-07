"""Score the Table 2 STAGE checkpoint (boradorish/qwen3-4b-new) on STAGE-Eval: free and xgrammar, seeds 42/43/44,
means/std on all 851 and on the 798 xgrammar-compilable schemas. Writes benchmark/paper_figures/data/t2_ckpt_summary.json"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, "benchmark")
from evaluate import evaluate_row
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def ms(v): return {"mean": statistics.mean(v), "std": statistics.pstdev(v), "per_seed": v}
def agg(rows):
    m=[evaluate_row(r) for r in rows]; k=max(1,len(m))
    return {"n":len(m),"PFR":100*sum(not x["no_output"] for x in m)/k,"EMR":100*sum(x["exact_match"] for x in m)/k,"SCR":100*sum(x["schema_valid"] for x in m)/k,"NR":100*sum(x["noise_ratio"] for x in m)/k,"VA":100*sum(x["value_match"] for x in m)/k}
# the 53 schemas xgrammar cannot compile, from the earlier sampling3 run
skipped={json.loads(l)["stem"] for l in open("outputs/sampling3/base_nothink_xgrammar_s42.jsonl") if json.loads(l).get("skip_reason")}
out={"protocol":"boradorish/qwen3-4b-new (Table 2 checkpoint), temperature 0.6, top-p 1.0, seeds 42/43/44, 3,100 tokens","compat_n":851-len(skipped)}
for cond in ("t2_free","t2_xgrammar"):
    per={}
    for s in (42,43,44):
        p=Path(f"outputs/t2_ckpt/{cond}_s{s}.jsonl")
        if p.exists() and sum(1 for _ in open(p))==851:
            rows=load(p); per[s]={"all":agg([r for r in rows if not r.get("skip_reason")]),"compat":agg([r for r in rows if r["stem"] not in skipped])}
    if per:
        seeds=sorted(per); out[cond]={"seeds":seeds,**{sub:{k:ms([per[s][sub][k] for s in seeds]) for k in ("PFR","EMR","SCR","NR","VA")} for sub in ("all","compat")}}
        for sub in ("all","compat"): print(cond, sub, seeds, {k:f"{out[cond][sub][k]['mean']:.2f}±{out[cond][sub][k]['std']:.2f}" for k in ("PFR","EMR","SCR","NR","VA")})
Path("benchmark/paper_figures/data").mkdir(parents=True, exist_ok=True)
Path("benchmark/paper_figures/data/t2_ckpt_summary.json").write_text(json.dumps(out,indent=2))
