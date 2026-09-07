"""Score the thinking-enabled untrained Qwen3-4B reruns: STAGE-Eval xgrammar (851 all / 798 compat), RealKIE-FCC 16k,
ExtractBench 131k buckets. Writes benchmark/paper_figures/data/think_on_summary.json"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, "benchmark")
from evaluate import evaluate_row
from score_seeds_realworld import rk_score, load, bucket
from transformers import AutoTokenizer
def ms(v): return {"mean": statistics.mean(v), "std": statistics.pstdev(v), "per_seed": v, "n_seeds": len(v)}
def agg(rows):
    m=[evaluate_row(r) for r in rows]; k=max(1,len(m))
    return {"n":len(m),"PFR":100*sum(not x["no_output"] for x in m)/k,"EMR":100*sum(x["exact_match"] for x in m)/k,"SCR":100*sum(x["schema_valid"] for x in m)/k,"NR":100*sum(x["noise_ratio"] for x in m)/k,"VA":100*sum(x["value_match"] for x in m)/k}
out={"protocol":"untrained Qwen3-4B, default thinking mode, temperature 0.6, top-p 1.0, seeds 42/43/44, 3,100 tokens (RealKIE 16,384)"}
# STAGE-Eval xgrammar
skipped=set()
runs={}
for s in (42,43,44):
    p=Path(f"outputs/think_on/stage_eval_think_xgrammar_s{s}.jsonl")
    if p.exists() and p.stat().st_size>200:
        rows=load(p); runs[s]=rows; skipped|={r["stem"] for r in rows if r.get("skip_reason")}
if runs:
    per_all=[agg([r for r in rows if not r.get("skip_reason")]) for rows in runs.values()]
    per_c=[agg([r for r in rows if r["stem"] not in skipped]) for rows in runs.values()]
    out["stage_eval_think_xgrammar"]={"seeds":sorted(runs),"n_skipped":len(skipped),"all":{k:ms([p[k] for p in per_all]) for k in ("PFR","EMR","SCR","NR","VA")},"compat":{k:ms([p[k] for p in per_c]) for k in ("PFR","EMR","SCR","NR","VA")}}
# RealKIE 16k
tok=AutoTokenizer.from_pretrained("/root/work/sunghee/models/Qwen3-4B")
rk={}
for s in (42,43,44):
    p=Path(f"outputs/think_on/realkie_think_s{s}.jsonl")
    if p.exists() and sum(1 for _ in open(p))==74:
        rows=load(p); nt={r["stem"]:len(tok(r["user_prompt"])["input_ids"]) for r in rows}
        h,li,rec,cnt=rk_score(rows); m=[evaluate_row(r) for r in rows]
        d={"header_va":h,"item_field_va":li,"item_recall":rec,"SCR":100*sum(x["schema_valid"] for x in m)/len(m),"PFR":100*sum(not x["no_output"] for x in m)/len(m),
           "truncated":100*sum(len(tok(r["raw_output"] or "")["input_ids"])>=16000 for r in rows)/len(rows),"buckets":{}}
        for b in ["<=4k","4-8k","8-16k",">16k"]:
            rs=[r for r in rows if bucket(nt[r["stem"]],[4096,8192,16384])==b]; hb,lb,_,_=rk_score(rs); mb=[evaluate_row(r) for r in rs]
            d["buckets"][b]={"n":len(rs),"header_va":hb,"item_field_va":lb,"SCR":100*sum(x["schema_valid"] for x in mb)/max(1,len(mb))}
        rk[s]=d
if rk:
    seeds=sorted(rk); a={"seeds":seeds,**{k:ms([rk[s][k] for s in seeds]) for k in ("header_va","item_field_va","item_recall","SCR","PFR","truncated")},"buckets":{}}
    for b in rk[seeds[0]]["buckets"]:
        a["buckets"][b]={"n":rk[seeds[0]]["buckets"][b]["n"],**{k:ms([rk[s]["buckets"][b][k] for s in seeds]) for k in ("header_va","item_field_va","SCR")}}
    out["realkie_think_16384"]=a
# ExtractBench 131k
eb={}
for s in (42,43,44):
    p=Path(f"outputs/think_on/eb131_think_s{s}.jsonl")
    if p.exists() and sum(1 for _ in open(p))==237:
        rows=load(p); nt={r["stem"]:len(tok(r["user_prompt"])["input_ids"]) for r in rows}
        d={"all":agg(rows),"buckets":{}}
        for b in ["<=4k","4-8k","8-16k","16-32k","32-64k",">64k"]:
            rs=[r for r in rows if bucket(nt[r["stem"]],[4096,8192,16384,32768,65536])==b]; d["buckets"][b]=agg(rs)
        eb[s]=d
if eb:
    seeds=sorted(eb); a={"seeds":seeds,"all":{k:ms([eb[s]["all"][k] for s in seeds]) for k in ("PFR","SCR","VA")},"buckets":{}}
    for b in eb[seeds[0]]["buckets"]:
        a["buckets"][b]={"n":eb[seeds[0]]["buckets"][b]["n"],**{k:ms([eb[s]["buckets"][b][k] for s in seeds]) for k in ("PFR","SCR","VA")}}
    out["extractbench_131k_think"]=a
Path("benchmark/paper_figures/data/think_on_summary.json").write_text(json.dumps(out,indent=2))
for k,v in out.items():
    if k=="protocol": continue
    print("==",k, "seeds", v.get("seeds"))
    if "all" in v: print("  all:", {m:f"{v['all'][m]['mean']:.1f}±{v['all'][m]['std']:.1f}" for m in v["all"]})
    if "compat" in v: print("  compat:", {m:f"{v['compat'][m]['mean']:.1f}±{v['compat'][m]['std']:.1f}" for m in v["compat"]})
    if "header_va" in v: print("  ", {m:f"{v[m]['mean']:.1f}±{v[m]['std']:.1f}" for m in ("header_va","item_field_va","item_recall","SCR","PFR","truncated")})
    for b,bv in v.get("buckets",{}).items(): print("   ",b, {m:f"{bv[m]['mean']:.1f}" for m in bv if m!="n"}, "n",bv["n"])
