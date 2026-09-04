"""RealKIE scoring: header fields (normalized) + order-insensitive line-item matching."""
import json, sys, glob, os, re
import sys as _s; R=(_s.argv[1] if len(_s.argv)>1 else "outputs/realworld_realkie")
def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def J(v):
    try: return json.loads(v) if isinstance(v,str) else v
    except Exception: return None
def norm(v):
    if v is None: return "null"
    if isinstance(v,bool): return str(v).lower()
    if isinstance(v,(int,float)): return f"{float(v):.2f}"
    s=str(v).strip()
    m=re.fullmatch(r"[$]?\s*([-\d][\d,]*\.?\d*)", s)
    if m:
        try: return f"{float(m.group(1).replace(',','')):.2f}"
        except Exception: pass
    return re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
HDR=["Agency","Advertiser","GrossTotal","PaymentTerms","AgencyCommission","NetAmountDue"]
def item_sim(a,b):
    keys=set(a)|set(b); return sum(1 for k in keys if norm(a.get(k))==norm(b.get(k)))/max(1,len(keys))
def score(rows):
    hdr=[0,0]; li_field=[0,0]; li_recall=[0,0]; cnt_ok=0
    for r in rows:
        g=J(r["gold_json"]); p=J(r.get("pred_json")) or {}
        for k in HDR:
            if k in g: hdr[1]+=1; hdr[0]+= norm(p.get(k,"<M>"))==norm(g[k])
        gi=g.get("LineItems") or []; pi=(p.get("LineItems") if isinstance(p,dict) else None) or []
        pi=[x for x in pi if isinstance(x,dict)]
        cnt_ok += len(pi)==len(gi)
        used=set()
        for gitem in gi:
            best,bi=-1,None
            for j,pitem in enumerate(pi):
                if j in used: continue
                s=item_sim(gitem,pitem)
                if s>best: best,bi=s,j
            li_recall[1]+=1
            if bi is not None and best>=0.5: used.add(bi); li_recall[0]+=1
            for k,v in gitem.items():
                li_field[1]+=1
                if bi is not None and norm(pi[bi].get(k,"<M>"))==norm(v): li_field[0]+=1
    return hdr[0]/hdr[1]*100, li_field[0]/li_field[1]*100, li_recall[0]/li_recall[1]*100, cnt_ok/len(rows)*100
print(f"{'run':36} {'header VA':>10} {'item-field VA':>14} {'item recall':>12} {'count ok':>9}")
for f in sorted(glob.glob(f"{R}/*.jsonl")):
    rows=load(f)
    if not any(r.get("pred_json") for r in rows): print(f"{os.path.basename(f):36} (no outputs)"); continue
    h,lf,lr,c=score(rows); print(f"{os.path.basename(f):36} {h:10.1f} {lf:14.1f} {lr:12.1f} {c:9.1f}")

# Optional prompt-length breakdown: score_realkie.py <outputs_dir> <benchmark.jsonl with prompt_tokens> [edges]
if len(_s.argv)>2:
    bench={json.loads(l)["stem"]:json.loads(l) for l in open(_s.argv[2]) if l.strip()}
    if any("prompt_tokens" not in b for b in bench.values()):
        from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER","/root/work/sunghee/models/Qwen3-4B"))
        for b in bench.values():
            if "prompt_tokens" not in b: b["prompt_tokens"]=len(tok(b["user_prompt"])["input_ids"])
    edges=[int(x) for x in (_s.argv[3] if len(_s.argv)>3 else "4096,8192,16384").split(",")]
    def bucket(n):
        lo=0
        for e in edges:
            if n<=e: return f"{lo//1024}-{e//1024}k"
            lo=e
        return f">{lo//1024}k"
    print("\n-- header VA / item-field VA by prompt-token bucket --")
    files=[f for f in sorted(glob.glob(f"{R}/*.jsonl")) if any(r.get("pred_json") for r in load(f))]
    print(f"{'bucket':10}{'n':>5}"+"".join(f"{os.path.basename(f)[:22]:>24}" for f in files))
    for bk in sorted({bucket(b["prompt_tokens"]) for b in bench.values()}, key=lambda k:(k.startswith(">"),int(k.lstrip(">").split("-")[0].rstrip("k")))):
        cells=[]; n=0
        for f in files:
            rows=[r for r in load(f) if r["stem"] in bench and bucket(bench[r["stem"]]["prompt_tokens"])==bk]; n=len(rows)
            h,lf,lr,c=score(rows) if rows else (0,0,0,0); cells.append(f"{h:>11.1f}/{lf:<12.1f}")
        print(f"{bk:10}{n:>5}"+"".join(f"{c:>24}" for c in cells))
