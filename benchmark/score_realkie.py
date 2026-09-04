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
