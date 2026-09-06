import json,re,statistics
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/root/work/sunghee/models/Qwen3-4B")
def load(p): return [json.loads(l) for l in open(p)]
def bucket(n):
    for e,l in [(4096,"<=4k"),(8192,"4-8k"),(16384,"8-16k")]:
        if n<=e: return l
    return ">16k"
def norm(v):
    s=str(v).lower().strip(); s=re.sub(r"\s+"," ",s).replace(",","").replace("$","")
    if re.fullmatch(r"-?\d+\.0+",s): s=s[:s.index(".")]
    return s
HDR=("Agency","Advertiser","GrossTotal","PaymentTerms","AgencyCommission","NetAmountDue")
def hdr_from_raw(raw):
    out={}
    for k in HDR:
        m=re.search(r'"%s"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|null)'%k, raw or "")
        if m:
            try: out[k]=json.loads(m.group(1))
            except Exception: pass
    return out
B,T="base_nothink","sft"
R={c:{r["stem"]:r for r in load(f"outputs/realkie_sampling3/{c}_s42.jsonl")} for c in (B,T)}
stems=sorted(R[T]); nt={s:len(tok(R[T][s]["user_prompt"])["input_ids"]) for s in stems}
print("bucket n | truncated@3100 base/STAGE | mean out tok base/STAGE | gold items/doc, pred items base/STAGE | header VA as scored base/STAGE | header VA from raw prefix base/STAGE")
for b in ["<=4k","4-8k","8-16k",">16k"]:
    S=[s for s in stems if bucket(nt[s])==b]
    tr={};ot={};items={};va={};vraw={}
    for c in (B,T):
        tr[c]=ot[c]=0; items[c]=[]; hit=[0,0]; hraw=[0,0]
        for s in S:
            r=R[c][s]; raw=r["raw_output"] or ""; n=len(tok(raw)["input_ids"]); tr[c]+=n>=3000; ot[c]+=n
            g=json.loads(r["gold_json"])
            try: p=json.loads(r["pred_json"])
            except Exception: p=None
            p=p if isinstance(p,dict) else {}
            li=p.get("LineItems"); items[c].append(len(li) if isinstance(li,list) else 0)
            h=hdr_from_raw(raw)
            for k in HDR:
                if k in g:
                    hit[1]+=1; hit[0]+=norm(p.get(k,"<M>"))==norm(g[k]); hraw[1]+=1; hraw[0]+=norm(h.get(k,"<M>"))==norm(g[k])
        va[c]=100*hit[0]/max(1,hit[1]); vraw[c]=100*hraw[0]/max(1,hraw[1])
    gi=statistics.mean(len(json.loads(R[T][s]["gold_json"]).get("LineItems") or []) for s in S)
    print(f"{b:6s} {len(S):2d} | {tr[B]}/{tr[T]} | {ot[B]/len(S):.0f}/{ot[T]/len(S):.0f} | {gi:.1f}, {statistics.mean(items[B]):.1f}/{statistics.mean(items[T]):.1f} | {va[B]:.1f}/{va[T]:.1f} | {vraw[B]:.1f}/{vraw[T]:.1f}")
# how long are STAGE vs base line items (tokens per item) on truncated docs?
print("\nper-line-item tokens (untruncated docs only): base / STAGE")
for c in (B,T):
    per=[]
    for s in stems:
        r=R[c][s]; raw=r["raw_output"] or ""; n=len(tok(raw)["input_ids"])
        try: p=json.loads(r["pred_json"])
        except Exception: continue
        li=p.get("LineItems") if isinstance(p,dict) else None
        if n<3000 and isinstance(li,list) and len(li)>=3: per.append(n/len(li))
    print(c, f"median {statistics.median(per):.0f} tok/item over {len(per)} docs")
