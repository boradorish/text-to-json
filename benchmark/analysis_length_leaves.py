"""Why does STAGE's value accuracy rise with length on RealKIE-FCC but fall on ExtractBench?
Leaf-level breakdown per length bucket: gold leaves split into verbatim-in-source / derived (not in source) / null,
accuracy of base vs STAGE on each, and error type (missing / null / wrong)."""
import json, re, sys, statistics
from pathlib import Path
sys.path.insert(0, "benchmark")
from evaluate import extract_leaves, evaluate_row
from transformers import AutoTokenizer
src = Path("benchmark/score_seeds_realworld.py").read_text()
tokpath = "/root/work/sunghee/models/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(tokpath)

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def J(v):
    if isinstance(v, (dict, list)) or v is None: return v
    try: return json.loads(v)
    except Exception: return None
def parse_pred(r):
    return J(r.get("pred_json"))  # same object the paper scorer (evaluate_row) uses
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"\s+", " ", s); s = s.replace(",", "")
    if re.fullmatch(r"-?\d+\.0+", s): s = s[: s.index(".")]
    return s
def in_source(v, text):
    if isinstance(v, bool) or v is None: return False
    n = norm(v)
    if not n: return False
    if isinstance(v, (int, float)):
        cands = {n, f"{v:,}", f"{v:,.2f}", f"{v:.2f}", str(v)}
        return any(norm(c) in text for c in cands)
    return n in text
def leaf_eq(p, g):
    return p == g  # exact equality, as in evaluate.value_match
def bucket(n, edges):
    lo = 0
    for e in edges:
        if n <= e: return f"{lo//1024}-{e//1024}k" if lo else f"<={e//1024}k"
        lo = e
    return f">{lo//1024}k"

def analyse(name, files, edges, leaf_filter=None):
    conds = {c: {r["stem"]: r for r in load(p)} for c, p in files.items()}
    base_c, stage_c = list(files)
    stems = sorted(conds[base_c]); ntok = {s: len(tok(conds[base_c][s]["user_prompt"])["input_ids"]) for s in stems}
    text = {s: norm(conds[base_c][s]["user_prompt"]) for s in stems}
    print(f"\n#### {name}: {len(stems)} docs")
    agg = {}
    for s in stems:
        b = bucket(ntok[s], edges); g = J(conds[base_c][s]["gold_json"]); gl = extract_leaves(g)
        if leaf_filter: gl = {k: v for k, v in gl.items() if leaf_filter(k)}
        A = agg.setdefault(b, {"docs": 0, "leaves": 0, "cat": {"verbatim": 0, "derived": 0, "null": 0}, "hit": {c: {"verbatim": 0, "derived": 0, "null": 0} for c in files}, "err": {c: {"missing": 0, "null": 0, "wrong": 0} for c in files}, "docva": {c: [] for c in files}, "parsed": {c: 0 for c in files}})
        A["docs"] += 1; A["leaves"] += len(gl)
        preds = {c: parse_pred(conds[c][s]) for c in files}
        pl = {c: (extract_leaves(preds[c]) if isinstance(preds[c], (dict, list)) else {}) for c in files}
        for c in files: A["parsed"][c] += isinstance(preds[c], (dict, list))
        for c in files: A["docva"][c].append(evaluate_row(conds[c][s])["value_match"])
        for k, v in gl.items():
            cat = "null" if v is None else ("verbatim" if in_source(v, text[s]) else "derived"); A["cat"][cat] += 1
            for c in files:
                if k not in pl[c]:
                    if v is None: A["hit"][c][cat] += 1
                    else: A["err"][c]["missing"] += 1
                    continue
                if leaf_eq(pl[c][k], v): A["hit"][c][cat] += 1
                elif pl[c][k] is None: A["err"][c]["null"] += 1
                else: A["err"][c]["wrong"] += 1
    order = [b for b in ["<=4k", "4-8k", "8-16k", "16-32k", "32-64k", ">64k", ">16k"] if b in agg]
    print(f"{'bucket':7s} {'n':>3s} {'leaves/doc':>10s} | share verbatim/derived/null | leaf acc verbatim base/STAGE | derived base/STAGE | null base/STAGE | docVA(base/STAGE) | STAGE errors miss/null/wrong | base errors miss/null/wrong")
    for b in order:
        A = agg[b]; L = max(1, A["leaves"]); sh = {k: 100 * A["cat"][k] / L for k in A["cat"]}
        acc = lambda c, k: 100 * A["hit"][c][k] / max(1, A["cat"][k])
        print(f"{b:7s} {A['docs']:3d} {A['leaves']/A['docs']:10.1f} | {sh['verbatim']:5.1f}/{sh['derived']:5.1f}/{sh['null']:5.1f} | {acc(base_c,'verbatim'):5.1f}/{acc(stage_c,'verbatim'):5.1f} | {acc(base_c,'derived'):5.1f}/{acc(stage_c,'derived'):5.1f} | {acc(base_c,'null'):5.1f}/{acc(stage_c,'null'):5.1f} | {100*statistics.mean(A['docva'][base_c]):5.1f}/{100*statistics.mean(A['docva'][stage_c]):5.1f} | {A['err'][stage_c]['missing']:4d}/{A['err'][stage_c]['null']:4d}/{A['err'][stage_c]['wrong']:4d} | {A['err'][base_c]['missing']:4d}/{A['err'][base_c]['null']:4d}/{A['err'][base_c]['wrong']:4d}")
    return agg

EB = {"base_nothink_yarn": "outputs/extractbench_sampling3/base_nothink_yarn_s42.jsonl", "sft_yarn": "outputs/extractbench_sampling3/sft_yarn_s42.jsonl"}
analyse("ExtractBench 237 @131k (seed 42), all gold leaves", EB, [4096, 8192, 16384, 32768, 65536])
RK = {"base_nothink": "outputs/realkie_sampling3/base_nothink_s42.jsonl", "sft": "outputs/realkie_sampling3/sft_s42.jsonl"}
HDR = ("Agency", "Advertiser", "GrossTotal", "PaymentTerms", "AgencyCommission", "NetAmountDue")
analyse("RealKIE-FCC 74 (seed 42), header fields only", RK, [4096, 8192, 16384], leaf_filter=lambda k: k.split(".")[0].split("[")[0] in HDR and "LineItems" not in k)
analyse("RealKIE-FCC 74 (seed 42), line-item leaves only", RK, [4096, 8192, 16384], leaf_filter=lambda k: "LineItems" in k)
# ExtractBench: which document families sit in each bucket (stem prefix before '__')
rows = load(EB["base_nothink_yarn"]); fam = {}
for r in rows:
    b = bucket(len(tok(r["user_prompt"])["input_ids"]), [4096, 8192, 16384, 32768, 65536]); fam.setdefault(b, {}); f = r["stem"].split("__")[0]; fam[b][f] = fam[b].get(f, 0) + 1
print("\n#### ExtractBench document families per bucket"); [print(b, fam[b]) for b in fam]
# leaf-type mix per bucket (numbers / strings / bools) in gold
mix = {}
for r in rows:
    b = bucket(len(tok(r["user_prompt"])["input_ids"]), [4096, 8192, 16384, 32768, 65536]); m = mix.setdefault(b, {"num": 0, "str": 0, "bool": 0, "null": 0, "strlen": []})
    for k, v in extract_leaves(J(r["gold_json"])).items():
        if v is None: m["null"] += 1
        elif isinstance(v, bool): m["bool"] += 1
        elif isinstance(v, (int, float)): m["num"] += 1
        else: m["str"] += 1; m["strlen"].append(len(str(v)))
print("\n#### gold leaf type mix per bucket (share %) and median string length")
for b, m in mix.items():
    T = max(1, m["num"] + m["str"] + m["bool"] + m["null"]); print(b, {k: round(100 * m[k] / T, 1) for k in ("num", "str", "bool", "null")}, "median strlen", statistics.median(m["strlen"]) if m["strlen"] else None)
