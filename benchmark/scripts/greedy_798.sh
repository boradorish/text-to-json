#!/bin/bash
# Re-run the Figure 3(b) conditions with the Table 1 decoding (greedy, max_new_tokens 4096) on STAGE-Eval 851; GPU0 after Glaive.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
V=/root/work/sunghee/venv/bin/python; BASE=/root/work/sunghee/models/Qwen3-4B; SFT=/root/work/sunghee/models/STAGE-Qwen3-4B-SFT
O=outputs/greedy_798; mkdir -p $O
until grep -q "ALIGNED_DONE gpu0" /root/aligned_gpu0.log 2>/dev/null; do sleep 60; done
echo "GREEDY_START $(date -u)"
run() { local name=$1 model=$2; shift 2; [ -s $O/$name.jsonl ] && return; echo "START $name $(date -u)"
  CUDA_VISIBLE_DEVICES=0 $V benchmark/inference.py --model $model --benchmark-file benchmark/data/stage_eval_test.jsonl --output $O/$name --temperature 0.0 --max-new-tokens 4096 --max-model-len 16384 --batch-size 16 --gpu-memory-utilization 0.85 "$@" > $O/$name.log 2>&1; echo "END $name rc=$? $(date -u) rows=$(wc -l < $O/$name.jsonl 2>/dev/null)"; }
run qwen3_4b_base_nothink_free $BASE --no-thinking
run qwen3_4b_base_nothink_xgrammar $BASE --no-thinking --guided-json-backend xgrammar
run qwen3_4b_sft_free $SFT
run qwen3_4b_sft_xgrammar $SFT --guided-json-backend xgrammar
$V - <<'PY'
import json, sys
sys.path.insert(0, "benchmark"); from evaluate import evaluate_row
O = "outputs/greedy_798"; names = ["qwen3_4b_base_nothink_free", "qwen3_4b_base_nothink_xgrammar", "qwen3_4b_sft_free", "qwen3_4b_sft_xgrammar"]
runs = {n: {json.loads(l)["stem"]: json.loads(l) for l in open(f"{O}/{n}.jsonl") if l.strip()} for n in names}
skipped = {s for n in names if "xgrammar" in n for s, r in runs[n].items() if r.get("skip_reason")}
compat = [s for s in runs["qwen3_4b_sft_free"] if s not in skipped]
def agg(rows):
    ms = [evaluate_row(r) for r in rows]; n = len(ms)
    return {"n": n, "PFR": 100 * sum(not m["no_output"] for m in ms) / n, "EMR": 100 * sum(m["exact_match"] for m in ms) / n, "SCR": 100 * sum(m["schema_valid"] for m in ms) / n, "NR": 100 * sum(m["noise_ratio"] for m in ms) / n, "VA": 100 * sum(m["value_match"] for m in ms) / n}
out = {"decoding": "greedy (temperature 0), max_new_tokens 4096, max_model_len 16384, thinking disabled for the untrained model; single run", "compat_n": len(compat)}
for n in names:
    rows_all = [r for r in runs[n].values() if not r.get("skip_reason")]
    out[n] = {"all": agg(rows_all), "compat": agg([runs[n][s] for s in compat])}
    print(n, {k: round(v, 1) for k, v in out[n]["compat"].items()})
json.dump(out, open(f"{O}/summary.json", "w"), indent=2)
PY
echo "GREEDY_DONE $(date -u)"
