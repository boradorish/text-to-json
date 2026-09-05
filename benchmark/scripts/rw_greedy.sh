#!/bin/bash
# Greedy (Table 1 decoding) re-run of the real-world experiments: RealKIE-FCC 74 (5 conditions), ExtractBench 32k (4 conditions, 200 docs -> 194 compat), ExtractBench 131k YaRN (base, STAGE). GPU0 after greedy_798.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; BASE=/root/work/sunghee/models/Qwen3-4B; SFT=/root/work/sunghee/models/STAGE-Qwen3-4B-SFT; DLG=outputs/stage_dialog/lora_stage_sft_mix_v2
YARN='{"max_position_embeddings":131072,"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
until grep -q "GREEDY_DONE" /root/greedy_798.log 2>/dev/null; do sleep 60; done
MEM=0.5; grep -q "ALIGNED_DONE gpu0" /root/aligned_gpu0.log 2>/dev/null && MEM=0.85
echo "RW_GREEDY_START $(date -u) mem=$MEM"
run() { local out=$1 model=$2 bench=$3; shift 3; [ -s $out.jsonl ] && { echo "skip $out"; return; }; echo "START $(basename $out) $(date -u)"
  CUDA_VISIBLE_DEVICES=0 $V benchmark/inference.py --model $model --benchmark-file $bench --output $out --temperature 0.0 --max-new-tokens 4096 --gpu-memory-utilization $MEM "$@" > $out.log 2>&1; echo "END $(basename $out) rc=$? $(date -u) rows=$(wc -l < $out.jsonl 2>/dev/null)"; }
R=outputs/realworld_realkie_greedy; mkdir -p $R; B=benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl
run $R/qwen3_4b_base_nothink $BASE $B --no-thinking --batch-size 4 --max-model-len 40960
run $R/qwen3_4b_base_nothink_xgrammar $BASE $B --no-thinking --batch-size 4 --max-model-len 40960 --guided-json-backend xgrammar
run $R/qwen3_4b_stage_sft $SFT $B --batch-size 4 --max-model-len 40960
run $R/qwen3_4b_stage_sft_xgrammar $SFT $B --batch-size 4 --max-model-len 40960 --guided-json-backend xgrammar
run $R/qwen3_4b_stage_dialog_v2 $DLG $B --batch-size 4 --max-model-len 40960
$V benchmark/score_realkie.py $R $B > $R/score.txt 2>&1; cp outputs/length_buckets/realkie_header.json $R/realkie_header_buckets.json 2>/dev/null
E=outputs/extractbench_greedy; mkdir -p $E; B32=benchmark/data/extractbench_context32768.jsonl
run $E/qwen3_4b_base_nothink_free $BASE $B32 --no-thinking --batch-size 4 --max-model-len 36864
run $E/qwen3_4b_base_nothink_xgrammar $BASE $B32 --no-thinking --batch-size 4 --max-model-len 36864 --guided-json-backend xgrammar
run $E/qwen3_4b_sft_free $SFT $B32 --batch-size 4 --max-model-len 36864
run $E/qwen3_4b_sft_xgrammar $SFT $B32 --batch-size 4 --max-model-len 36864 --guided-json-backend xgrammar
B131=benchmark/data/extractbench_context131072.jsonl
run $E/qwen3_4b_base_nothink_yarn $BASE $B131 --no-thinking --batch-size 2 --max-model-len 131072 --enforce-eager --hf-overrides "$YARN"
run $E/qwen3_4b_stage_sft_yarn $SFT $B131 --batch-size 2 --max-model-len 131072 --enforce-eager --hf-overrides "$YARN"
$V benchmark/length_bucket_analysis.py --benchmark $B131 --tokenizer $BASE base=$E/qwen3_4b_base_nothink_yarn.jsonl sft=$E/qwen3_4b_stage_sft_yarn.jsonl --edges 4096,8192,16384,32768,65536 --out $E/extractbench_long_buckets.json > $E/score_long.txt 2>&1
$V - <<'PY'
import json, sys
sys.path.insert(0, "benchmark"); from evaluate import evaluate_row
E = "outputs/extractbench_greedy"; names = ["qwen3_4b_base_nothink_free", "qwen3_4b_base_nothink_xgrammar", "qwen3_4b_sft_free", "qwen3_4b_sft_xgrammar"]
runs = {n: {json.loads(l)["stem"]: json.loads(l) for l in open(f"{E}/{n}.jsonl") if l.strip()} for n in names}
skipped = {s for n in names if "xgrammar" in n for s, r in runs[n].items() if r.get("skip_reason")}
bench = {json.loads(l)["stem"]: json.loads(l) for l in open("benchmark/data/extractbench_context32768.jsonl")}
compat = [s for s in runs["qwen3_4b_sft_free"] if s not in skipped]
def agg(rows):
    ms = [evaluate_row(r) for r in rows]; n = max(1, len(ms))
    return {"n": len(ms), "PFR": 100 * sum(not m["no_output"] for m in ms) / n, "SCR": 100 * sum(m["schema_valid"] for m in ms) / n, "VA": 100 * sum(m["value_match"] for m in ms) / n, "NR": 100 * sum(m["noise_ratio"] for m in ms) / n}
out = {"compat_n": len(compat), "decoding": "greedy, max_new_tokens 4096"}
for n in names:
    rows = [runs[n][s] for s in compat]
    out[n] = {"all194": agg(rows), "short": agg([r for r in rows if bench[r["stem"]]["source_split"].endswith("short")]), "medium": agg([r for r in rows if not bench[r["stem"]]["source_split"].endswith("short")])}
    print(n, {k: round(v, 1) for k, v in out[n]["all194"].items()})
json.dump(out, open(f"{E}/summary_194.json", "w"), indent=2)
PY
echo "RW_GREEDY_DONE $(date -u)"
