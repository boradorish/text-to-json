#!/bin/bash
# Qwen2.5-3B (base) and + STAGE on the 205 ExtractBench digital documents within the native 32k context (no YaRN; the model
# crashes on >32k inputs under YaRN in this vLLM build). Seeds 42/43/44, temperature 0.6, 3,100 tokens.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; O=outputs/extractbench_sampling3; B=benchmark/data/extractbench_context131072_le32k.jsonl
BASE=/mnt/nvme/cache/interns/hf/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
SFT=/mnt/nvme/cache/interns/hf/hub/models--boradorish--qwen2.5-3b-sft/snapshots/3de5c63967fda091ec22270b104078eba2babae5
$V - <<PY
import json
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("$BASE")
rows=[json.loads(l) for l in open("benchmark/data/extractbench_context131072.jsonl")]
keep=[r for r in rows if len(tok(r["user_prompt"])["input_ids"])<=32768]
open("$B","w").write("".join(json.dumps(r)+"\n" for r in keep)); print("kept", len(keep))
PY
for c in q25_base_yarn q25_sft_yarn; do for s in 42 43 44; do mv -f $O/${c}_s$s.jsonl $O/${c}_s$s.failed.jsonl 2>/dev/null; done; done
echo "EB32_Q25_START $(date -u)"
run() { local gpu=$1 name=$2 model=$3 seed=$4; [ -s $O/${name}_s$seed.jsonl ] && return; echo "START ${name}_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model $model --benchmark-file $B --output $O/${name}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens 3100 --max-model-len 36864 --batch-size 4 --enforce-eager --gpu-memory-utilization 0.85 > $O/${name}_s$seed.log 2>&1; echo "END ${name}_s$seed rc=$? $(date -u) rows=$(wc -l < $O/${name}_s$seed.jsonl 2>/dev/null)"; }
( for s in 42 43 44; do run 0 q25_base_32k $BASE $s; done; echo LANE0_DONE ) &
( for s in 42 43 44; do run 1 q25_sft_32k $SFT $s; done; echo LANE1_DONE ) &
wait; echo "EB32_Q25_DONE $(date -u)"
