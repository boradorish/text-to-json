#!/bin/bash
# Qwen2.5-3B (base) and + STAGE on RealKIE-FCC 74, 16,384-token budget (Figure 4a / Figure 7 protocol), seeds 42/43/44.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; O=outputs/realkie_longout; BR=benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl
BASE=/mnt/nvme/cache/interns/hf/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
SFT=/mnt/nvme/cache/interns/hf/hub/models--boradorish--qwen2.5-3b-sft/snapshots/3de5c63967fda091ec22270b104078eba2babae5
echo "RK_Q25_START $(date -u)"
run() { local gpu=$1 name=$2 model=$3 seed=$4; [ -s $O/${name}_s$seed.jsonl ] && return; echo "START ${name}_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model $model --benchmark-file $BR --output $O/${name}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens 16384 --max-model-len 40960 --batch-size 4 --enforce-eager --gpu-memory-utilization 0.85 > $O/${name}_s$seed.log 2>&1; echo "END ${name}_s$seed rc=$? $(date -u) rows=$(wc -l < $O/${name}_s$seed.jsonl 2>/dev/null)"; }
( for s in 42 43 44; do run 0 q25_base $BASE $s; done; echo LANE0_DONE ) &
( for s in 42 43 44; do run 1 q25_sft $SFT $s; done; echo LANE1_DONE ) &
wait; echo "RK_Q25_DONE $(date -u)"
