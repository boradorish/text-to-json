#!/bin/bash
# Qwen2.5-3B (base) and Qwen2.5-3B + STAGE on ExtractBench 237 @131k YaRN, seeds 42/43/44, Figure 6 protocol (3,100 tokens, temperature 0.6).
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; O=outputs/extractbench_sampling3; B131=benchmark/data/extractbench_context131072.jsonl
BASE=/mnt/nvme/cache/interns/hf/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
SFT=/mnt/nvme/cache/interns/hf/hub/models--boradorish--qwen2.5-3b-sft/snapshots/3de5c63967fda091ec22270b104078eba2babae5
YARN="{\"max_position_embeddings\":131072,\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}"
echo "EB131_Q25_START $(date -u)"
run() { local gpu=$1 name=$2 model=$3 seed=$4; [ -s $O/${name}_s$seed.jsonl ] && return; echo "START ${name}_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model $model --benchmark-file $B131 --output $O/${name}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens 3100 --max-model-len 131072 --batch-size 2 --enforce-eager --gpu-memory-utilization 0.85 --hf-overrides "$YARN" > $O/${name}_s$seed.log 2>&1; echo "END ${name}_s$seed rc=$? $(date -u) rows=$(wc -l < $O/${name}_s$seed.jsonl 2>/dev/null)"; }
( for s in 42 43 44; do run 0 q25_base_yarn $BASE $s; done; echo LANE0_DONE ) &
( for s in 42 43 44; do run 1 q25_sft_yarn $SFT $s; done; echo LANE1_DONE ) &
wait; echo "EB131_Q25_DONE $(date -u)"
