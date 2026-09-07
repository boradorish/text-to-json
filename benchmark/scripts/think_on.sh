#!/bin/bash
# Untrained Qwen3-4B with thinking ENABLED (the Table 1 setting) under the Figure 3 / Figure 4 protocols:
# GPU0: STAGE-Eval 851, free + xgrammar, seeds 42/43/44, 3,100 tokens.
# GPU1: RealKIE-FCC 74 (16,384-token budget), ExtractBench 32k (free + xgrammar), ExtractBench 131k YaRN; seeds 42/43/44.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; BASE=/root/work/sunghee/models/Qwen3-4B; O=outputs/think_on; mkdir -p $O
SE=benchmark/data/stage_eval_test.jsonl; BR=benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl; B32=benchmark/data/extractbench_context32768.jsonl; B131=benchmark/data/extractbench_context131072.jsonl
YARN="{\"max_position_embeddings\":131072,\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}"
echo "THINK_ON_START $(date -u)"
run() { local gpu=$1 name=$2 bench=$3 seed=$4 mnt=$5; shift 5; [ -s $O/${name}_s$seed.jsonl ] && return; echo "START ${name}_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model $BASE --benchmark-file $bench --output $O/${name}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens $mnt --gpu-memory-utilization 0.85 "$@" > $O/${name}_s$seed.log 2>&1; echo "END ${name}_s$seed rc=$? $(date -u) rows=$(wc -l < $O/${name}_s$seed.jsonl 2>/dev/null)"; }
laneA() { for s in 42 43 44; do
    run 0 stage_eval_think_free $SE $s 3100 --max-model-len 16384 --batch-size 16
    run 0 stage_eval_think_xgrammar $SE $s 3100 --max-model-len 16384 --batch-size 16 --guided-json-backend xgrammar; done; echo LANE_A_DONE; }
laneB() { for s in 42 43 44; do run 1 realkie_think $BR $s 16384 --max-model-len 40960 --batch-size 4; done
  for s in 42 43 44; do
    run 1 eb32_think_free $B32 $s 3100 --max-model-len 36864 --batch-size 4
    run 1 eb32_think_xgrammar $B32 $s 3100 --max-model-len 36864 --batch-size 4 --guided-json-backend xgrammar; done
  for s in 42 43 44; do run 1 eb131_think $B131 $s 3100 --max-model-len 131072 --batch-size 2 --enforce-eager --hf-overrides "$YARN"; done; echo LANE_B_DONE; }
laneA & laneB & wait
echo "THINK_ON_DONE $(date -u)"
