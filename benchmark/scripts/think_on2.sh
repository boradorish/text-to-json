#!/bin/bash
# Untrained Qwen3-4B with thinking ENABLED (Table 2 setting): STAGE-Eval xgrammar + free (GPU1), then RealKIE 16k and ExtractBench 131k (GPU0 after the Qwen2.5 run).
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; BASE=/root/work/sunghee/models/Qwen3-4B; O=outputs/think_on; mkdir -p $O
SE=benchmark/data/stage_eval_test.jsonl; BR=benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl; B131=benchmark/data/extractbench_context131072.jsonl
YARN="{\"max_position_embeddings\":131072,\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}"
echo "THINK2_START $(date -u)"
run() { local gpu=$1 name=$2 bench=$3 seed=$4 mnt=$5; shift 5; [ -s $O/${name}_s$seed.jsonl ] && return; rm -f $O/${name}_s$seed.jsonl; echo "START ${name}_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model $BASE --benchmark-file $bench --output $O/${name}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens $mnt --gpu-memory-utilization 0.85 --enforce-eager "$@" > $O/${name}_s$seed.log 2>&1; echo "END ${name}_s$seed rc=$? $(date -u) rows=$(wc -l < $O/${name}_s$seed.jsonl 2>/dev/null)"; }
laneB() { for s in 42 43 44; do run 1 stage_eval_think_xgrammar $SE $s 3100 --max-model-len 16384 --batch-size 16 --guided-json-backend xgrammar; done
  for s in 42 43 44; do run 1 stage_eval_think_free $SE $s 3100 --max-model-len 16384 --batch-size 16; done; echo LANE_B_DONE; }
laneA() { until grep -q RK_Q25_DONE /root/work/sunghee/runners/rk_qwen25.log 2>/dev/null; do sleep 30; done
  for s in 42 43 44; do run 0 realkie_think $BR $s 16384 --max-model-len 40960 --batch-size 4; done
  run 0 eb131_think $B131 42 3100 --max-model-len 131072 --batch-size 2 --hf-overrides "$YARN"; echo LANE_A_DONE; }
laneA & laneB & wait; echo "THINK2_DONE $(date -u)"
