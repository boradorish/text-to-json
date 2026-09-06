#!/bin/bash
# RealKIE-FCC 74 with a 16,384-token generation budget instead of 3,100 (truncation check for Figure 4a). Same sampling protocol otherwise.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; BASE=/root/work/sunghee/models/Qwen3-4B; SFT=/root/work/sunghee/models/STAGE-Qwen3-4B-SFT
BR=benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl; OUT=outputs/realkie_longout; mkdir -p $OUT
echo "RKLONG_START $(date -u)"
run() { local gpu=$1 out=$2 model=$3 seed=$4; shift 4; [ -s ${out}_s$seed.jsonl ] && return; echo "START $(basename $out)_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model $model --benchmark-file $BR --output ${out}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens 16384 --max-model-len 40960 --batch-size 4 --gpu-memory-utilization 0.85 "$@" > ${out}_s$seed.log 2>&1; echo "END $(basename $out)_s$seed rc=$? $(date -u) rows=$(wc -l < ${out}_s$seed.jsonl 2>/dev/null)"; }
( for s in 42 43 44; do run 0 $OUT/base_nothink $BASE $s --no-thinking; done; echo LANE0_DONE ) &
( for s in 42 43 44; do run 1 $OUT/sft $SFT $s; done; echo LANE1_DONE ) &
wait; echo "RKLONG_DONE $(date -u)"
