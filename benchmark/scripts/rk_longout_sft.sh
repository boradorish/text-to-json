#!/bin/bash
# Pod has a single GPU: run the STAGE lane of rk_longout.sh after the base lane finishes.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; SFT=/root/work/sunghee/models/STAGE-Qwen3-4B-SFT
BR=benchmark/data/realworld/realkie_fcc_verified_ctx40960.jsonl; OUT=outputs/realkie_longout
until grep -q LANE0_DONE /root/work/sunghee/runners/rk_longout.log 2>/dev/null; do sleep 60; done
echo "RKLONG_SFT_START $(date -u)"
for s in 42 43 44; do echo "START sft_s$s $(date -u)"
  CUDA_VISIBLE_DEVICES=0 $V benchmark/inference.py --model $SFT --benchmark-file $BR --output $OUT/sft_s$s --temperature 0.6 --top-p 1.0 --seed $s --max-new-tokens 16384 --max-model-len 40960 --batch-size 4 --gpu-memory-utilization 0.85 > $OUT/sft_s$s.log 2>&1; echo "END sft_s$s rc=$? $(date -u) rows=$(wc -l < $OUT/sft_s$s.jsonl 2>/dev/null)"; done
echo "RKLONG_SFT_DONE $(date -u)"
