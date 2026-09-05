#!/bin/bash
# Aligned full-FT baselines: comparison datasets trained with the STAGE recipe (train_full_sft.py defaults = Table 3),
# then STAGE-Eval 851 inference (thinking off) and rule scoring. Usage: aligned_ft.sh <gpu> <name> <train jsonl> [<name2> <jsonl2> ...]
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
V=/root/work/sunghee/venv/bin/python; BASE=/root/work/sunghee/models/Qwen3-4B
GPU=$1; shift
O=outputs/aligned_baselines; mkdir -p $O/eval
while [ $# -ge 2 ]; do
  name=$1; data=$2; shift 2
  if [ ! -f $O/$name/model.safetensors ] && [ ! -f $O/$name/model.safetensors.index.json ]; then
    echo "TRAIN_START $name $(date -u)"
    CUDA_VISIBLE_DEVICES=$GPU $V benchmark/train_full_sft.py --model $BASE --train-data $data --output $O/$name > $O/$name.train.log 2>&1
    echo "TRAIN_END $name rc=$? $(date -u)"
  else echo "skip train $name"; fi
  if [ -f $O/$name/config.json ] && [ ! -s $O/eval/${name}_stage_eval851.jsonl ]; then
    echo "EVAL_START $name $(date -u)"
    CUDA_VISIBLE_DEVICES=$GPU $V benchmark/inference.py --model $O/$name --benchmark-file benchmark/data/stage_eval_test.jsonl --output $O/eval/${name}_stage_eval851 --no-thinking --batch-size 16 --max-model-len 16384 --gpu-memory-utilization 0.85 > $O/$name.eval.log 2>&1
    $V benchmark/evaluate.py --input $O/eval/${name}_stage_eval851.jsonl > $O/eval/${name}_stage_eval851_eval.txt 2>&1
    echo "EVAL_END $name rc=$? $(date -u)"; grep -E "no_output|exact_match|schema_valid|value_match" $O/eval/${name}_stage_eval851_eval.txt
  fi
done
echo "ALIGNED_DONE gpu$GPU $(date -u)"
