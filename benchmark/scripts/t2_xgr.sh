#!/bin/bash
# Table 2 STAGE checkpoint (boradorish/qwen3-4b-new) on STAGE-Eval: xgrammar (GPU0) and free decoding (GPU1), seeds 42/43/44, 3,100 tokens.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/mnt/nvme/cache/interns/hf VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
V=/root/work/sunghee/venv/bin/python; O=outputs/t2_ckpt; mkdir -p $O; SE=benchmark/data/stage_eval_test.jsonl
echo "T2_START $(date -u)"
M=$(HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 $V - <<'PY'
from huggingface_hub import snapshot_download
try:
    print(snapshot_download("boradorish/qwen3-4b-new", allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"]))
except Exception as e:
    print("DOWNLOAD_FAILED", type(e).__name__, str(e)[:200])
PY
)
echo "MODEL=$M"
case "$M" in *DOWNLOAD_FAILED*) echo "T2_DONE_FAILED $(date -u)"; exit 1;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run() { local gpu=$1 name=$2 seed=$3; shift 3; [ -s $O/${name}_s$seed.jsonl ] && return; echo "START ${name}_s$seed $(date -u)"
  CUDA_VISIBLE_DEVICES=$gpu $V benchmark/inference.py --model "$M" --benchmark-file $SE --output $O/${name}_s$seed --temperature 0.6 --top-p 1.0 --seed $seed --max-new-tokens 3100 --max-model-len 16384 --batch-size 16 --gpu-memory-utilization 0.85 "$@" > $O/${name}_s$seed.log 2>&1; echo "END ${name}_s$seed rc=$? $(date -u) rows=$(wc -l < $O/${name}_s$seed.jsonl 2>/dev/null)"; }
( for s in 42 43 44; do run 0 t2_xgrammar $s --guided-json-backend xgrammar; done; echo LANE0_DONE ) &
( for s in 42 43 44; do run 1 t2_free $s; done; echo LANE1_DONE ) &
wait; echo "T2_DONE $(date -u)"
