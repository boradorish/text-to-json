#!/bin/bash
# Rebuild the JSONSchemaBench comparison data the way the paper describes (full 5.75K train schemas, gold JSON and
# short report synthesised with Qwen3-4B via --gold-mode llm), then train with the STAGE recipe and evaluate. GPU0, after Glaive.
cd /root/work/sunghee/text-to-json
export CUDA_DEVICE_ORDER=PCI_BUS_ID TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 HF_HOME=/mnt/nvme/cache/interns/hf
V=/root/work/sunghee/venv/bin/python
until grep -q "ALIGNED_DONE gpu0" /root/aligned_gpu0.log 2>/dev/null; do sleep 120; done
echo "JSB_GEN_START $(date -u)"
CUDA_VISIBLE_DEVICES=0 $V src/prepare_jsonschemabench_report_sft.py --gold-mode llm --split train --num-samples 6000 --batch-size 64 --output data/sft/jsonschemabench_report_llm_full.jsonl > outputs/aligned_baselines/jsonschemabench_llm.gen.log 2>&1
echo "JSB_GEN_END rc=$? rows=$(wc -l < data/sft/jsonschemabench_report_llm_full.jsonl 2>/dev/null) $(date -u)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/aligned_ft.sh 0 jsonschemabench_llm_full data/sft/jsonschemabench_report_llm_full.jsonl
echo "JSB_DONE $(date -u)"
