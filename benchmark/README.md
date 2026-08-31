# text-to-json benchmark

Minimal reproduction path for the text-to-json benchmark:

1. download the published HF benchmark split
2. run inference with a local or HF model
3. evaluate deterministic JSON/schema metrics

## Install

```bash
pip install -r benchmark/requirements.txt
```

## 1. Download Benchmark Data

```bash
python benchmark/download_benchmark.py \
  --dataset boradorish/text-to-json-benchmark \
  --split test \
  --output benchmark/data/test.jsonl
```

## 2. Run Inference

`--model` may be either a local path or a Hugging Face model id.

```bash
python benchmark/inference.py \
  --benchmark-source local \
  --benchmark-file benchmark/data/test.jsonl \
  --model saves/qwen3-0.6b/full/sft \
  --batch-size 4 \
  --max-model-len 8192 \
  --output benchmark/runs/qwen3_0_6b_sft
```

You can also skip the local download and stream/load directly from HF:

```bash
python benchmark/inference.py \
  --benchmark-source hf \
  --hf-split test \
  --model saves/qwen3-0.6b/full/sft \
  --max-model-len 8192 \
  --output benchmark/runs/qwen3_0_6b_sft
```

The progress bar is global over the full benchmark, independent of batch size.
The script writes both `.jsonl` and `.xlsx` outputs and resumes from an
existing JSONL file if present.

Inference does not truncate benchmark inputs. Generation is capped by
`--max-new-tokens`, which defaults to `3100`. If you set vLLM
`--max-model-len`, make sure it is larger than `max input tokens + max new
tokens`.

## 3. Evaluate

```bash
python benchmark/evaluate.py \
  --input benchmark/runs/qwen3_0_6b_sft.jsonl
```

Evaluation intentionally excludes LLM-as-a-judge semantic scoring. It reports
only deterministic metrics: parse/no-output, exact match, JSON Schema validity,
noise ratio, and rule-based leaf value match.

The output Excel has two sheets:

- `rows`: per-sample metrics
- `summary`: aggregate benchmark metrics
