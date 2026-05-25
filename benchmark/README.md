# text-to-json benchmark

This folder contains the fixed benchmark data and the scripts needed to rerun
inference and evaluation.

## 1. Build benchmark data

From the repo root:

```bash
python benchmark/prepare_benchmark.py
```

The script mirrors `src/train/prepare_dataset.ipynb`: it loads valid rows,
shuffles them with seed `42`, creates the 90/10 train/test split, optionally
filters test rows by max input tokens, then selects a seed-stable random sample.

Outputs:

- `benchmark/benchmark_samples.jsonl`
- `benchmark/benchmark_samples.xlsx`
- `benchmark/test_stems.txt`
- `benchmark/hf_splits/train.jsonl`
- `benchmark/hf_splits/test.jsonl`
- `benchmark/benchmark_metadata.json`

To build directly from Hugging Face:

```bash
HF_TOKEN=... python benchmark/prepare_benchmark.py --source hf
```

To cap input length while keeping random sampling:

```bash
python benchmark/prepare_benchmark.py --max-input-tokens 3000
```

## Upload to Hugging Face

`prepare_benchmark.py` also writes train/test JSONL files under
`benchmark/hf_splits/`. Upload them as a HF DatasetDict:

```bash
HF_TOKEN=... python benchmark/upload_to_hf.py \
  --repo-id boradorish/text-to-json-benchmark
```

Use `--private` if the dataset should not be public.

## 2. Run inference

`--model` may be either a local path or a Hugging Face model id. Local LoRA
adapter directories are detected automatically by the shared vLLM loader.

```bash
python benchmark/inference.py \
  --model saves/qwen3-0.6b/full/sft \
  --batch-size 4 \
  --output benchmark/runs/qwen3_0_6b_sft
```

Or run directly from the uploaded HF test split:

```bash
HF_TOKEN=... python benchmark/inference.py \
  --benchmark-source hf \
  --hf-dataset boradorish/text-to-json-benchmark \
  --hf-split test \
  --model saves/qwen3-0.6b/full/sft \
  --output benchmark/runs/qwen3_0_6b_sft
```

The progress bar is global over the full benchmark, independent of batch size.
The script writes both `.jsonl` and `.xlsx` outputs and resumes from an
existing JSONL file if present.

## 3. Evaluate

```bash
python benchmark/evaluate.py \
  --input benchmark/runs/qwen3_0_6b_sft.jsonl
```

Evaluation intentionally excludes LLM-as-a-judge semantic scoring. It reports
only deterministic metrics: parse/no-output, exact match, JSON Schema validity,
noise ratio, and rule-based leaf value match.

The output Excel has two sheets:

- `rows`: per-sample metrics, including `language_group`
- `language_summary`: metrics grouped by `ko`, `mixed`, `non_ko`, or `unknown`

Language grouping is heuristic and uses Hangul/Latin character ratios from
`user_prompt` by default. To classify with another field:

```bash
python benchmark/evaluate.py \
  --input benchmark/runs/qwen3_0_6b_sft.jsonl \
  --language-field raw_output
```
