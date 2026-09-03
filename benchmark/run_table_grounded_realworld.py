"""Train table-grounded STAGE SFT from Qwen base and evaluate two OOD sets.

The base and adapter arms share the exact benchmark files, decoding settings,
and seed.  The adapter is trained from raw Qwen3-4B; it is not a continuation
of the existing STAGE checkpoint.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/mnt/ddn/prod-runs/interns/sunghee/models/Qwen3-4B")
DOCUBENCH_ROOT = Path("/mnt/ddn/prod-runs/interns/sunghee/datasets/DocuBench")


def run(command: list[str], gpu: int) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)})


def line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8")) if path.exists() else 0


def infer(model: Path, benchmark: Path, output: Path, gpu: int, max_model_len: int, utilization: float, expected: int) -> Path:
    jsonl = output.with_suffix(".jsonl")
    if line_count(jsonl) != expected:
        run([
            sys.executable, "benchmark/inference.py", "--model", str(model),
            "--benchmark-file", str(benchmark), "--output", str(output),
            "--batch-size", "4", "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(utilization), "--seed", "42", "--no-thinking",
        ], gpu)
    return jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/sft/stage_table_grounded_all.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/table_grounded_realworld")
    parser.add_argument("--docubench", type=Path, default=ROOT / "benchmark/data/realworld/docubench_nonreceipt.jsonl")
    parser.add_argument("--kleister", type=Path, default=ROOT / "benchmark/data/realworld/kleister_nda_dev-0.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    args = parser.parse_args()
    required = [BASE, args.train_data, args.docubench, args.kleister, DOCUBENCH_ROOT]
    if any(not path.exists() for path in required):
        raise SystemExit("Missing base model, training data, benchmark, or DocuBench checkout.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = args.output_dir / "qwen3_4b_table_grounded_adapter"
    if not (adapter / "adapter_config.json").exists():
        run([
            sys.executable, "benchmark/train_toolfew_lora.py", "--model", str(BASE),
            "--train-data", str(args.train_data), "--output", str(adapter),
            "--epochs", str(args.epochs), "--batch-size", "1", "--grad-accum", "16",
            "--learning-rate", "2e-5", "--max-length", str(args.max_length), "--seed", "42",
            "--gradient-checkpointing",
        ], args.gpu)
    benchmarks = [("docubench", args.docubench, line_count(args.docubench)), ("kleister", args.kleister, line_count(args.kleister))]
    for dataset, benchmark, expected in benchmarks:
        for arm, model in (("base", BASE), ("stage_table_grounded", adapter)):
            result = infer(model, benchmark, args.output_dir / f"{dataset}_{arm}", args.gpu, args.max_length, args.gpu_memory_utilization, expected)
            if dataset == "docubench":
                run([sys.executable, "benchmark/evaluate_docubench.py", "--input", str(result), "--docubench-root", str(DOCUBENCH_ROOT)], args.gpu)
            else:
                run([sys.executable, "benchmark/evaluate_kleister_nda.py", "--input", str(result)], args.gpu)


if __name__ == "__main__":
    main()
