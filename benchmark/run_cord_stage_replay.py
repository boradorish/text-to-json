"""Fair CORD continued-SFT comparison: identical data/settings for base and STAGE.

The only intentional difference between the two arms is their initial checkpoint.
This is deliberately sequential so it can run on one free GPU without competing
with unrelated jobs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "base": "/mnt/ddn/prod-runs/interns/sunghee/models/Qwen3-4B",
    "stage": "/mnt/ddn/prod-runs/interns/sunghee/models/STAGE-Qwen3-4B-SFT",
}


def run(command: list[str], gpu: int) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/sft/cord_stage_replay_50x4_s200.jsonl")
    parser.add_argument("--test-data", type=Path, default=ROOT / "benchmark/data/cord_v2_adaptation/test_100.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cord_stage_replay/50r4_s200")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.train_data.exists() or not args.test_data.exists():
        raise SystemExit("Missing fixed train or test file")

    for arm, model in MODELS.items():
        adapter = args.output_dir / f"{arm}_adapter"
        result = args.output_dir / f"{arm}_test"
        if not (adapter / "adapter_config.json").exists():
            run([
                sys.executable, "benchmark/train_toolfew_lora.py", "--model", model,
                "--train-data", str(args.train_data), "--output", str(adapter),
                "--epochs", str(args.epochs), "--batch-size", "1", "--grad-accum", "16",
                "--learning-rate", str(args.learning_rate), "--max-length", str(args.max_length),
                "--seed", str(args.seed), "--gradient-checkpointing",
            ], args.gpu)
        jsonl = result.with_suffix(".jsonl")
        if not jsonl.exists() or sum(bool(line.strip()) for line in jsonl.open(encoding="utf-8")) != 100:
            command = [
                sys.executable, "benchmark/inference.py", "--model", str(adapter),
                "--benchmark-file", str(args.test_data),
                "--output", str(result), "--batch-size", "8", "--max-model-len", str(args.max_length),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--seed", str(args.seed), "--no-thinking",
            ]
            run(command, args.gpu)
        run([sys.executable, "benchmark/evaluate.py", "--input", str(jsonl)], args.gpu)


if __name__ == "__main__":
    main()
