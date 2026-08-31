"""
Run benchmark inference for the three merged Qwen3-4B LoRA baseline models.

Run benchmark/merge_lora_adapters.py first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIRS = [
    "merged_models/qwen3-4b/qwen3-4b-jsonschemabench",
    "merged_models/qwen3-4b/qwen3-4b-glaive",
    "merged_models/qwen3-4b/qwen3-4b-scrapegraph",
]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark inference for merged LoRA baseline models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_DIRS)
    parser.add_argument("--hf-dataset", default="boradorish/text-to-json-benchmark")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--output-dir", default="benchmark/runs/lora_baselines")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3100)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model_path = resolve_path(model)
        run_name = model_path.name
        output_base = output_dir / run_name
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "benchmark" / "inference.py"),
            "--benchmark-source",
            "hf",
            "--hf-dataset",
            args.hf_dataset,
            "--hf-split",
            args.hf_split,
            "--model",
            str(model_path),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--output",
            str(output_base),
        ]
        print("=" * 80)
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
