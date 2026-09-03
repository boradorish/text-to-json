"""Resumable two-GPU runner for experiment 9's CORD layout ablation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen3_4b_base": "/mnt/ddn/prod-runs/interns/sunghee/models/Qwen3-4B",
    "qwen3_4b_sft": "/mnt/ddn/prod-runs/interns/sunghee/models/STAGE-Qwen3-4B-SFT",
    "qwen2_5_3b_base": "/mnt/nvme/cache/interns/hf/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "qwen2_5_3b_sft": "/mnt/nvme/cache/interns/hf/hub/models--boradorish--qwen2.5-3b-sft/snapshots/3de5c63967fda091ec22270b104078eba2babae5",
}
INPUTS = {
    "a": ROOT / "benchmark/data/cord_v2_layout/a_rows.jsonl",
    "ab": ROOT / "benchmark/data/cord_v2_layout/ab_rows_descriptions.jsonl",
    "abc": ROOT / "benchmark/data/cord_v2_layout/abc_rows_descriptions_oneshot.jsonl",
}


def complete(path: Path) -> bool:
    return path.exists() and sum(bool(line.strip()) for line in path.open(encoding="utf-8")) == 100


def run_one(variant: str, name: str, guided: bool, gpu: int) -> None:
    output = ROOT / "outputs/cord_v2_layout" / variant / f"{name}_{'xgrammar' if guided else 'free'}"
    if not complete(output.with_suffix(".jsonl")):
        cmd = [sys.executable, "benchmark/inference.py", "--model", MODELS[name], "--benchmark-file", str(INPUTS[variant]), "--output", str(output), "--batch-size", "8", "--max-model-len", "8192"]
        if guided:
            cmd.append("--guided-json-backend=xgrammar")
        if name == "qwen3_4b_base":
            cmd.append("--no-thinking")
        subprocess.run(cmd, cwd=ROOT, env=os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)
    subprocess.run([sys.executable, "benchmark/evaluate.py", "--input", str(output.with_suffix(".jsonl"))], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(INPUTS), required=True)
    parser.add_argument("--name", choices=tuple(MODELS), required=True)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    run_one(args.variant, args.name, args.guided, args.gpu)


if __name__ == "__main__":
    main()
