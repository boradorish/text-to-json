"""Run the remaining CORD-v2 zero-shot base/SFT comparisons on two GPUs.

The Qwen3 pair is launched separately as part of the main experiment queue.
This runner waits for that pair, then evaluates the remaining checkpoints in
parallel, preserving the benchmark's standard vLLM decoding configuration.
It is resumable: an output with all 100 records is evaluated if needed and is
not generated again.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark/data/cord_v2_test_100.jsonl"
OUTPUT = ROOT / "outputs/cord_v2"
INITIAL = ("qwen3_4b_base", "qwen3_4b_sft")
PAIRS = (
    (
        ("qwen2_5_3b_base", "/mnt/nvme/cache/interns/hf/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", 0),
        ("qwen2_5_3b_sft", "/mnt/nvme/cache/interns/hf/hub/models--boradorish--qwen2.5-3b-sft/snapshots/3de5c63967fda091ec22270b104078eba2babae5", 1),
    ),
    (
        ("llama3_2_1b_base", "/mnt/nvme/cache/interns/hf/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08", 0),
        ("llama3_2_1b_sft", "/mnt/nvme/cache/interns/hf/hub/models--boradorish--llama3-1B-sft/snapshots/2ff0a67f63657348f2940383d800123008ce6c76", 1),
    ),
    (
        ("llama3_2_3b_base", "/mnt/nvme/cache/interns/hf/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062", 0),
        ("llama3_2_3b_sft", "/mnt/nvme/cache/interns/hf/hub/models--boradorish--llama3-3B-sft/snapshots/3047999726f486321d16b405ac2e06fa8b596248", 1),
    ),
)


def complete(name: str) -> bool:
    path = OUTPUT / f"{name}.jsonl"
    return path.exists() and sum(1 for line in path.open(encoding="utf-8") if line.strip()) == 100


def command(name: str, model: str) -> list[str]:
    return [
        sys.executable, "benchmark/inference.py", "--model", model,
        "--benchmark-file", str(DATA), "--output", str(OUTPUT / name),
        "--batch-size", "4", "--max-new-tokens", "3100", "--max-model-len", "8192",
        "--temperature", "0.6", "--top-p", "1.0", "--seed", "42", "--gpu-memory-utilization", "0.9",
    ]


def evaluate(name: str) -> None:
    subprocess.run(
        [sys.executable, "benchmark/evaluate.py", "--input", str(OUTPUT / f"{name}.jsonl"),
         "--output", str(OUTPUT / f"{name}_eval.xlsx")],
        cwd=ROOT,
        check=True,
    )


def run_one(spec: tuple[str, str, int]) -> None:
    name, model, gpu = spec
    if not complete(name):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
        subprocess.run(command(name, model), cwd=ROOT, env=env, check=True)
    evaluate(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    while not all(complete(name) for name in INITIAL):
        print("Waiting for initial Qwen3 CORD pair...", flush=True)
        time.sleep(args.poll_seconds)
    for pair in PAIRS:
        workers = [subprocess.Popen([sys.executable, __file__, "--run-one", *spec[:2], "--gpu", str(spec[2])], cwd=ROOT) for spec in pair]
        failures = [worker.wait() for worker in workers]
        if any(failures):
            raise SystemExit(f"CORD pair failed: {failures}")
    print("All four CORD model pairs completed.")


if __name__ == "__main__":
    # Internal worker interface keeps paired GPU runs isolated but lets the
    # parent process report a single resumable suite status.
    if "--run-one" in sys.argv:
        pos = sys.argv.index("--run-one")
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--run-one", nargs=2, metavar=("NAME", "MODEL"), required=True)
        worker_parser.add_argument("--gpu", type=int, required=True)
        worker_args = worker_parser.parse_args()
        run_one((worker_args.run_one[0], worker_args.run_one[1], worker_args.gpu))
    else:
        main()
