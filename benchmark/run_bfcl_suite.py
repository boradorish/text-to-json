"""Run Qwen3 base/SFT BFCL-v4 offline evaluation after the CORD suite."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORD = ROOT / "outputs/cord_v2"
CORD_RUNS = (
    "qwen3_4b_base", "qwen3_4b_sft", "qwen2_5_3b_base", "qwen2_5_3b_sft",
    "llama3_2_1b_base", "llama3_2_1b_sft", "llama3_2_3b_base", "llama3_2_3b_sft",
)
MODELS = (
    ("qwen3_4b_base", "/mnt/nvme/cache/interns/hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c", 0),
    # This cache snapshot contains weights but not config.json; use the HF id
    # so BFCL's local-server preflight resolves the complete model repository.
    ("qwen3_4b_sft", "boradorish/baseline-qwen3-4b-best", 1),
)


def complete(name: str) -> bool:
    path = CORD / f"{name}.jsonl"
    return path.exists() and sum(1 for line in path.open(encoding="utf-8") if line.strip()) == 100


def main() -> None:
    while not all(complete(name) for name in CORD_RUNS):
        print("Waiting for full CORD model suite...", flush=True)
        time.sleep(30)
    workers = []
    for run_name, path, gpu in MODELS:
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
        workers.append(subprocess.Popen(
            [sys.executable, "benchmark/run_bfcl_local.py", "--model-path", path, "--run-name", run_name],
            cwd=ROOT, env=env,
        ))
    failures = [worker.wait() for worker in workers]
    if any(failures):
        raise SystemExit(f"BFCL failed: {failures}")
    for run_name, _, _ in MODELS:
        subprocess.run([sys.executable, "benchmark/summarize_bfcl.py", "--result-root", f"outputs/bfcl/{run_name}"], cwd=ROOT, check=True)
    print("BFCL base/SFT offline suite completed.")


if __name__ == "__main__":
    main()
