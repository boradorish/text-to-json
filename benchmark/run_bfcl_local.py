"""Run BFCL-v4 offline categories with a local Qwen-compatible checkpoint.

The upstream BFCL registry maps current Qwen3 entries to DashScope API
handlers.  This thin wrapper keeps BFCL's official prompts, response format,
and AST checker, but replaces that handler with its bundled local-vLLM Qwen
handler.  It deliberately limits evaluation to the requested offline
categories: simple_python, multiple, and parallel.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_CATEGORIES = ["simple_python", "multiple", "parallel"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFCL v4 offline categories through local vLLM.")
    parser.add_argument("--model-path", required=True, help="HF id or local complete model checkpoint.")
    parser.add_argument("--run-name", required=True, help="Output subdirectory name, e.g. qwen3_4b_base.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "bfcl" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    # BFCL resolves output paths at import time from this variable.
    os.environ["BFCL_PROJECT_ROOT"] = str(output_root)

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.eval_runner import main as evaluate_main
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler

    # Use a separate standard registry key only inside this process.  BFCL's
    # checker still receives the known Qwen3 naming conventions.
    config = MODEL_CONFIG_MAPPING["qwen3-4b"]
    config.model_handler = QwenHandler
    config.model_name = args.model_path
    config.is_fc_model = False

    generation_args = SimpleNamespace(
        model=["qwen3-4b"],
        test_category=OFFLINE_CATEGORIES,
        temperature=args.temperature,
        include_input_log=True,
        exclude_state_log=True,
        num_gpus=1,
        num_threads=args.num_threads,
        gpu_memory_utilization=args.gpu_memory_utilization,
        backend="vllm",
        skip_server_setup=False,
        local_model_path=args.model_path if Path(args.model_path).is_dir() else None,
        result_dir="result",
        allow_overwrite=False,
        run_ids=False,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )
    generation_main(generation_args)
    evaluate_main(
        ["qwen3-4b"],
        OFFLINE_CATEGORIES,
        result_dir="result",
        score_dir="score",
        partial_eval=False,
    )
    print(f"BFCL artifacts: {output_root}")


if __name__ == "__main__":
    main()
