"""
Download HF LoRA adapters and merge them into full Hugging Face model folders.

Default targets:
  - boradorish/baseline-qwen3-4b-jsonschemabench
  - boradorish/baseline-qwen3-4b-glaive
  - boradorish/baseline-qwen3-4b-scrapegraph

Example:
    python benchmark/merge_lora_adapters.py \
      --output-dir merged_models/qwen3-4b
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch


DEFAULT_ADAPTERS = [
    "boradorish/baseline-qwen3-4b-jsonschemabench",
    "boradorish/baseline-qwen3-4b-glaive",
    "boradorish/baseline-qwen3-4b-scrapegraph",
]
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge HF LoRA adapters into full model directories.")
    parser.add_argument("--adapters", nargs="+", default=DEFAULT_ADAPTERS)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="merged_models/qwen3-4b")
    parser.add_argument("--device-map", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def slug_repo_id(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    name = re.sub(r"^baseline-", "", name)
    name = re.sub(r"[^0-9A-Za-z_.-]+", "-", name)
    return name.strip("-")


def torch_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def device_map_arg(name: str):
    if name == "auto":
        return "auto"
    if name == "cuda":
        return {"": "cuda:0"}
    return {"": "cpu"}


def read_adapter_base(adapter_id: str, fallback: str, trust_remote_code: bool) -> str:
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(adapter_id, "adapter_config.json")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return config.get("base_model_name_or_path") or fallback
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] adapter_config base lookup failed for {adapter_id}: {exc}")
        return fallback


def merge_one(
    *,
    adapter_id: str,
    base_model: str,
    output_path: Path,
    dtype_name: str,
    device_map_name: str,
    trust_remote_code: bool,
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actual_base_model = read_adapter_base(adapter_id, base_model, trust_remote_code)
    print("=" * 80)
    print(f"adapter: {adapter_id}")
    print(f"base:    {actual_base_model}")
    print(f"output:  {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        actual_base_model,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        actual_base_model,
        torch_dtype=torch_dtype(dtype_name),
        device_map=device_map_arg(device_map_name),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_id,
        torch_dtype=torch_dtype(dtype_name),
        is_trainable=False,
    )
    print("merging LoRA weights...")
    merged = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)
    print(f"saved merged model: {output_path}")

    del merged
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for adapter_id in args.adapters:
        output_path = output_root / slug_repo_id(adapter_id)
        if args.skip_existing and (output_path / "config.json").exists():
            print(f"[SKIP] merged model already exists: {output_path}")
            continue
        merge_one(
            adapter_id=adapter_id,
            base_model=args.base_model,
            output_path=output_path,
            dtype_name=args.dtype,
            device_map_name=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    print("\nMerged model directories:")
    for adapter_id in args.adapters:
        print(output_root / slug_repo_id(adapter_id))


if __name__ == "__main__":
    main()
