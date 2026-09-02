"""Make a shared, context-safe ExtractBench subset for local model evaluation.

ExtractBench contains documents ranging from short forms to full books.  This
project's registered protocol fixes ``max_model_len=8192`` and
``max_new_tokens=3100``.  Rather than silently truncating documents, retain
only examples that fit *every* evaluated tokenizer with the complete prompt
and reserve the configured generation budget.  All other digital-text rows
are written to a manifest with their per-model prompt lengths and reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def prompt_text(tokenizer, user_prompt: str) -> str:
    # This exactly mirrors src.utils.vllm_inference.build_chat_prompts(),
    # including its fallback for Meta base checkpoints without a template.
    tokenizer_id = str(getattr(tokenizer, "name_or_path", ""))
    if getattr(tokenizer, "chat_template", None) is None and "Llama" in tokenizer_id:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def token_length(tokenizer, user_prompt: str) -> int:
    return len(tokenizer(prompt_text(tokenizer, user_prompt), add_special_tokens=False).input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter ExtractBench without document truncation.")
    parser.add_argument("--input", default="benchmark/data/extractbench_digital.jsonl")
    parser.add_argument("--output", default="benchmark/data/extractbench_context8192.jsonl")
    parser.add_argument("--skipped-output", default="outputs/extractbench/context_skipped.jsonl")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=3100)
    parser.add_argument("--qwen-tokenizer", default="../models/Qwen3-4B")
    parser.add_argument(
        "--llama-tokenizer",
        default="/mnt/nvme/cache/interns/hf/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08",
    )
    args = parser.parse_args()
    if args.max_model_len <= args.max_new_tokens:
        raise SystemExit("max-model-len must exceed max-new-tokens")

    tokenizers = {
        "qwen": AutoTokenizer.from_pretrained(resolve(args.qwen_tokenizer)),
        "llama": AutoTokenizer.from_pretrained(resolve(args.llama_tokenizer)),
    }
    input_path, output_path, skipped_path = map(resolve, (args.input, args.output, args.skipped_output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    limit = args.max_model_len - args.max_new_tokens
    kept = skipped = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as out, skipped_path.open("w", encoding="utf-8") as skipped_out:
        for line in source:
            row = json.loads(line)
            lengths = {name: token_length(tokenizer, row["user_prompt"]) for name, tokenizer in tokenizers.items()}
            if max(lengths.values()) <= limit:
                row["context_token_lengths"] = lengths
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped_out.write(json.dumps({
                    "stem": row["stem"],
                    "source_split": row["source_split"],
                    "prompt_token_lengths": lengths,
                    "skip_reason": f"context_budget_exceeded: prompt must be <= {limit} tokens to reserve {args.max_new_tokens} generation tokens within max_model_len={args.max_model_len}",
                }, ensure_ascii=False) + "\n")
                skipped += 1
    print(f"kept={kept}; context-skipped={skipped}; budget={limit}")
    print(f"data: {output_path}")
    print(f"manifest: {skipped_path}")


if __name__ == "__main__":
    main()
