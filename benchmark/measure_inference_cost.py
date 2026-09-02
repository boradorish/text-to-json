"""Measure the inference-time cost of unconstrained vs xgrammar decoding.

Each schema is first checked with the same vLLM compatibility predicate used
by ``inference.py``.  The resulting shared subset is then used for every
condition, so a grammar coverage gap never silently changes a denominator.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd

from inference import SYSTEM_PROMPT, build_chat_prompts, load_benchmark, resolve_model, resolve_path
from utils.vllm_inference import load_vllm_model


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(pd.Series(values).quantile(p))


def stats(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "p90": percentile(values, 0.9), "mean": statistics.mean(values), "sum": sum(values)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="e.g. base_free or sft_xgrammar")
    parser.add_argument("--benchmark-file", default="benchmark/data/stage_eval_test.jsonl")
    parser.add_argument("--output-dir", default="outputs/inference_cost")
    parser.add_argument("--guided-json", action="store_true")
    parser.add_argument("--compile-only", action="store_true", help="Measure xgrammar compilation without loading vLLM/GPU.")
    parser.add_argument("--batch-size", type=int, choices=(1, 32), default=1)
    parser.add_argument("--pass-name", choices=("cold", "warm", "throughput"), default="cold")
    parser.add_argument("--second-pass", action="store_true", help="Measure a cached warm pass in the same engine.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Debug-only cap after compatibility filtering.")
    parser.add_argument("--max-new-tokens", type=int, default=3100)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def compatible_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    from vllm.v1.structured_output.backend_xgrammar import has_xgrammar_unsupported_json_features
    import xgrammar

    good, skipped = [], []
    for row in rows:
        schema = json.loads(row["json_schema"])
        if has_xgrammar_unsupported_json_features(schema):
            skipped.append({"stem": row["stem"], "skip_reason": "vllm feature precheck"})
            continue
        try:
            xgrammar.Grammar.from_json_schema(schema)
        except Exception as exc:
            skipped.append({"stem": row["stem"], "skip_reason": f"grammar compile: {exc}"})
            continue
        good.append(row)
    return good, skipped


def grammar_compile_times(rows: list[dict], tokenizer) -> list[float]:
    import xgrammar

    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo.from_huggingface(tokenizer), cache_enabled=False)
    elapsed = []
    for row in rows:
        start = time.perf_counter()
        compiler.compile_json_schema(json.loads(row["json_schema"]))
        elapsed.append(time.perf_counter() - start)
    return elapsed


def make_params(row: dict | None, args: argparse.Namespace):
    from vllm import SamplingParams

    kwargs = dict(max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    if row is None:
        return SamplingParams(**kwargs)
    from vllm.sampling_params import GuidedDecodingParams
    return SamplingParams(**kwargs, guided_decoding=GuidedDecodingParams(json=json.loads(row["json_schema"])))


def generate(engine, prompts: list[str], batch_rows: list[dict], args: argparse.Namespace) -> tuple[list, float]:
    params = [make_params(row if args.guided_json else None, args) for row in batch_rows]
    start = time.perf_counter()
    outputs = engine.llm.generate(prompts, params if args.guided_json else params[0], lora_request=engine.lora_request, use_tqdm=False)
    return outputs, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_benchmark(resolve_path(args.benchmark_file))
    compatible, skipped = compatible_rows(rows)
    if args.limit is not None:
        compatible = compatible[: args.limit]
    if not compatible:
        raise SystemExit("No xgrammar-compatible rows.")
    if args.compile_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(resolve_model(args.model), trust_remote_code=True)
        compile_seconds = grammar_compile_times(compatible, tokenizer)
        summary = {
            "label": args.label, "pass": "compile_only", "batch_size": 0,
            "samples": len(compatible), "xgrammar_skipped": len(skipped), "warmup_samples": 0,
            "latency_median_seconds": 0.0, "latency_p90_seconds": 0.0, "wall_seconds": sum(compile_seconds),
            "examples_per_second": 0.0, "generated_tokens": 0, "tokens_per_second": 0.0,
            "mean_generated_tokens": 0.0, "token_seconds": 0.0,
            "grammar_compile_median_seconds": stats(compile_seconds)["median"],
            "grammar_compile_p90_seconds": stats(compile_seconds)["p90"],
            "grammar_compile_total_seconds": sum(compile_seconds),
        }
        summary_path = out_dir / "summary.csv"
        existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        pd.concat([existing, pd.DataFrame([summary])], ignore_index=True).to_csv(summary_path, index=False)
        (out_dir / "xgrammar_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    # Free decoding still uses exactly this compatibility-filtered population.
    engine = load_vllm_model(
        resolve_model(args.model), gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, guided_decoding_backend="xgrammar" if args.guided_json else None,
    )
    prompts = build_chat_prompts(engine.tokenizer, SYSTEM_PROMPT, [row["user_prompt"] for row in compatible])
    compile_seconds = grammar_compile_times(compatible, engine.tokenizer) if args.guided_json else []
    # Warm-up never enters the reported per-example results.
    if args.warmup:
        generate(engine, prompts[: args.warmup], compatible[: args.warmup], args)

    summary_path = out_dir / "summary.csv"
    summaries = []
    for pass_name in [args.pass_name] + (["warm"] if args.second_pass else []):
        records = []
        whole_start = time.perf_counter()
        for start in range(0, len(compatible), args.batch_size):
            batch_rows = compatible[start : start + args.batch_size]
            batch_prompts = prompts[start : start + args.batch_size]
            outputs, batch_seconds = generate(engine, batch_prompts, batch_rows, args)
            for row, output in zip(batch_rows, outputs):
                records.append({"stem": row["stem"], "latency_seconds": batch_seconds / len(batch_rows), "batch_seconds": batch_seconds, "generated_tokens": len(output.outputs[0].token_ids)})
        wall_seconds = time.perf_counter() - whole_start
        record_path = out_dir / f"{args.label}_{pass_name}_b{args.batch_size}.jsonl"
        with record_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        latencies, token_total = [record["latency_seconds"] for record in records], sum(record["generated_tokens"] for record in records)
        summaries.append({"label": args.label, "pass": pass_name, "batch_size": args.batch_size, "samples": len(records), "xgrammar_skipped": len(skipped), "warmup_samples": args.warmup, "latency_median_seconds": stats(latencies)["median"], "latency_p90_seconds": stats(latencies)["p90"], "wall_seconds": wall_seconds, "examples_per_second": len(records) / wall_seconds, "generated_tokens": token_total, "tokens_per_second": token_total / wall_seconds, "mean_generated_tokens": token_total / len(records), "token_seconds": wall_seconds / token_total if token_total else float("nan"), "grammar_compile_median_seconds": stats(compile_seconds)["median"] if compile_seconds else 0.0, "grammar_compile_p90_seconds": stats(compile_seconds)["p90"] if compile_seconds else 0.0, "grammar_compile_total_seconds": sum(compile_seconds)})
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(summaries)], ignore_index=True).to_csv(summary_path, index=False)
    (out_dir / "xgrammar_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
