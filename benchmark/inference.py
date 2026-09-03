"""
Run model inference on a downloaded benchmark JSONL file or a Hugging Face
Dataset split.

Works with either a local model path or a Hugging Face model id.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.vllm_inference import VllmModel, build_chat_prompts, generate_texts, load_vllm_model


SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")
EXCEL_TRUNCATE = 32000
DISPLAY_COLS = ("user_prompt", "raw_output")


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_model(model: str) -> str | Path:
    model_path = Path(model)
    if model_path.is_absolute() or model_path.exists():
        return model_path
    repo_local_path = PROJECT_ROOT / model
    return repo_local_path if repo_local_path.exists() else model


def extract_json_from_output(text: str) -> Any | None:
    parts = re.split(r"</think>", text, maxsplit=1)
    content = parts[-1].strip()
    fenced = re.search(r"```json\s*([\s\S]+?)\s*```", content)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    loose = re.search(r"(\{[\s\S]*\})", content)
    if loose:
        try:
            return json.loads(loose.group(1))
        except json.JSONDecodeError:
            pass
    return None


def load_benchmark(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hf_benchmark(repo_id: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install datasets first: pip install datasets") from exc

    token = os.environ.get("HF_TOKEN") or None
    dataset = load_dataset(repo_id, split=split, token=token)
    records = []
    for row in dataset:
        records.append(
            {
                "stem": row.get("stem") or row.get("id"),
                "user_prompt": row["user_prompt"],
                "gold_json": row["gold_json"] if "gold_json" in row else row.get("json", ""),
                "json_schema": row["json_schema"],
                "input_tokens": row.get("input_tokens"),
            }
        )
    return records


def save_records(records: list[dict], jsonl_path: Path, xlsx_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    display_records = []
    for record in records:
        row = dict(record)
        for col in DISPLAY_COLS:
            if isinstance(row.get(col), str):
                # JSONL keeps the exact model output. XLSX, however, uses XML
                # cells and rejects control bytes occasionally emitted by base
                # models, so sanitize only this human-readable copy.
                row[col] = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", row[col])
                if len(row[col]) > EXCEL_TRUNCATE:
                    row[col] = row[col][:EXCEL_TRUNCATE] + "..."
        display_records.append(row)
    pd.DataFrame(display_records).to_excel(xlsx_path, index=False)


def run_batch_inference(
    engine: VllmModel,
    rows: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    guided_json_backend: str | None = None,
    enable_thinking: bool | None = None,
) -> list[dict]:
    prompts = build_chat_prompts(
        engine.tokenizer, SYSTEM_PROMPT, [row["user_prompt"] for row in rows], enable_thinking=enable_thinking
    )
    if guided_json_backend:
        # Each STAGE-Eval example has its own schema. vLLM accepts a list of
        # per-request SamplingParams, allowing a batch to retain its individual
        # schema constraints.
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from vllm.v1.structured_output.backend_xgrammar import has_xgrammar_unsupported_json_features
        import xgrammar

        results: list[dict | None] = [None] * len(rows)
        active_prompts: list[str] = []
        active_params: list[SamplingParams] = []
        active_indices: list[int] = []
        for index, (prompt, row) in enumerate(zip(prompts, rows)):
            schema = json.loads(row["json_schema"])
            # vLLM performs this validation *after* handing the request to its
            # engine, which tears down the engine on an invalid schema.  Check
            # it locally first so unsupported examples can be recorded and the
            # rest of an 851-example run remains usable.
            if has_xgrammar_unsupported_json_features(schema):
                results[index] = (
                    {
                        "raw_output": "",
                        "json_obj": None,
                        "skip_reason": "xgrammar_schema_unsupported: vLLM feature precheck",
                    }
                )
                continue
            try:
                xgrammar.Grammar.from_json_schema(schema)
            except Exception as exc:
                results[index] = (
                    {
                        "raw_output": "",
                        "json_obj": None,
                        "skip_reason": f"xgrammar_schema_compile_failed: {exc}",
                    }
                )
                continue
            params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                guided_decoding=GuidedDecodingParams(json=schema),
            )
            active_prompts.append(prompt)
            active_params.append(params)
            active_indices.append(index)
        if active_prompts:
            outputs = engine.llm.generate(
                active_prompts,
                active_params,
                lora_request=engine.lora_request,
                use_tqdm=False,
            )
            for index, output in zip(active_indices, outputs):
                text = output.outputs[0].text
                results[index] = {"raw_output": text, "json_obj": extract_json_from_output(text)}
        # Every position is populated either by a local compatibility skip or
        # by vLLM generation, preserving row order for the caller.
        return [result for result in results if result is not None]

    outputs = generate_texts(
        engine,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_tqdm=False,
        seed=seed,
    )
    return [{"raw_output": text, "json_obj": extract_json_from_output(text)} for text in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark inference with vLLM.")
    parser.add_argument("--model", required=True, help="Local model path or HF model id")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--tokenizer-mode",
        choices=["auto", "slow"],
        default="auto",
        help="vLLM tokenizer implementation; use slow for affected Llama checkpoints.",
    )
    parser.add_argument("--benchmark-source", choices=["local", "hf"], default="local")
    parser.add_argument("--benchmark-file", default="benchmark/data/test.jsonl")
    parser.add_argument("--hf-dataset", default="boradorish/text-to-json-benchmark", help="HF dataset id when --benchmark-source hf")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--output", default="benchmark/runs/infer_results", help="Output path without suffix")
    parser.add_argument("--max-new-tokens", type=int, default=3100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guided-json-backend",
        choices=["xgrammar"],
        default=None,
        help="Constrain each example with its json_schema using the specified vLLM backend.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N benchmark rows.")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Pass enable_thinking=False to the chat template (Qwen3) so base models answer without a <think> block.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM/Torch compile paths; useful on servers without a C compiler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = resolve_path(args.benchmark_file)
    output_base = resolve_path(args.output)
    jsonl_path = output_base.with_suffix(".jsonl")
    xlsx_path = output_base.with_suffix(".xlsx")

    if args.benchmark_source == "hf":
        if not args.hf_dataset:
            print("[ERROR] --hf-dataset is required when --benchmark-source hf", file=sys.stderr)
            sys.exit(1)
        benchmark_rows = load_hf_benchmark(args.hf_dataset, args.hf_split)
    else:
        benchmark_rows = load_benchmark(benchmark_path)
    if args.limit is not None:
        benchmark_rows = benchmark_rows[: args.limit]
    # STAGE/CORD use ``stem`` while the prepared SGD turns use their stable
    # dialogue-turn ``id``.  Keep one resume key so both benchmark families
    # can use this runner without silently dropping SGD metadata.
    def record_key(row: dict) -> str:
        return str(row.get("stem", row.get("id")))

    saved_records: list[dict] = []
    done_stems: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    saved_records.append(record)
                    done_stems.add(record_key(record))
        print(f"[RESUME] existing records: {len(done_stems)}")

    rows = [row for row in benchmark_rows if record_key(row) not in done_stems]
    if not rows:
        if saved_records:
            save_records(saved_records, jsonl_path, xlsx_path)
            print(f"[RESUME] 결과 파일을 갱신했습니다: {xlsx_path}")
        print("[WARN] 처리할 benchmark row가 없습니다.")
        return

    engine = load_vllm_model(
        resolve_model(args.model),
        tokenizer_path=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        guided_decoding_backend=args.guided_json_backend,
        tokenizer_mode=args.tokenizer_mode,
    )

    final_total = len(saved_records) + len(rows)
    progress = tqdm(total=final_total, initial=len(saved_records), desc="benchmark infer", unit="sample") if tqdm else None
    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            results = run_batch_inference(
                engine,
                batch_rows,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.seed,
                args.guided_json_backend,
                enable_thinking=False if args.no_thinking else None,
            )
            for row, result in zip(batch_rows, results):
                saved_records.append(
                    {
                        "stem": record_key(row),
                        **({"id": row["id"]} if "id" in row else {}),
                        **({"service": row["service"]} if "service" in row else {}),
                        **({"seen_service": row["seen_service"]} if "seen_service" in row else {}),
                        "user_prompt": row["user_prompt"],
                        "gold_json": row["gold_json"],
                        "json_schema": row["json_schema"],
                        "input_tokens": row.get("input_tokens"),
                        "raw_output": result["raw_output"],
                        "pred_json": (
                            json.dumps(result["json_obj"], ensure_ascii=False)
                            if result["json_obj"] is not None
                            else ""
                        ),
                        **({"skip_reason": result["skip_reason"]} if "skip_reason" in result else {}),
                    }
                )
            save_records(saved_records, jsonl_path, xlsx_path)
            parsed = sum(1 for record in saved_records if record.get("pred_json"))
            if progress:
                progress.update(len(batch_rows))
                progress.set_postfix(
                    saved=len(saved_records),
                    remaining=final_total - len(saved_records),
                    parsed=parsed,
                )
            else:
                print(f"[{len(saved_records)}/{final_total}] remaining={final_total - len(saved_records)} parsed={parsed}")
    finally:
        if progress:
            progress.close()

    print(f"done: {len(saved_records)}")
    print(f"jsonl: {jsonl_path}")
    print(f"xlsx:  {xlsx_path}")


if __name__ == "__main__":
    main()
