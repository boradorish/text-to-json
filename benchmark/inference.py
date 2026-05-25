"""
Run model inference on benchmark/benchmark_samples.jsonl or a Hugging Face
Dataset split.

Works with either a local model path or a Hugging Face model id. LoRA adapter
directories are handled by src/utils/vllm_inference.py.
"""
from __future__ import annotations

import argparse
import json
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
            if isinstance(row.get(col), str) and len(row[col]) > EXCEL_TRUNCATE:
                row[col] = row[col][:EXCEL_TRUNCATE] + "..."
        display_records.append(row)
    pd.DataFrame(display_records).to_excel(xlsx_path, index=False)


def run_batch_inference(engine: VllmModel, user_texts: list[str], max_new_tokens: int) -> list[dict]:
    prompts = build_chat_prompts(engine.tokenizer, SYSTEM_PROMPT, user_texts)
    outputs = generate_texts(
        engine,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        use_tqdm=False,
    )
    return [{"raw_output": text, "json_obj": extract_json_from_output(text)} for text in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark inference with vLLM.")
    parser.add_argument("--model", required=True, help="Local model path or HF model id")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--benchmark-source", choices=["local", "hf"], default="local")
    parser.add_argument("--benchmark-file", default="benchmark/benchmark_samples.jsonl")
    parser.add_argument("--hf-dataset", default=None, help="HF dataset id when --benchmark-source hf")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--output", default="benchmark/runs/infer_results", help="Output path without suffix")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
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
    saved_records: list[dict] = []
    done_stems: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    saved_records.append(record)
                    done_stems.add(str(record["stem"]))
        print(f"[RESUME] existing records: {len(done_stems)}")

    rows = [row for row in benchmark_rows if row["stem"] not in done_stems]
    if not rows:
        print("[WARN] 처리할 benchmark row가 없습니다.")
        return

    engine = load_vllm_model(
        resolve_model(args.model),
        tokenizer_path=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    final_total = len(saved_records) + len(rows)
    progress = tqdm(total=final_total, initial=len(saved_records), desc="benchmark infer", unit="sample") if tqdm else None
    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            results = run_batch_inference(
                engine,
                [row["user_prompt"] for row in batch_rows],
                args.max_new_tokens,
            )
            for row, result in zip(batch_rows, results):
                saved_records.append(
                    {
                        "stem": row["stem"],
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
