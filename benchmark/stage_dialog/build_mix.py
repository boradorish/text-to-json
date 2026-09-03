"""Build the continued-SFT mix: original STAGE examples + STAGE-Dialog examples.

Output rows use the ``messages`` format consumed by ``train_toolfew_lora.py``:
system = STAGE inference system prompt, user = benchmark-style prompt,
assistant = gold JSON (pretty-printed, indent 2, like the STAGE SFT targets).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = (ROOT / "prompt" / "infer_SYSTEM_prompt.txt").read_text(encoding="utf-8")


def pretty(gold: str) -> str:
    try:
        return json.dumps(json.loads(gold), ensure_ascii=False, indent=2)
    except Exception:
        return gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-parquet", type=Path, required=True)
    ap.add_argument("--dialog", type=Path, required=True, help="stage_dialog_examples.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-stage", type=int, default=3000)
    ap.add_argument("--n-dialog", type=int, default=4000)
    ap.add_argument("--max-states-per-dialog", type=int, default=2)
    ap.add_argument("--tokenizer", default=None, help="If set, drop rows longer than --max-tokens")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import pyarrow.parquet as pq
    stage = pq.read_table(a.stage_parquet).to_pylist()
    rng.shuffle(stage)

    dialog = [json.loads(l) for l in a.dialog.open(encoding="utf-8") if l.strip()]
    # limit states per dialogue (stem prefix before the trailing _i index groups a dialogue)
    by_dialog: dict[str, list[dict]] = {}
    for r in dialog:
        by_dialog.setdefault(r["stem"].rsplit("_", 1)[0], []).append(r)
    picked = []
    for stem, rows in by_dialog.items():
        cuts = sorted({r["cut"] for r in rows})
        keep_cuts = set(rng.sample(cuts, min(a.max_states_per_dialog, len(cuts))))
        picked += [r for r in rows if r["cut"] in keep_cuts]
    rng.shuffle(picked)

    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    def row(user_prompt: str, gold: str, source: str, fmt: str = "") -> dict | None:
        assistant = pretty(gold)
        if tok is not None and len(tok(user_prompt + assistant)["input_ids"]) > a.max_tokens:
            return None
        return {"source": source, "format": fmt, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ]}

    out, stats = [], Counter()
    for r in stage:
        if stats["stage"] >= a.n_stage:
            break
        x = row(r["user_prompt"], r["gold_json"], "stage")
        if x:
            out.append(x); stats["stage"] += 1
        else:
            stats["stage_dropped_long"] += 1
    for r in picked:
        if stats["dialog"] >= a.n_dialog:
            break
        x = row(r["user_prompt"], r["gold_json"], "dialog", r["format"])
        if x:
            out.append(x); stats["dialog"] += 1; stats[f"dialog_{r['format']}"] += 1
    rng.shuffle(out)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats["total"] = len(out)
    print(json.dumps(dict(stats)))
    print("saved", a.output)


if __name__ == "__main__":
    main()
