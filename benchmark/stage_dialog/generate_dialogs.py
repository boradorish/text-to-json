"""STAGE-Dialog: source-grounded dialogue-state data from spreadsheet rows.

Follows the STAGE recipe for a new artifact type:
  1. source = one spreadsheet row (column -> cell) extracted from STAGE reports;
  2. an LLM writes a task-oriented USER/SYSTEM dialogue in which the user states
     a chosen subset of the cells verbatim, spread over several turns, and never
     states the remaining cells;
  3. every generated dialogue is validated against the source: each mentioned
     value must appear verbatim in a USER turn, no unmentioned value may appear
     anywhere, turns must alternate. Failures are dropped, not repaired.
The surviving dialogues yield per-turn dialogue-state examples in the same two
prompt formats used for SGD evaluation (standard: only specified slots;
explicit: every slot, "no output" when unspecified).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GEN_SYSTEM = (
    "You write realistic task-oriented dialogues between a USER and an assistant SYSTEM. "
    "You are given a spreadsheet record (column: value) and a list of columns the user MUST state. "
    "Write the dialogue as alternating lines that start with 'USER:' or 'SYSTEM:', beginning with USER, "
    "8 to 14 lines total. Rules:\n"
    "1. The user states the value of every MUST-STATE column exactly as written (character for character), "
    "and spreads them over at least three different USER lines rather than listing them all at once.\n"
    "2. Neither speaker ever mentions, paraphrases, or hints at the values of the OTHER columns. "
    "The SYSTEM may ask about them, but the USER defers (e.g. 'not sure yet', 'I'll decide later').\n"
    "3. The dialogue should sound like the user is requesting or arranging something related to the record's topic.\n"
    "4. Output only the dialogue lines. No headings, no JSON, no commentary."
)

PROMPT_PREFIX = "Extract the current dialogue state according to the JSON Schema.\n\n"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def build_gen_prompt(record: dict[str, str], mention: list[str], topic_hint: str) -> str:
    lines = ["Spreadsheet record:"]
    lines += [f"- {k}: {v}" for k, v in record.items()]
    lines.append("")
    lines.append("MUST-STATE columns (user says these values verbatim): " + ", ".join(mention))
    other = [k for k in record if k not in mention]
    lines.append("OTHER columns (never state their values): " + (", ".join(other) if other else "(none)"))
    lines.append(f"Topic hint: {topic_hint}")
    lines.append("")
    lines.append("Write the dialogue now.")
    return "\n".join(lines)


def parse_dialogue(text: str) -> list[tuple[str, str]] | None:
    turns = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(USER|SYSTEM)\s*:\s*(.+)$", line, flags=re.I)
        if not m:
            return None
        turns.append((m.group(1).upper(), m.group(2).strip()))
    if len(turns) < 6 or turns[0][0] != "USER":
        return None
    for a, b in zip(turns, turns[1:]):
        if a[0] == b[0]:
            return None
    return turns


def validate(turns: list[tuple[str, str]], record: dict[str, str], mention: list[str]) -> dict | None:
    """Return first-mention user-turn index per mentioned column, or None if invalid."""
    user_idx = [i for i, (spk, _) in enumerate(turns) if spk == "USER"]
    full = norm(" \n ".join(t for _, t in turns))
    first = {}
    for col in mention:
        v = norm(record[col])
        hit = next((i for i in user_idx if v in norm(turns[i][1])), None)
        if hit is None:
            return None
        first[col] = hit
    for col, val in record.items():
        if col in mention:
            continue
        v = norm(val)
        if len(v) >= 2 and re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", full):
            return None
    if len(set(first.values())) < 2:
        return None
    return first


def schema_for(record: dict[str, str], explicit: bool) -> dict:
    props = {}
    for col in record:
        spec = {"type": "string", "description": f"Value of '{col}' the user has specified."}
        if explicit:
            spec["description"] += ' Use "no output" if the user has not specified it.'
        props[col] = spec
    schema = {
        "type": "object",
        "description": (
            'Fill every slot. Write "no output" for any slot the user has not specified so far.'
            if explicit else "Include only slots the user has specified so far."
        ),
        "properties": props,
        "additionalProperties": False,
    }
    if explicit:
        schema["required"] = list(record)
    return schema


def make_examples(turns, record, mention, first, states_per_dialog, rng) -> list[dict]:
    user_idx = [i for i, (spk, _) in enumerate(turns) if spk == "USER"]
    # candidate cut points: every user turn; sample a few, always include the last
    cuts = sorted(set(rng.sample(user_idx, min(states_per_dialog - 1, len(user_idx))) + [user_idx[-1]]))
    out = []
    for cut in cuts:
        history = "\n".join(f"{spk}: {txt}" for spk, txt in turns[: cut + 1])
        state = {c: record[c] for c in mention if first[c] <= cut}
        for fmt in ("standard", "explicit"):
            explicit = fmt == "explicit"
            schema = schema_for(record, explicit)
            gold = {c: state.get(c, "no output") for c in record} if explicit else dict(state)
            prompt = f"{PROMPT_PREFIX}=== Report ===\n{history}\n\n=== JSON Schema ===\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            out.append({"format": fmt, "cut": cut, "user_prompt": prompt, "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema, ensure_ascii=False)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "benchmark" / "data" / "stage_dialog_records.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "stage_dialog")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples-per-record", type=int, default=1)
    ap.add_argument("--states-per-dialog", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from utils.vllm_inference import build_chat_prompts, generate_texts, load_vllm_model

    records = [json.loads(l) for l in a.records.open(encoding="utf-8") if l.strip()]
    if a.limit:
        records = records[: a.limit]
    jobs = []
    for r in records:
        rec = r["record"]; cols = list(rec)
        for _ in range(a.samples_per_record):
            k = max(2, min(len(cols) - 1, round(len(cols) * rng.uniform(0.5, 0.8))))
            mention = rng.sample(cols, k)
            topic = r.get("stem", "")
            jobs.append({"stem": r["stem"], "record": rec, "mention": mention, "gen_prompt": build_gen_prompt(rec, mention, f"columns: {', '.join(cols)}")})

    engine = load_vllm_model(a.model, gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=4096)
    prompts = build_chat_prompts(engine.tokenizer, GEN_SYSTEM, [j["gen_prompt"] for j in jobs])
    texts = generate_texts(engine, prompts, max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_p=0.95, seed=a.seed, use_tqdm=True)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"jobs": len(jobs), "parsed": 0, "valid": 0, "examples": 0}
    raw_path = a.output_dir / "dialogs_raw.jsonl"; ex_path = a.output_dir / "stage_dialog_examples.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw, ex_path.open("w", encoding="utf-8") as exf:
        for j, text in zip(jobs, texts):
            turns = parse_dialogue(text)
            first = validate(turns, j["record"], j["mention"]) if turns else None
            stats["parsed"] += bool(turns); stats["valid"] += bool(first)
            raw.write(json.dumps({**j, "text": text, "valid": bool(first)}, ensure_ascii=False) + "\n")
            if not first:
                continue
            for i, ex in enumerate(make_examples(turns, j["record"], j["mention"], first, a.states_per_dialog, rng)):
                ex["stem"] = f'{j["stem"]}_{i}'; stats["examples"] += 1
                exf.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (a.output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))
    print("saved", ex_path)


if __name__ == "__main__":
    main()
