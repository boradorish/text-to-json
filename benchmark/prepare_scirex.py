"""Export SciREX (Jain et al., 2020) to STAGE JSONL: salient datasets / methods / metrics / tasks of a paper.

Each document is a full ML paper (median 5.8k words) with expert-annotated salient entity
clusters (``coref``: entity -> mention spans). The schema asks for four arrays (datasets,
methods, metrics, tasks); the gold lists one canonical surface form per salient entity (its
most frequent mention) and keeps every mention form in ``gold_alts`` for scoring. Also stores
all annotated (non-salient) mentions per type so the scorer can report lenient precision.
Input: release_data/{test,dev}.jsonl from https://github.com/allenai/SciREX.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPE_KEY = {"Material": "datasets", "Method": "methods", "Metric": "metrics", "Task": "tasks"}
DESC = {"datasets": "Datasets the paper trains or evaluates on, as named in the paper",
        "methods": "Methods, models or systems the paper proposes or compares against, as named in the paper",
        "metrics": "Evaluation metrics the paper reports, as named in the paper",
        "tasks": "Tasks or problems the paper addresses, as named in the paper"}
PROMPT_PREFIX = ("Extract the salient datasets, methods, metrics and tasks of this scientific paper according to the JSON Schema. "
                 "Use the names as they appear in the paper, one entry per distinct entity. Return exactly one JSON object.\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="directory with test.jsonl / dev.jsonl")
    ap.add_argument("--splits", default="test,dev")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "scirex_salient.jsonl")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    schema = {"type": "object", "additionalProperties": False, "required": list(DESC),
              "properties": {k: {"type": "array", "items": {"type": "string"}, "description": v} for k, v in DESC.items()}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    n, lengths, n_ent = 0, [], 0
    with a.output.open("w", encoding="utf-8") as fh:
        for split in a.splits.split(","):
            for line in (a.src / f"{split}.jsonl").open(encoding="utf-8"):
                d = json.loads(line)
                words = d["words"]
                span_type = {(s, e): t for s, e, t in d["ner"]}
                gold = {k: [] for k in DESC}; alts = {k: [] for k in DESC}
                for ent, spans in d["coref"].items():
                    if not spans:
                        continue
                    types = collections.Counter(span_type.get(tuple(sp), None) for sp in spans)
                    types.pop(None, None)
                    if not types:
                        continue
                    key = TYPE_KEY[types.most_common(1)[0][0]]
                    forms = collections.Counter(" ".join(words[sp[0]:sp[1]]) for sp in spans)
                    canon = forms.most_common(1)[0][0]
                    gold[key].append(canon); alts[key].append(sorted(forms)); n_ent += 1
                all_mentions = {k: sorted({" ".join(words[s:e]) for s, e, t in d["ner"] if TYPE_KEY[t] == k}) for k in DESC}
                parts = []
                for s, e in d["sections"]:
                    parts.append(" ".join(words[s:e]))
                text = "\n\n".join(parts) if parts else " ".join(words)
                prompt = f"{PROMPT_PREFIX}=== Report ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(schema, indent=2)}"
                rec = {"stem": f"scirex_{n:03d}", "dataset": "scirex_salient", "source_id": f"{split}:{d['doc_id']}", "user_prompt": prompt,
                       "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(schema), "gold_alts": json.dumps(alts, ensure_ascii=False),
                       "all_mentions": json.dumps(all_mentions, ensure_ascii=False), "n_words": len(words)}
                if tok is not None:
                    t = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = t; lengths.append(t)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} papers, {n_ent} salient entities to {a.output}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}")


if __name__ == "__main__":
    main()
