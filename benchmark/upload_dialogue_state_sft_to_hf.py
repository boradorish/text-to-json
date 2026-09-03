"""Upload the verified dialogue-state extension corpus without storing tokens."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="boradorish/STAGE-Dialogue-State-SFT")
    parser.add_argument("--data", type=Path, default=ROOT / "data/sft/source_grounded_dialogue_state_18096.jsonl")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN for this process; do not store it in the repository.")
    metadata = args.data.with_suffix(".metadata.json")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    readme = f"""---
language:
- en
tags:
- dialogue-state-tracking
- information-extraction
- json-schema
size_categories:
- 10K<n<100K
---

# Source-grounded dialogue-state SFT data

This is a dialogue-state-tracking extension of the STAGE data-generation pipeline, not an SGD-derived dataset.

## Construction

1. A flat source record is formed from a spreadsheet-table header and row.
2. An LLM writes USER/SYSTEM dialogue after a random 50–80% subset of record columns is designated as user-mentioned.
3. The dialogue is retained only when every designated value occurs verbatim in a USER turn, all remaining record values are absent from the entire dialogue, and turns alternate.
4. State targets are made at USER-turn cut points. The two formats are an open schema containing only mentioned slots and an explicit schema that uses `no output` for unmentioned slots.

Validation retained {info['validated_dialogues']:,}/{info['generation_jobs']:,} generated dialogues and produced {info['examples']:,} state examples ({info['formats']}). SGD data is never used in generation, filtering, or training data construction.
"""
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(args.data), path_in_repo="data/train.jsonl", repo_id=args.repo_id, repo_type="dataset", commit_message="Upload source-grounded dialogue-state SFT data")
    api.upload_file(path_or_fileobj=str(metadata), path_in_repo="data/metadata.json", repo_id=args.repo_id, repo_type="dataset", commit_message="Upload dialogue-state construction metadata")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo_id, repo_type="dataset", commit_message="Document dialogue-state construction")
    print(f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
