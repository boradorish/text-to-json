"""Upload the verified table-grounded STAGE SFT corpus to Hugging Face.

Pass a token through ``HF_TOKEN``; this script does not persist credentials.
The default is private so source-data licensing can be reviewed before release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="boradorish/STAGE-Table-Grounded-SFT")
    parser.add_argument("--data", type=Path, default=ROOT / "data/sft/stage_table_grounded_all.jsonl")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN for this process; do not store it in the repository.")
    metadata = args.data.with_suffix(".metadata.json")
    if not args.data.is_file() or not metadata.is_file():
        raise SystemExit("Expected the generated JSONL and its metadata JSON beside it.")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    readme = f"""---
language:
- en
- ko
tags:
- information-extraction
- json-schema
- table-understanding
size_categories:
- 1K<n<10K
---

# STAGE Table-Grounded SFT

{info['examples']:,} SFT examples built from {info['sources']:,} existing STAGE training reports.

## Construction

- The source is restricted to Markdown tables directly under `## Sheet:` headings.
- Markdown, TSV, and HTML are lossless renderings of those parsed source tables.
- The source is never rendered from the answer JSON.
- An example is retained only when every primitive gold value occurs literally in its source tables (`table_value_coverage = 1.0`).
- Full-schema tasks teach ordinary extraction. Requested-field-subset tasks retain the same source but request only a deterministic subset of top-level schema fields, to discourage filling unrequested fields.

The corpus is private until its original source-data licensing is reviewed.
"""
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(args.data), path_in_repo="data/train.jsonl", repo_id=args.repo_id, repo_type="dataset", commit_message="Upload table-grounded STAGE SFT data")
    api.upload_file(path_or_fileobj=str(metadata), path_in_repo="data/metadata.json", repo_id=args.repo_id, repo_type="dataset", commit_message="Upload construction metadata")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo_id, repo_type="dataset", commit_message="Document table-grounded construction")
    print(f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
