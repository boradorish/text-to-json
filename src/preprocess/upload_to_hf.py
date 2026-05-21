from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from .common import resolve_path, stem_to_related_paths
except ImportError:  # pragma: no cover - supports `python src/preprocess/upload_to_hf.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocess.common import resolve_path, stem_to_related_paths

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def load_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def dataset_stems(data_dir: str | Path) -> list[str]:
    user_prompt_dir = resolve_path(data_dir) / "user_prompt"
    json_dir = resolve_path(data_dir) / "json"

    if user_prompt_dir.is_dir():
        stems = sorted(path.stem for path in user_prompt_dir.glob("*.txt") if path.is_file())
    elif json_dir.is_dir():
        stems = sorted(path.stem for path in json_dir.glob("*.json") if path.is_file())
    else:
        raise NotADirectoryError(f"Expected data/user_prompt or data/json under: {resolve_path(data_dir)}")

    return stems


def build_records(data_dir: str | Path, *, require_json: bool = True) -> list[dict]:
    stems = dataset_stems(data_dir)
    iterator = stems
    if tqdm is not None:
        iterator = tqdm(stems, desc="build HF records", unit="sample")

    records: list[dict] = []
    skipped_missing_json = 0

    for stem in iterator:
        paths = stem_to_related_paths(stem, data_dir=data_dir, include_missing=True)
        if require_json and not paths["json"].is_file():
            skipped_missing_json += 1
            continue

        records.append(
            {
                "id": stem,
                "user_prompt": load_text(paths["user_prompt"]),
                "json": load_text(paths["json"]),
                "json_schema": load_text(paths["json_schema"]),
                "report": load_text(paths["report"]),
                "user_prompt_question": load_text(paths["user_prompt_question"]),
            }
        )

    if skipped_missing_json:
        print(f"skipped missing json: {skipped_missing_json:,}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload preprocessed data/* files to a Hugging Face Dataset repo.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--repo-id", default="boradorish/text-to-json-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="HF token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--revision", default=None, help="Optional branch/revision to push to.")
    parser.add_argument("--commit-message", default="Upload preprocessed text-to-json data")
    parser.add_argument("--allow-missing-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Paths")
    print(f"  data_dir:  {resolve_path(args.data_dir)}")
    print(f"  repo_id:   {args.repo_id}")
    print(f"  split:     {args.split}")

    records = build_records(args.data_dir, require_json=not args.allow_missing_json)
    print(f"records: {len(records):,}")
    if records:
        print(f"first id: {records[0]['id']}")
        print(f"last id:  {records[-1]['id']}")

    if args.dry_run:
        print("[dry-run] upload skipped")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN is not set. Export HF_TOKEN or pass --token.", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install datasets first: pip install datasets") from exc

    dataset = Dataset.from_list(records)
    dataset_dict = DatasetDict({args.split: dataset})

    push_kwargs = {
        "repo_id": args.repo_id,
        "token": token,
        "private": args.private,
        "commit_message": args.commit_message,
    }
    if args.revision:
        push_kwargs["revision"] = args.revision

    print(f"uploading to https://huggingface.co/datasets/{args.repo_id}")
    dataset_dict.push_to_hub(**push_kwargs)
    print("done")


if __name__ == "__main__":
    main()

