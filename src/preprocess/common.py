from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RELATED_DIRS = {
    "json": "json",
    "json_schema": "json_schema",
    "report": "report",
    "user_prompt": "user_prompt",
    "user_prompt_question": "user_prompt_question",
}


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for path in [here, *here.parents]:
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_json_files(json_dir: str | Path) -> list[Path]:
    path = resolve_path(json_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"JSON directory not found: {path}")
    return sorted(p for p in path.glob("*.json") if p.is_file())


def stem_to_related_paths(
    stem: str,
    *,
    data_dir: str | Path = "data",
    include_missing: bool = False,
) -> dict[str, Path]:
    data_path = resolve_path(data_dir)
    candidates = {
        "json": data_path / DEFAULT_RELATED_DIRS["json"] / f"{stem}.json",
        "json_schema": data_path / DEFAULT_RELATED_DIRS["json_schema"] / f"{stem}.json",
        "report": data_path / DEFAULT_RELATED_DIRS["report"] / f"{stem}.txt",
        "user_prompt": data_path / DEFAULT_RELATED_DIRS["user_prompt"] / f"{stem}.txt",
        "user_prompt_question": data_path / DEFAULT_RELATED_DIRS["user_prompt_question"] / f"{stem}.txt",
    }
    if include_missing:
        return candidates
    return {key: path for key, path in candidates.items() if path.is_file()}


@dataclass(frozen=True)
class DeleteResult:
    deleted: list[Path]
    missing: list[Path]
    failed: list[tuple[Path, str]]


def delete_related_files(
    stems: Iterable[str],
    *,
    data_dir: str | Path = "data",
    dry_run: bool = True,
) -> DeleteResult:
    deleted: list[Path] = []
    missing: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for stem in sorted(set(stems)):
        paths = stem_to_related_paths(stem, data_dir=data_dir, include_missing=True)
        for path in paths.values():
            if not path.exists():
                missing.append(path)
                continue
            if dry_run:
                deleted.append(path)
                continue
            try:
                path.unlink()
                deleted.append(path)
            except OSError as exc:
                failed.append((path, str(exc)))

    return DeleteResult(deleted=deleted, missing=missing, failed=failed)


def flatten_json_leaves(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key, child in value.items():
            leaves.extend(flatten_json_leaves(child, f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        leaves = []
        for idx, child in enumerate(value):
            leaves.extend(flatten_json_leaves(child, f"{path}[{idx}]"))
        return leaves
    return [(path, value)]


def load_text_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")

