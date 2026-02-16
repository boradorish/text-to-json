from __future__ import annotations

from pathlib import Path
from typing import Iterable


def rename_xlsx_sequential(
    target_dir: str | Path,
    *,
    prefix: str = "data_",
    start: int = 1,
    recursive: bool = False,
    dry_run: bool = True,
    sort_by: str = "name",  # "name" | "mtime"
) -> list[tuple[Path, Path]]:
    """
    Rename .xlsx files in target_dir to prefix{n}.xlsx (e.g., data_1.xlsx, data_2.xlsx...).

    Safety:
      - Uses a 2-step rename via temporary filenames to avoid collisions.
      - Returns list of (old_path, new_path).

    Args:
      target_dir: Directory containing xlsx files.
      prefix: New filename prefix.
      start: Starting index.
      recursive: If True, includes subdirectories.
      dry_run: If True, print plan only and do not rename.
      sort_by: "name" (alphabetical) or "mtime" (modified time ascending).

    Returns:
      List of (old_path, new_path) renames.
    """
    d = Path(target_dir).resolve()
    if not d.exists() or not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    files = [p for p in d.glob(pattern) if p.is_file()]

    # Exclude temporary/lock files if any (optional)
    files = [p for p in files if not p.name.startswith("~$")]

    if sort_by == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name.lower())

    # Plan new names (in the SAME directory as each file)
    plan: list[tuple[Path, Path]] = []
    n = start
    for p in files:
        new_name = f"{prefix}{n}.xlsx"
        new_path = p.with_name(new_name)
        plan.append((p, new_path))
        n += 1

    # Detect collisions with existing files not in the rename set
    target_set = {new for _, new in plan}
    existing_conflicts = [new for new in target_set if new.exists() and new not in {old for old, _ in plan}]
    if existing_conflicts:
        raise FileExistsError(
            "Some target filenames already exist and are not part of the rename set:\n"
            + "\n".join(str(p) for p in existing_conflicts)
        )

    # Print plan
    for old, new in plan:
        print(f"{old.name}  ->  {new.name}")

    if dry_run:
        print("\n[dry_run=True] No files were renamed.")
        return plan

    # Two-step rename to avoid collisions (old -> temp -> final)
    temp_paths: list[tuple[Path, Path]] = []
    for i, (old, _) in enumerate(plan, start=1):
        temp = old.with_name(f".__renaming_tmp__{i}__{old.name}")
        old.rename(temp)
        temp_paths.append((temp, old))

    # Now temp -> final
    for (temp, _old), (_old2, final) in zip(temp_paths, plan):
        temp.rename(final)

    print("\nRenaming complete.")
    return plan


if __name__ == "__main__":
    rename_xlsx_sequential(
        "downloads",
        prefix="data_",
        start=1,
        recursive=False,
        dry_run=False,  
        sort_by="name",
    )
