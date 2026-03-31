from __future__ import annotations

import re
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
    extensions: Iterable[str] = (".xlsx",),
) -> list[tuple[Path, Path]]:
    """
    이미 {prefix}{숫자}.xlsx 규칙에 맞는 파일은 건드리지 않고,
    규칙에 맞지 않는 파일만 기존 번호와 겹치지 않는 다음 번호로 rename합니다.

    Safety:
      - Uses a 2-step rename via temporary filenames to avoid collisions.
      - Returns list of (old_path, new_path).
    """
    d = Path(target_dir).resolve()
    if not d.exists() or not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}

    glob_prefix = "**/*" if recursive else "*"
    files: list[Path] = []
    for ext in exts:
        files.extend(p for p in d.glob(f"{glob_prefix}{ext}") if p.is_file())

    files = [p for p in files if not p.name.startswith("~$")]

    if sort_by == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name.lower())

    # 이미 규칙에 맞는 파일의 번호를 수집
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    used_ids: set[int] = set()
    already_named: set[Path] = set()
    for p in files:
        m = pattern.match(p.stem)
        if m:
            used_ids.add(int(m.group(1)))
            already_named.add(p)

    # 이상한 이름 파일들만 추림
    to_rename = [p for p in files if p not in already_named]

    # 겹치지 않는 번호 순서대로 할당
    def next_id(current: int) -> int:
        while current in used_ids:
            current += 1
        return current

    plan: list[tuple[Path, Path]] = []
    n = start
    for p in to_rename:
        n = next_id(n)
        new_name = f"{prefix}{n}.xlsx"
        new_path = p.with_name(new_name)
        plan.append((p, new_path))
        used_ids.add(n)
        n += 1

    if not plan:
        print("rename할 파일이 없습니다. (모두 이미 규칙에 맞는 이름)")
        return plan

    for old, new in plan:
        print(f"{old.name}  ->  {new.name}")

    if dry_run:
        print("\n[dry_run=True] No files were renamed.")
        return plan

    # Two-step rename to avoid collisions (old -> temp -> final)
    temp_paths: list[tuple[Path, Path]] = []
    for i, (old, _) in enumerate(plan, start=1):
        temp = old.with_name(f"data{i}.xlsx")
        old.rename(temp)
        temp_paths.append((temp, old))

    for (temp, _old), (_old2, final) in zip(temp_paths, plan):
        temp.rename(final)

    print("\nRenaming complete.")
    return plan


if __name__ == "__main__":
    # xlsx만 처리
    rename_xlsx_sequential(
        # "download_processed",
        #  "downloads",
        "sheetpedia_xlsx",
        prefix="data",
        start=20000,
        recursive=False,
        dry_run=False,
        sort_by="name",
    )

    # xlsx + csv 함께 처리
    # rename_xlsx_sequential(
    #     "downloads",
    #     prefix="data",
    #     extensions=(".xlsx", ".csv"),
    #     dry_run=True,
    # )
