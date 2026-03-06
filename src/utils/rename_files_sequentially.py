from __future__ import annotations

import argparse
from pathlib import Path


def rename_files_sequentially(
    json_dir: str | Path = "data/json",
    schema_dir: str | Path = "data/json_schema",
    report_dir: str | Path = "data/report",
    *,
    prefix: str = "data",
    dry_run: bool = False,
) -> dict[str, str]:
    """
    data/json, data/json_schema, data/report 세 디렉터리의 파일들을
    베이스 이름 기준으로 그룹화하여 순서대로 넘버링합니다.
    동일한 베이스 이름을 가진 파일들은 동일한 번호를 부여받습니다.

    Args:
        json_dir: JSON 파일들이 있는 디렉터리.
        schema_dir: JSON 스키마 파일들이 있는 디렉터리.
        report_dir: 리포트 파일들이 있는 디렉터리.
        prefix: 새 파일 이름의 접두사 (기본값: 'data').
        dry_run: True이면 실제 이름 변경 없이 계획만 출력합니다.

    Returns:
        {기존 베이스 이름: 새 베이스 이름} 매핑 딕셔너리.
    """
    dirs = [Path(json_dir), Path(schema_dir), Path(report_dir)]

    # 각 디렉터리에서 베이스 이름 수집
    all_bases: set[str] = set()
    for d in dirs:
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    all_bases.add(f.stem)

    if not all_bases:
        print("처리할 파일이 없습니다.")
        return {}

    sorted_bases = sorted(all_bases)
    base_to_new: dict[str, str] = {
        base: f"{prefix}-{i + 1}" for i, base in enumerate(sorted_bases)
    }

    print(f"총 {len(base_to_new)}개의 고유 파일 그룹을 발견했습니다.")
    print(f"{'dry_run 모드 - 실제 변경 없음' if dry_run else '파일 이름 변경을 시작합니다.'}\n")

    rename_count = 0
    skip_count = 0

    for d in dirs:
        if not d.is_dir():
            print(f"[건너뜀] 디렉터리 없음: {d}")
            continue

        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue

            new_base = base_to_new.get(f.stem)
            if new_base is None:
                continue

            new_name = new_base + f.suffix
            new_path = f.parent / new_name

            if f.name == new_name:
                skip_count += 1
                continue

            print(f"  {d.name}/{f.name}  ->  {d.name}/{new_name}")

            if not dry_run:
                f.rename(new_path)
            rename_count += 1

    print(f"\n완료: {rename_count}개 변경, {skip_count}개 이미 동일한 이름으로 건너뜀.")
    if dry_run:
        print("[dry_run=True] 실제 파일 변경은 이루어지지 않았습니다.")

    return base_to_new


def main():
    parser = argparse.ArgumentParser(
        description="data/json, data/json_schema, data/report 디렉터리의 파일 이름을 순서대로 넘버링합니다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--json-dir", type=str, default="data/json",
        help="JSON 파일 디렉터리 (기본값: 'data/json')",
    )
    parser.add_argument(
        "--schema-dir", type=str, default="data/json_schema",
        help="JSON 스키마 파일 디렉터리 (기본값: 'data/json_schema')",
    )
    parser.add_argument(
        "--report-dir", type=str, default="data/report",
        help="리포트 파일 디렉터리 (기본값: 'data/report')",
    )
    parser.add_argument(
        "--prefix", type=str, default="data",
        help="새 파일 이름의 접두사 (기본값: 'data')",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 파일 변경 없이 실행 계획만 출력합니다.",
    )
    args = parser.parse_args()

    rename_files_sequentially(
        json_dir=args.json_dir,
        schema_dir=args.schema_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    # 사용 예시:
    # python src/utils/rename_files_sequentially.py
    # python src/utils/rename_files_sequentially.py --dry-run
    # python src/utils/rename_files_sequentially.py --prefix item --dry-run
    main()
