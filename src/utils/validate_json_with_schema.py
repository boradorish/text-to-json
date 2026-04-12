from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

try:
    import jsonschema
except ImportError:
    print("오류: 'jsonschema' 라이브러리가 설치되지 않았습니다.")
    print("설치 명령어: pip install jsonschema")
    exit(1)


def _delete_related_files(
    files_to_delete: list[Path],
    *,
    schema_dir: Path | None,
    report_dir: Path | None,
    dry_run: bool,
    label: str = "파일",
) -> list[Path]:
    """
    주어진 JSON 파일 목록과 연관된 스키마/리포트 파일을 함께 삭제합니다.

    Args:
        files_to_delete: 삭제할 JSON 파일 목록.
        schema_dir: JSON 스키마 파일이 있는 디렉터리.
        report_dir: 리포트 파일이 있는 디렉터리.
        dry_run: True이면 실제 삭제 없이 계획만 출력합니다.
        label: 출력 메시지에 사용할 레이블.

    Returns:
        삭제되었거나 삭제될 파일 경로 목록.
    """
    deleted_files: list[Path] = []

    action_verb = "삭제 예정" if dry_run else "삭제 중"
    print(f"\n--- {label} {action_verb} ---")

    for path in tqdm(files_to_delete, desc=f"파일 {action_verb}", unit="file"):
        schema_path = schema_dir / path.name if schema_dir else None
        report_path = report_dir / path.with_suffix(".txt").name if report_dir else None

        targets = [p for p in [path, schema_path, report_path] if p and p.is_file()]

        if not dry_run:
            failed = False
            for target in targets:
                try:
                    target.unlink()
                except OSError as e:
                    tqdm.write(f"  - 삭제 실패: {target} ({e})")
                    failed = True
            if failed:
                continue

        deleted_files.extend(targets)

    return deleted_files

def validate_json_files_with_schemas(
    json_dir: str | Path,
    schema_dir: str | Path,
    *,
    report_dir: str | Path | None = None,
    delete_invalid: bool = False,
    delete_missing: bool = False,
    dry_run: bool = False,
) -> tuple[list[Path], list[tuple[Path, str]], list[Path], list[Path]]:
    """
    지정된 디렉터리의 JSON 파일들을 해당 JSON 스키마와 비교하여 유효성을 검사합니다.
    - 두 디렉터리에서 이름이 같은 파일들을 찾아 짝을 맞춥니다.
    - jsonschema 라이브러리를 사용하여 유효성을 검사합니다.

    Args:
        json_dir: 유효성을 검사할 JSON 파일들이 있는 디렉터리.
        schema_dir: JSON 스키마 파일들이 있는 디렉터리.

        report_dir: 리포트 파일이 있는 디렉터리 (기본값: 'data/report').
        delete_invalid: True일 경우, 스키마 검사에 실패한 JSON/스키마/리포트 파일을 삭제합니다.
        delete_missing: True일 경우, 짝이 되는 스키마가 없는 JSON 파일을 삭제합니다.
        dry_run: True일 경우, 파일을 실제로 삭제하지 않고 실행 계획만 출력합니다.
    Returns:
        다음 네 가지 리스트를 담은 튜플:
        - valid_files (list[Path]): 유효성 검사를 통과한 JSON 파일 목록.
        - invalid_files (list[tuple[Path, str]]): 유효성 검사에 실패한 JSON 파일과 오류 메시지 목록.
        - missing_schema_files (list[Path]): 짝이 되는 스키마 파일을 찾지 못한 JSON 파일 목록.
        - deleted_files (list[Path]): 삭제되었거나 (dry_run=False) 삭제될 (dry_run=True) 파일 목록.
    """
    json_p = Path(json_dir).resolve()
    schema_p = Path(schema_dir).resolve()
    report_p = Path(report_dir).resolve() if report_dir else Path("data/report").resolve()

    if not json_p.is_dir():
        raise NotADirectoryError(f"JSON 디렉터리를 찾을 수 없습니다: {json_p}")
    if not schema_p.is_dir():
        raise NotADirectoryError(f"JSON 스키마 디렉터리를 찾을 수 없습니다: {schema_p}")

    json_files = sorted([p for p in json_p.glob("*.json") if p.is_file()])

    valid_files: list[Path] = []
    invalid_files: list[tuple[Path, str]] = []
    missing_schema_files: list[Path] = []
    deleted_files: list[Path] = []

    print(f"'{json_p.name}' 디렉터리에서 총 {len(json_files)}개의 JSON 파일을 검사합니다.")

    for json_path in tqdm(json_files, desc="JSON 유효성 검사 중", unit="file"):
        schema_path = schema_p / json_path.name

        if not schema_path.is_file():
            missing_schema_files.append(json_path)
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)

            jsonschema.validate(instance=json_data, schema=schema_data)
            valid_files.append(json_path)

        except (json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError) as e:
            invalid_files.append((json_path, str(e)))
        except Exception as e:
            invalid_files.append((json_path, f"예상치 못한 오류 발생: {e}"))

    # --- 결과 요약 출력 ---
    print("\n--- 검사 완료 ---")
    print(f"  - ✅ 유효한 파일: {len(valid_files)}개")
    print(f"  - ❌ 유효하지 않은 파일: {len(invalid_files)}개")
    print(f"  - ❓ 스키마 없음: {len(missing_schema_files)}개")

    # if invalid_files:
    #     print("\n[유효하지 않은 파일 상세 정보]")
    #     for path, error in invalid_files:
    #         print(f"  - 파일: {path.name}")
    #         print(f"    오류: {error[:200]}...") # 오류 메시지가 길 경우 일부만 표시

    # if missing_schema_files:
    #     print("\n[짝이 되는 스키마를 찾을 수 없는 파일]")
    #     for path in missing_schema_files:
    #         print(f"  - {path.name}")

    # if delete_invalid and invalid_files:
    #     invalid_paths = [path for path, _ in invalid_files]
    #     deleted_files.extend(_delete_related_files(
    #         invalid_paths,
    #         schema_dir=schema_p,
    #         report_dir=report_p,
    #         dry_run=dry_run,
    #         label="유효하지 않은 파일",
    #     ))

    # if delete_missing and missing_schema_files:
    #     deleted_files.extend(_delete_related_files(
    #         missing_schema_files,
    #         schema_dir=None,
    #         report_dir=report_p,
    #         dry_run=dry_run,
    #         label="스키마 없는 파일",
    #     ))

    if dry_run:
        print("\n[dry_run=True] 실제 파일 변경/삭제는 이루어지지 않았습니다.")

    return valid_files, invalid_files, missing_schema_files, deleted_files


def main():
    """커맨드 라인 인터페이스를 위한 메인 함수입니다."""
    parser = argparse.ArgumentParser(
        description="JSON 파일들이 스키마에 맞는지 유효성을 검사합니다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "json_dir", type=str, default="data/json", nargs="?",
        help="JSON 파일들이 있는 디렉터리 (기본값: 'data/json')"
    )
    parser.add_argument(
        "schema_dir", type=str, default="data/json_schema", nargs="?",
        help="JSON 스키마 파일들이 있는 디렉터리 (기본값: 'data/json_schema')"
    )
    parser.add_argument(
        "--report-dir", type=str, default="data/report",
        help="리포트 파일이 있는 디렉터리 (기본값: 'data/report')"
    )
    parser.add_argument(
        "--delete-invalid", action="store_true",
        help="스키마 검사에 실패한 JSON/스키마/리포트 파일을 삭제합니다."
    )
    parser.add_argument(
        "--delete-missing", action="store_true",
        help="짝이 되는 스키마 파일이 없는 JSON 파일을 삭제합니다."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="파일을 실제로 삭제하지 않고 실행 계획만 출력합니다."
    )
    args = parser.parse_args()

    try:
        validate_json_files_with_schemas(
            args.json_dir,
            args.schema_dir,
            report_dir=args.report_dir,
            delete_invalid=args.delete_invalid,
            delete_missing=args.delete_missing,
            dry_run=args.dry_run,
        )
    except (NotADirectoryError, FileNotFoundError) as e:
        print(f"오류: {e}")


if __name__ == "__main__":
    # 사용 예시:
    # python src/utils/validate_json_with_schema.py
    # python src/utils/validate_json_with_schema.py "path/to/my_jsons" "path/to/my_schemas"
    # python src/utils/validate_json_with_schema.py --delete-missing --dry-run
    main()