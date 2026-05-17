from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm


def clean_sparse_xlsx_files(
    target_dir: str | Path,
    *,
    threshold: float = 0.60,
    recursive: bool = False,
    dry_run: bool = True,
    delete_unopenable: bool = False,
) -> list[Path]:
    """
    지정된 디렉터리에서 데이터가 채워진 셀의 비율이 임계값 미만인 .xlsx 파일을 찾아 삭제합니다.

    - 각 파일의 모든 시트를 검사합니다.
    - 데이터 채워진 비율 = (비어 있지 않은 셀의 총 개수) / (전체 셀의 총 개수)

    Args:
        target_dir: 검사할 디렉터리 경로.
        threshold: 이 비율 미만이면 파일이 삭제 대상이 됩니다. (기본값: 0.60, 즉 60%)
        recursive: True이면 하위 디렉터리까지 모두 검사합니다.
        dry_run: True이면 파일을 실제로 삭제하지 않고 대상 목록만 출력합니다.
        delete_unopenable: True이면 열리지 않는(손상된) 파일도 삭제합니다.

    Returns:
        삭제되었거나 (dry_run=False) 삭제될 (dry_run=True) 파일 경로의 리스트.
    """
    d = Path(target_dir).resolve()
    if not d.is_dir():
        raise NotADirectoryError(f"디렉터리를 찾을 수 없습니다: {d}")

    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    files_to_check = [p for p in d.glob(pattern) if p.is_file() and not p.name.startswith("~$")]

    sparse_files: list[Path] = []
    unopenable_files: list[Path] = []

    print(f"총 {len(files_to_check)}개의 .xlsx 파일을 검사합니다...")

    for file_path in tqdm(files_to_check, desc="파일 검사 중", unit="file"):
        total_cells_in_workbook = 0
        non_empty_cells_in_workbook = 0

        try:
            # sheet_name=None으로 설정하여 모든 시트를 불러옵니다.
            xls = pd.read_excel(file_path, sheet_name=None)

            if not xls:  # 시트가 없는 경우
                fill_ratio = 0.0
            else:
                for sheet_name, df in xls.items():
                    total_cells_in_workbook += df.size
                    non_empty_cells_in_workbook += df.count().sum()

                if total_cells_in_workbook == 0:
                    fill_ratio = 0.0
                else:
                    fill_ratio = non_empty_cells_in_workbook / total_cells_in_workbook

            if fill_ratio < threshold:
                sparse_files.append(file_path)

        except Exception as e:
            unopenable_files.append(file_path)
            tqdm.write(f"  - 오류: {file_path.name} 파일을 읽는 중 에러 발생 ({e})")
            continue

    # --- 결과 요약 및 처리 ---
    print("\n--- 검사 완료 ---")

    deleted_files: list[Path] = []

    if sparse_files:
        print(f"\n[내용 부족 파일 목록 (데이터 비율 < {threshold:.0%})] - 총 {len(sparse_files)}개")
        for f in sparse_files:
            print(f"  - {f.name}")
    else:
        print("\n[내용 부족 파일 목록] - 해당 파일 없음")

    if unopenable_files:
        print(f"\n[읽기 실패 파일 목록] - 총 {len(unopenable_files)}개")
        for f in unopenable_files:
            print(f"  - {f.name}")
    else:
        print("\n[읽기 실패 파일 목록] - 해당 파일 없음")

    if dry_run:
        print("\n[dry_run=True] 실제 파일은 삭제되지 않았습니다.")
        # dry_run 모드에서도 삭제될 파일 목록을 반환값으로 설정
        deleted_files.extend(sparse_files)
        if delete_unopenable:
            deleted_files.extend(unopenable_files)
    else:
        print("\n--- 파일 삭제 실행 ---")
        files_to_process = (unopenable_files if delete_unopenable else []) + sparse_files 
        
        if not files_to_process:
            print("삭제할 파일이 없습니다.")
        else:
            for file_path in tqdm(files_to_process, desc="파일 삭제 중", unit="file"):
                try:
                    file_path.unlink()
                    deleted_files.append(file_path)
                    tqdm.write(f"  - 삭제 완료: {file_path.name}")
                except OSError as e:
                    tqdm.write(f"  - 삭제 실패: {file_path.name} ({e})")
            print("\n파일 삭제가 완료되었습니다.")

    return deleted_files


if __name__ == "__main__":
    # 사용 예시: 'downloads_google' 폴더에 있는 파일들을 대상으로 실행
    # 1. dry_run=True: 어떤 파일이 삭제될지 미리보기만 실행
    # 2. dry_run=False: 실제로 '내용 부족 파일'만 삭제
    # 3. dry_run=False, delete_unopenable=True: '내용 부족 파일'과 '읽기 실패 파일' 모두 삭제
    clean_sparse_xlsx_files(
        "downloads", threshold=0.1, recursive=False, dry_run=False, delete_unopenable=True
    )
