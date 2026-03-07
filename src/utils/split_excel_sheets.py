import os
import glob
import pandas as pd


def split_excel_sheets(input_dir: str, output_dir: str | None = None):
    """
    특정 디렉토리의 엑셀 파일들을 순회하며 sheet별로 분리해 저장.

    Args:
        input_dir: 엑셀 파일이 있는 디렉토리 경로
        output_dir: 분리된 파일을 저장할 디렉토리 경로 (None이면 input_dir 사용)
    """
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(
        os.path.join(input_dir, "*.xls")
    )

    if not excel_files:
        print(f"엑셀 파일을 찾을 수 없습니다: {input_dir}")
        return

    for excel_path in excel_files:
        base_name = os.path.splitext(os.path.basename(excel_path))[0]
        print(f"처리 중: {excel_path}")

        xl = pd.ExcelFile(excel_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            safe_sheet_name = sheet_name.replace("/", "_").replace("\\", "_")
            output_filename = f"{base_name}_{safe_sheet_name}.xlsx"
            output_path = os.path.join(output_dir, output_filename)
            df.to_excel(output_path, index=False)
            print(f"  저장: {output_path}")


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="엑셀 파일을 시트별로 분리해 저장합니다.")
    # parser.add_argument("input_dir", help="엑셀 파일이 있는 디렉토리")
    # parser.add_argument("--output_dir", help="저장할 디렉토리 (기본값: input_dir)", default=None)
    # args = parser.parse_args()

    # split_excel_sheets(args.input_dir, args.output_dir)
    split_excel_sheets("downloads_google/", "download_processed/")
