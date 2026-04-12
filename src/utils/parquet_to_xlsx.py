"""
parquet 파일을 xlsx로 변환합니다.

사용법:
    python src/utils/parquet_to_xlsx.py sheetpedia_xlsx/output_part_0000.parquet
    python src/utils/parquet_to_xlsx.py sheetpedia_xlsx/  # 디렉토리 전체
"""
import argparse
from pathlib import Path

import pandas as pd


def parquet_to_xlsx(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.with_suffix(".xlsx").name
    df = pd.read_parquet(input_path)
    df.to_excel(output_path, index=False)
    print(f"[OK] {input_path.name} → {output_path} ({len(df)}행)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="parquet 파일 또는 디렉토리")
    parser.add_argument("--output", default="sheetpedia_xlsx_converted", help="출력 디렉토리")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        parquet_to_xlsx(input_path, output_dir)
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        print(f"총 {len(files)}개 파일 변환 시작")
        for f in files:
            parquet_to_xlsx(f, output_dir)
    else:
        print(f"[ERROR] 경로를 찾을 수 없음: {input_path}")


if __name__ == "__main__":
    main()
