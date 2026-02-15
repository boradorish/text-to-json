from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


SUPPORTED_EXCEL_EXTS = {".xlsx", ".xls", ".xlsm", ".xlsb"}
SUPPORTED_CSV_EXTS = {".csv"}


def _df_to_llm_markdown(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """
    Convert DataFrame to a markdown table string suitable for LLM input.
    Optionally truncate rows to avoid huge prompts.
    """
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()

    # Column name cleanup (optional but helps)
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    # Convert to markdown table
    return df.to_markdown(index=False)


def parse_raw_data(
    input_path: str | os.PathLike,
    *,
    # CSV options
    csv_encoding: str | None = "utf-8-sig",
    csv_sep: str | None = None,  # None => pandas tries to infer if engine="python" + sep=None (see code)
    # Excel options
    sheet_name: int | str | None = 0,  # default first sheet
    # Common options
    usecols: str | list[str] | None = None,
    skiprows: int | list[int] | None = None,
    nrows: int | None = None,
    dtype: dict | None = None,
    max_rows_for_markdown: int | None = None,  # truncate markdown rows if needed
    # Debug save options
    save_intermediate_markdown: bool = False,
    intermediate_markdown_path: str | os.PathLike | None = None,
) -> str:
    """
    Parse CSV or Excel into a DataFrame, then convert to LLM-input-friendly markdown.

    - Chooses parsing method by file extension.
    - Optionally saves the produced markdown to a given path for verification.

    Returns:
        markdown_table_str
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()

    # --- Load into DataFrame depending on file type ---
    if ext in SUPPORTED_CSV_EXTS:
        # If csv_sep is None, use python engine with sep=None to allow delimiter inference.
        # Note: inference isn't perfect; you can pass csv_sep="," or "\t" explicitly.
        read_kwargs = dict(
            encoding=csv_encoding,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
            dtype=dtype,
        )
        if csv_sep is None:
            df = pd.read_csv(path, sep=None, engine="python", **read_kwargs)
        else:
            df = pd.read_csv(path, sep=csv_sep, **read_kwargs)

    elif ext in SUPPORTED_EXCEL_EXTS:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
            dtype=dtype,
            engine=None,  # let pandas decide (usually openpyxl for xlsx)
        )
        # If sheet_name=None, pandas returns dict of DataFrames; handle that case.
        if isinstance(df, dict):
            # Concatenate sheets with a sheet indicator column
            frames = []
            for sh, d in df.items():
                d = d.copy()
                d.insert(0, "__sheet__", sh)
                frames.append(d)
            df = pd.concat(frames, ignore_index=True)

    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. "
            f"Supported: {sorted(SUPPORTED_CSV_EXTS | SUPPORTED_EXCEL_EXTS)}"
        )

    # --- Basic normalization (optional but practical) ---
    # Drop fully empty rows/cols to reduce noise
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # --- Convert to markdown ---
    markdown_table = _df_to_llm_markdown(df, max_rows=max_rows_for_markdown)

    # --- Optionally save intermediate markdown for verification ---
    if save_intermediate_markdown:
        if intermediate_markdown_path is None:
            # sensible default: alongside input file
            intermediate_markdown_path = path.with_suffix(".parsed.md")

        md_path = Path(intermediate_markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        md_path.write_text(markdown_table, encoding="utf-8")

    return markdown_table

def parse_workbook_all_sheets_to_markdown(
    input_path: str | os.PathLike,
    *,
    # Size control
    max_total_chars: int = 80_000,          # 전체 마크다운 최대 길이(문자 수)
    max_rows_per_sheet: int | None = 200,   # 시트별 row 제한 (LLM 입력용)
    # Optional: sheet filter
    include_sheets: list[str] | None = None,
    exclude_sheets: list[str] | None = None,
    # Debug save (combined markdown)
    save_combined_markdown: bool = False,
    combined_markdown_path: str | os.PathLike | None = None,
    # Pass-through options to parse_raw_data (common ones)
    usecols: str | list[str] | None = None,
    skiprows: int | list[int] | None = None,
    nrows: int | None = None,
    dtype: dict | None = None,
) -> str:
    """
    For an Excel workbook, parse all sheets using parse_raw_data(sheet_name=...),
    then concatenate each sheet's markdown until max_total_chars would be exceeded.

    Returns:
        combined_markdown (str)
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext not in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        raise ValueError(
            f"This function is for Excel workbooks with multiple sheets. "
            f"Got extension={ext}. If you need CSV, call parse_raw_data() directly."
        )

    # Get sheet names efficiently without loading everything
    xls = pd.ExcelFile(path)
    sheet_names = list(xls.sheet_names)

    # Apply include/exclude filters
    if include_sheets is not None:
        include_set = set(include_sheets)
        sheet_names = [s for s in sheet_names if s in include_set]
    if exclude_sheets is not None:
        exclude_set = set(exclude_sheets)
        sheet_names = [s for s in sheet_names if s not in exclude_set]

    parts: list[str] = []
    total_len = 0

    for sheet in sheet_names:
        # Parse this sheet to markdown (no intermediate save per-sheet here)
        sheet_md = parse_raw_data(
            path,
            sheet_name=sheet,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
            dtype=dtype,
            max_rows_for_markdown=max_rows_per_sheet,
            save_intermediate_markdown=False,
        )

        # Add a clear delimiter/header per sheet
        block = f"\n\n## Sheet: {sheet}\n\n{sheet_md}\n"

        # Stop before exceeding max_total_chars
        if total_len + len(block) > max_total_chars:
            # Optional: add a truncation note if there is still space
            note = (
                f"\n\n---\n"
                f"*Stopped concatenation before adding sheet '{sheet}' "
                f"because it would exceed max_total_chars={max_total_chars}.*\n"
            )
            if total_len + len(note) <= max_total_chars:
                parts.append(note)
                total_len += len(note)
            break

        parts.append(block)
        total_len += len(block)

    combined = "".join(parts).lstrip()

    # Optionally save combined markdown
    if save_combined_markdown:
        if combined_markdown_path is None:
            combined_markdown_path = path.with_suffix(".all_sheets.parsed.md")

        md_path = Path(combined_markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(combined, encoding="utf-8")

    return combined
