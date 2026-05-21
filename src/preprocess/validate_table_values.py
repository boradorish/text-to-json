from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .common import (
        delete_related_files,
        flatten_json_leaves,
        iter_json_files,
        read_json,
        resolve_path,
        stem_to_related_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - supports `python src/preprocess/validate_table_values.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocess.common import (
        delete_related_files,
        flatten_json_leaves,
        iter_json_files,
        read_json,
        resolve_path,
        stem_to_related_paths,
        write_json,
    )


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass
class ValueIssue:
    path: str
    value: str
    missing_tokens: list[str]


@dataclass
class TableValueCheck:
    stem: str
    status: str
    checked_values: int
    covered_values: int
    coverage: float
    issues: list[ValueIssue]
    error: str | None = None


def normalize_text(text: str) -> str:
    return "".join(TOKEN_RE.findall(text)).lower()


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def extract_markdown_table_text(report_text: str) -> str:
    blocks: list[str] = []
    current: list[str] = []
    pending_heading: str | None = None
    for line in report_text.splitlines():
        if "## Sheet:" in line or line.strip().startswith("Sheet:"):
            pending_heading = line
        if "|" in line:
            if pending_heading and not current:
                current.append(pending_heading)
                pending_heading = None
            current.append(line)
            continue
        if current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return "\n".join(blocks)


def value_is_table_covered(value: Any, *, table_compact: str, table_tokens: set[str]) -> tuple[bool, list[str]]:
    if value is None or isinstance(value, bool):
        return True, []

    text = str(value).strip()
    if not text:
        return True, []

    compact = normalize_text(text)
    if not compact:
        return True, []
    if compact in table_compact:
        return True, []

    value_tokens = tokens(text)
    missing = [token for token in value_tokens if token not in table_tokens]
    return not missing, sorted(set(missing))


def check_json_values_against_report_tables(
    json_obj: Any,
    report_text: str,
) -> tuple[int, int, list[ValueIssue]]:
    table_text = extract_markdown_table_text(report_text)
    table_compact = normalize_text(table_text)
    table_tokens = set(tokens(table_text))

    checked = 0
    covered = 0
    issues: list[ValueIssue] = []

    for path, value in flatten_json_leaves(json_obj):
        checked += 1
        ok, missing = value_is_table_covered(value, table_compact=table_compact, table_tokens=table_tokens)
        if ok:
            covered += 1
        else:
            issues.append(ValueIssue(path=path, value=str(value), missing_tokens=missing))

    return checked, covered, issues


def validate_directory(
    *,
    json_dir: str | Path = "data/json",
    data_dir: str | Path = "data",
) -> list[TableValueCheck]:
    results: list[TableValueCheck] = []
    for json_path in iter_json_files(json_dir):
        related = stem_to_related_paths(json_path.stem, data_dir=data_dir, include_missing=True)
        report_path = related["report"]
        if not report_path.is_file():
            results.append(
                TableValueCheck(
                    stem=json_path.stem,
                    status="missing_report",
                    checked_values=0,
                    covered_values=0,
                    coverage=0.0,
                    issues=[],
                    error=f"report not found: {report_path}",
                )
            )
            continue

        try:
            checked, covered, issues = check_json_values_against_report_tables(
                read_json(json_path),
                report_path.read_text(encoding="utf-8"),
            )
            coverage = covered / checked if checked else 1.0
            results.append(
                TableValueCheck(
                    stem=json_path.stem,
                    status="valid" if not issues else "invalid",
                    checked_values=checked,
                    covered_values=covered,
                    coverage=coverage,
                    issues=issues,
                )
            )
        except Exception as exc:
            results.append(
                TableValueCheck(
                    stem=json_path.stem,
                    status="error",
                    checked_values=0,
                    covered_values=0,
                    coverage=0.0,
                    issues=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analytically verify every JSON leaf value is covered by words/numbers in report markdown tables."
    )
    parser.add_argument("--json-dir", default="data/json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--delete-invalid", action="store_true")
    parser.add_argument("--delete-missing-report", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Without this, deletion is dry-run.")
    parser.add_argument("--report-out", default=None, help="Optional JSON report path.")
    parser.add_argument("--show", type=int, default=10, help="Number of invalid examples to print.")
    parser.add_argument("--max-issues-per-file", type=int, default=5)
    args = parser.parse_args()

    results = validate_directory(json_dir=args.json_dir, data_dir=args.data_dir)
    counts = Counter(item.status for item in results)
    invalid = [item for item in results if item.status == "invalid"]
    coverage_values = [item.coverage for item in results if item.checked_values]

    print("Table-value validation")
    print(f"  total:           {len(results):,}")
    print(f"  valid:           {counts['valid']:,}")
    print(f"  invalid:         {counts['invalid']:,}")
    print(f"  missing_report:  {counts['missing_report']:,}")
    print(f"  error:           {counts['error']:,}")
    if coverage_values:
        print(f"  avg coverage:    {sum(coverage_values) / len(coverage_values):.4f}")

    for item in invalid[: args.show]:
        print(f"  - {item.stem}: coverage={item.coverage:.3f} ({len(item.issues)} issue values)")
        for issue in item.issues[: args.max_issues_per_file]:
            print(f"      {issue.path}: {issue.value!r} missing={issue.missing_tokens}")
    if len(invalid) > args.show:
        print(f"  ... {len(invalid) - args.show:,} more invalid files")

    delete_stems: list[str] = []
    if args.delete_invalid:
        delete_stems.extend(item.stem for item in results if item.status in {"invalid", "error"})
    if args.delete_missing_report:
        delete_stems.extend(item.stem for item in results if item.status == "missing_report")

    if delete_stems:
        delete_result = delete_related_files(delete_stems, data_dir=args.data_dir, dry_run=not args.execute)
        mode = "deleted" if args.execute else "would delete"
        print(f"  {mode}: {len(delete_result.deleted):,} related files for {len(set(delete_stems)):,} stems")
        if delete_result.failed:
            print(f"  delete failures: {len(delete_result.failed):,}")

    if args.report_out:
        serializable = []
        for item in results:
            row = asdict(item)
            row["issues"] = [asdict(issue) for issue in item.issues]
            serializable.append(row)
        write_json(resolve_path(args.report_out), serializable)
        print(f"  report: {resolve_path(args.report_out)}")


if __name__ == "__main__":
    main()
