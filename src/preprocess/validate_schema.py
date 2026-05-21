from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .common import delete_related_files, iter_json_files, read_json, resolve_path, write_json
except ImportError:  # pragma: no cover - supports `python src/preprocess/validate_schema.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocess.common import delete_related_files, iter_json_files, read_json, resolve_path, write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class SchemaCheck:
    stem: str
    json_path: str
    schema_path: str | None
    status: str
    error: str | None = None


def validate_json_schema_pair(json_path: Path, schema_path: Path) -> tuple[bool, str | None]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install jsonschema first: pip install jsonschema") from exc

    try:
        data = read_json(json_path)
        if data in ({}, []):
            return False, "empty JSON object/array"
        schema = read_json(schema_path)
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as exc:
        return False, f"schema validation failed: {exc.message}"
    except jsonschema.SchemaError as exc:
        return False, f"invalid schema: {exc.message}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_directory(
    *,
    json_dir: str | Path = "data/json",
    schema_dir: str | Path = "data/json_schema",
) -> list[SchemaCheck]:
    json_files = iter_json_files(json_dir)
    schema_path = resolve_path(schema_dir)
    if not schema_path.is_dir():
        raise NotADirectoryError(f"Schema directory not found: {schema_path}")

    results: list[SchemaCheck] = []
    iterator = json_files
    if tqdm is not None:
        iterator = tqdm(json_files, desc="schema validation", unit="file")

    for json_path in iterator:
        schema_file = schema_path / json_path.name
        if not schema_file.is_file():
            results.append(
                SchemaCheck(
                    stem=json_path.stem,
                    json_path=str(json_path),
                    schema_path=None,
                    status="missing_schema",
                    error="matching schema file not found",
                )
            )
            continue

        ok, error = validate_json_schema_pair(json_path, schema_file)
        results.append(
            SchemaCheck(
                stem=json_path.stem,
                json_path=str(json_path),
                schema_path=str(schema_file),
                status="valid" if ok else "invalid",
                error=error,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data/json/*.json against matching data/json_schema/*.json.")
    parser.add_argument("--json-dir", default="data/json")
    parser.add_argument("--schema-dir", default="data/json_schema")
    parser.add_argument("--data-dir", default="data", help="Base data directory for related-file deletion.")
    parser.add_argument("--delete-invalid", action="store_true")
    parser.add_argument("--delete-missing-schema", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Without this, deletion is dry-run.")
    parser.add_argument("--report-out", default=None, help="Optional JSON report path.")
    parser.add_argument("--show", type=int, default=10, help="Number of failing examples to print.")
    args = parser.parse_args()

    results = validate_directory(json_dir=args.json_dir, schema_dir=args.schema_dir)
    print(f"Paths")
    print(f"  json_dir:    {resolve_path(args.json_dir)}")
    print(f"  schema_dir:  {resolve_path(args.schema_dir)}")
    print(f"  data_dir:    {resolve_path(args.data_dir)}")
    counts = Counter(item.status for item in results)
    print("Schema validation")
    print(f"  total:           {len(results):,}")
    print(f"  valid:           {counts['valid']:,}")
    print(f"  invalid:         {counts['invalid']:,}")
    print(f"  missing_schema:  {counts['missing_schema']:,}")

    failures = [item for item in results if item.status != "valid"]
    for item in failures[: args.show]:
        print(f"  - {item.stem}: {item.status} ({item.error})")
    if len(failures) > args.show:
        print(f"  ... {len(failures) - args.show:,} more")

    delete_stems: list[str] = []
    if args.delete_invalid:
        delete_stems.extend(item.stem for item in results if item.status == "invalid")
    if args.delete_missing_schema:
        delete_stems.extend(item.stem for item in results if item.status == "missing_schema")

    if delete_stems:
        delete_result = delete_related_files(delete_stems, data_dir=args.data_dir, dry_run=not args.execute)
        mode = "deleted" if args.execute else "would delete"
        print(f"  {mode}: {len(delete_result.deleted):,} related files for {len(set(delete_stems)):,} stems")
        if delete_result.failed:
            print(f"  delete failures: {len(delete_result.failed):,}")

    if args.report_out:
        write_json(resolve_path(args.report_out), [asdict(item) for item in results])
        print(f"  report: {resolve_path(args.report_out)}")


if __name__ == "__main__":
    main()
