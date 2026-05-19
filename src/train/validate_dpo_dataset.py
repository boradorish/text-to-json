"""
DPO preference 데이터 검수 스크립트.

검수 목표:
  - LLaMA-Factory sharegpt DPO 포맷 확인
  - chosen 이 JSON parse + gold schema validation 을 통과하는지 확인
  - rejected 가 의도대로 gold schema validation 에 실패하는지 확인
  - chosen/rejected 동일, prose leakage, empty/null 과다, gold 불일치 등 위험 신호 집계

사용법:
    python src/train/validate_dpo_dataset.py
    python src/train/validate_dpo_dataset.py --data ../LLaMA-Factory/data/sunny_dpo.jsonl
    python src/train/validate_dpo_dataset.py --write-clean ../LLaMA-Factory/data/sunny_dpo.clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.parsing_answer import _extract_json_from_chunk
from utils.prompt_loader import find_project_root


PROJECT_ROOT = find_project_root()
DEFAULT_DATA = "../LLaMA-Factory/data/sunny_dpo.jsonl"
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "data" / "json_schema"
DEFAULT_GOLD_JSON_DIR = PROJECT_ROOT / "data" / "json"

SCHEMA_RE = re.compile(r"=== JSON Schema ===\s*([\s\S]+)$", re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"^\s*```json\s*\n[\s\S]*?\n```\s*$", re.IGNORECASE)
JSON_SECTION_RE = re.compile(
    r"===\s*JSON\s*===\s*([\s\S]*?)(?:===\s*JSON_SCHEMA\s*===|$)",
    re.IGNORECASE,
)
PLACEHOLDER_VALUES = {"", "n/a", "na", "null", "none", "unknown", "알 수 없음", "없음", "-"}


@dataclass
class ParsedOutput:
    ok: bool
    obj: Any | None = None
    text: str = ""
    error: str | None = None
    has_prose: bool = False
    empty_like_count: int = 0
    total_leaf_count: int = 0
    top_keys: set[str] = field(default_factory=set)
    had_json_section: bool = False


@dataclass
class ValidationResult:
    idx: int
    stem: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.errors and not self.warnings


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_records(data_path: Path) -> tuple[list[dict], list[str]]:
    raw_errors: list[str] = []
    if data_path.suffix == ".jsonl":
        records: list[dict] = []
        with data_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raw_errors.append(f"line {line_no}: JSONL 파싱 실패: {exc}")
                    continue
                if not isinstance(rec, dict):
                    raw_errors.append(f"line {line_no}: record가 object가 아님")
                    continue
                rec["_line_no"] = line_no
                records.append(rec)
        return records, raw_errors

    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 파일은 record list여야 합니다.")
    records = []
    for idx, rec in enumerate(data, start=1):
        if not isinstance(rec, dict):
            raw_errors.append(f"index {idx}: record가 object가 아님")
            continue
        rec["_line_no"] = idx
        records.append(rec)
    return records, raw_errors


def extract_schema_from_user_prompt(user_text: str) -> dict | None:
    match = SCHEMA_RE.search(user_text)
    if not match:
        return None
    try:
        schema = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    return schema if isinstance(schema, dict) else None


def load_schema(record: dict, schema_dir: Path) -> tuple[dict | None, str]:
    user_text = get_user_text(record)
    if user_text:
        schema = extract_schema_from_user_prompt(user_text)
        if schema is not None:
            return schema, "prompt"

    stem = str(record.get("_stem") or "")
    if stem:
        schema_path = schema_dir / f"{stem}.json"
        if schema_path.exists():
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None, "schema_file_parse_error"
            if isinstance(schema, dict):
                return schema, "schema_file"

    return None, "missing"


def get_user_text(record: dict) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("from") == "human":
            value = msg.get("value")
            return value if isinstance(value, str) else ""
    return ""


def get_assistant_value(record: dict, key: str) -> str | None:
    value = record.get(key)
    if not isinstance(value, dict):
        return None
    text = value.get("value")
    return text if isinstance(text, str) else None


def strip_think_block(text: str) -> str:
    parts = re.split(r"</think>", text, maxsplit=1)
    return parts[-1].strip()


def count_leaf_values(obj: Any) -> tuple[int, int]:
    if isinstance(obj, dict):
        total = 0
        empty = 0
        for value in obj.values():
            child_empty, child_total = count_leaf_values(value)
            empty += child_empty
            total += child_total
        return empty, total
    if isinstance(obj, list):
        total = 0
        empty = 0
        for value in obj:
            child_empty, child_total = count_leaf_values(value)
            empty += child_empty
            total += child_total
        return empty, total

    normalized = str(obj).strip().lower() if obj is not None else "null"
    return (1 if normalized in PLACEHOLDER_VALUES else 0), 1


def expected_top_keys(schema: dict) -> set[str]:
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        return set(schema["properties"].keys())
    return set()


def parse_model_output(text: str) -> ParsedOutput:
    clean = strip_think_block(text)
    section_match = JSON_SECTION_RE.search(clean)
    json_text = section_match.group(1).strip() if section_match else clean

    try:
        obj = _extract_json_from_chunk(json_text)
    except Exception as exc:
        return ParsedOutput(ok=False, text=clean, error=f"parse_fail: {exc}")

    empty_count, leaf_count = count_leaf_values(obj)
    top_keys = set(obj.keys()) if isinstance(obj, dict) else set()
    normalized = json_text.strip()
    has_prose = not (
        normalized.startswith("{")
        or normalized.startswith("[")
        or FENCED_JSON_RE.match(normalized)
    )
    return ParsedOutput(
        ok=True,
        obj=obj,
        text=clean,
        has_prose=has_prose,
        empty_like_count=empty_count,
        total_leaf_count=leaf_count,
        top_keys=top_keys,
        had_json_section=section_match is not None,
    )


def schema_error_name(schema: dict, obj: Any) -> str | None:
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return None
    except jsonschema.ValidationError as exc:
        return exc.validator or "validation_error"
    except jsonschema.SchemaError:
        return "schema_error"


def load_gold_json(stem: str, gold_json_dir: Path) -> Any | None:
    if not stem:
        return None
    path = gold_json_dir / f"{stem}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def check_format(record: dict, result: ValidationResult) -> None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        result.errors.append("format: conversations 없음 또는 list 아님")
    else:
        from_values = [m.get("from") for m in conversations if isinstance(m, dict)]
        if "human" not in from_values:
            result.errors.append("format: human message 없음")
        if "system" not in from_values:
            result.warnings.append("format: system message 없음")
        for pos, msg in enumerate(conversations):
            if not isinstance(msg, dict):
                result.errors.append(f"format: conversations[{pos}] object 아님")
                continue
            if not isinstance(msg.get("value"), str) or not msg.get("value", "").strip():
                result.errors.append(f"format: conversations[{pos}] value 비어 있음")

    for key in ("chosen", "rejected"):
        if get_assistant_value(record, key) is None:
            result.errors.append(f"format: {key}.value 없음 또는 string 아님")


def validate_record(
    record: dict,
    idx: int,
    *,
    schema_dir: Path,
    gold_json_dir: Path,
    max_empty_ratio: float,
) -> tuple[ValidationResult, Counter]:
    stem = str(record.get("_stem") or f"line:{record.get('_line_no', idx)}")
    result = ValidationResult(idx=idx, stem=stem)
    stats: Counter = Counter()

    if record.get("_skipped"):
        result.warnings.append("skipped marker record")
        stats["skipped_marker"] += 1
        return result, stats

    check_format(record, result)
    if result.errors:
        return result, stats

    schema, schema_source = load_schema(record, schema_dir)
    stats[f"schema_source:{schema_source}"] += 1
    if schema is None:
        result.errors.append("schema: prompt/schema file에서 schema를 찾지 못함")
        return result, stats

    chosen_text = get_assistant_value(record, "chosen") or ""
    rejected_text = get_assistant_value(record, "rejected") or ""
    chosen = parse_model_output(chosen_text)
    rejected = parse_model_output(rejected_text)

    if chosen_text.strip() == rejected_text.strip():
        result.errors.append("pair: chosen과 rejected 텍스트가 동일함")

    if not chosen.ok:
        result.errors.append(f"chosen: JSON 파싱 실패 ({chosen.error})")
    else:
        chosen_schema_error = schema_error_name(schema, chosen.obj)
        if chosen_schema_error:
            result.errors.append(f"chosen: schema validation 실패 ({chosen_schema_error})")
            stats[f"chosen_schema_error:{chosen_schema_error}"] += 1
        else:
            stats["chosen_schema_valid"] += 1

        if chosen.has_prose:
            result.warnings.append("chosen: JSON 앞뒤에 prose/불필요 텍스트 가능성")
        if chosen.total_leaf_count:
            empty_ratio = chosen.empty_like_count / chosen.total_leaf_count
            if empty_ratio > max_empty_ratio:
                result.warnings.append(
                    f"chosen: empty/null/unknown 계열 leaf 비율 높음 ({empty_ratio:.1%})"
                )

        gold_obj = load_gold_json(str(record.get("_stem") or ""), gold_json_dir)
        if gold_obj is not None and chosen.obj != gold_obj:
            result.warnings.append("chosen: data/json gold와 내용이 다름")

    if not rejected.ok:
        stats["rejected_parse_fail"] += 1
    else:
        rejected_schema_error = schema_error_name(schema, rejected.obj)
        if rejected_schema_error is None:
            result.errors.append("rejected: schema validation을 통과함 (bad negative 아님)")
            stats["rejected_schema_valid"] += 1
        else:
            stats[f"rejected_schema_error:{rejected_schema_error}"] += 1

        if rejected.has_prose:
            stats["rejected_prose_leakage"] += 1
        if rejected.total_leaf_count:
            empty_ratio = rejected.empty_like_count / rejected.total_leaf_count
            if empty_ratio > max_empty_ratio:
                stats["rejected_empty_like_high"] += 1

    allowed_keys = expected_top_keys(schema)
    if allowed_keys and chosen.top_keys:
        extra = chosen.top_keys - allowed_keys
        if extra:
            result.warnings.append(f"chosen: schema properties 밖 top-level key 있음 ({sorted(extra)[:5]})")

    return result, stats


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def print_examples(title: str, results: list[ValidationResult], attr: str, limit: int) -> None:
    rows = [
        (res, message)
        for res in results
        for message in getattr(res, attr)
    ]
    print_section(title)
    if not rows:
        print("  없음 ✓")
        return
    for res, message in rows[:limit]:
        print(f"  [{res.idx}] {res.stem}: {message}")
    if len(rows) > limit:
        print(f"  ... 외 {len(rows) - limit}건")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO JSONL 데이터 품질 검수")
    parser.add_argument("--data", default=DEFAULT_DATA, help="검수할 DPO json/jsonl 경로")
    parser.add_argument("--schema-dir", default=str(DEFAULT_SCHEMA_DIR), help="fallback gold schema 디렉토리")
    parser.add_argument("--gold-json-dir", default=str(DEFAULT_GOLD_JSON_DIR), help="chosen 비교용 gold JSON 디렉토리")
    parser.add_argument("--max-empty-ratio", type=float, default=0.5, help="empty/null 계열 leaf 경고 기준")
    parser.add_argument("--examples", type=int, default=20, help="오류/경고 예시 출력 개수")
    parser.add_argument("--write-clean", default=None, help="오류/경고가 없는 record만 JSONL로 저장")
    parser.add_argument("--write-error-report", default=None, help="오류/경고 report를 JSONL로 저장")
    args = parser.parse_args()

    data_path = resolve_project_path(args.data)
    schema_dir = resolve_project_path(args.schema_dir)
    gold_json_dir = resolve_project_path(args.gold_json_dir)

    if not data_path.exists():
        print(f"[ERROR] 파일 없음: {data_path}")
        sys.exit(1)

    records, raw_errors = read_records(data_path)
    print(f"로드: {data_path} ({len(records)} records)")
    if raw_errors:
        print(f"raw parse 오류: {len(raw_errors)}건")
        for err in raw_errors[: args.examples]:
            print(f"  {err}")

    results: list[ValidationResult] = []
    aggregate: Counter = Counter()
    stem_counts: Counter = Counter()
    clean_records: list[dict] = []

    for idx, record in enumerate(tqdm(records, desc="DPO 검수")):
        stem = str(record.get("_stem") or "")
        if stem:
            stem_counts[stem] += 1

        result, stats = validate_record(
            record,
            idx,
            schema_dir=schema_dir,
            gold_json_dir=gold_json_dir,
            max_empty_ratio=args.max_empty_ratio,
        )
        results.append(result)
        aggregate.update(stats)
        if result.is_clean:
            clean_records.append(record)

    duplicate_stems = {stem: count for stem, count in stem_counts.items() if count > 1}

    error_count = sum(1 for res in results if res.errors)
    warning_count = sum(1 for res in results if res.warnings)
    hard_fail_count = sum(len(res.errors) for res in results)
    warning_item_count = sum(len(res.warnings) for res in results)

    print_section("요약")
    print(f"  records:                 {len(records):>8,}")
    print(f"  clean records:            {sum(res.is_clean for res in results):>8,}")
    print(f"  records with errors:      {error_count:>8,}")
    print(f"  records with warnings:    {warning_count:>8,}")
    print(f"  total error items:        {hard_fail_count:>8,}")
    print(f"  total warning items:      {warning_item_count:>8,}")
    print(f"  duplicate stems:          {len(duplicate_stems):>8,}")

    print_section("주요 지표")
    interesting_keys = [
        "chosen_schema_valid",
        "rejected_parse_fail",
        "rejected_schema_valid",
        "rejected_prose_leakage",
        "rejected_empty_like_high",
        "skipped_marker",
        "schema_source:prompt",
        "schema_source:schema_file",
        "schema_source:missing",
    ]
    for key in interesting_keys:
        if aggregate.get(key):
            print(f"  {key:<32} {aggregate[key]:>8,}")

    schema_errors = {
        key: value
        for key, value in aggregate.items()
        if key.startswith("rejected_schema_error:") or key.startswith("chosen_schema_error:")
    }
    if schema_errors:
        print_section("Schema Error 분포")
        for key, value in sorted(schema_errors.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {key:<32} {value:>8,}")

    print_examples("오류 예시", results, "errors", args.examples)
    print_examples("경고 예시", results, "warnings", args.examples)

    if args.write_clean:
        clean_path = resolve_project_path(args.write_clean)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        with clean_path.open("w", encoding="utf-8") as f:
            for record, result in zip(records, results):
                if result.is_clean:
                    record = {k: v for k, v in record.items() if k != "_line_no"}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nclean JSONL 저장: {clean_path} ({len(clean_records)} records)")

    if args.write_error_report:
        report_path = resolve_project_path(args.write_error_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            for result in results:
                if result.errors or result.warnings:
                    f.write(json.dumps({
                        "idx": result.idx,
                        "stem": result.stem,
                        "errors": result.errors,
                        "warnings": result.warnings,
                    }, ensure_ascii=False) + "\n")
        print(f"오류/경고 report 저장: {report_path}")

    if error_count:
        sys.exit(2)


if __name__ == "__main__":
    main()
