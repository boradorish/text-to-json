"""
JSONSchemaBench SFT 데이터 준비 스크립트 (Baseline)

epfl-dlab/JSONSchemaBench (train split)에서 N개의 SFT 학습 데이터를 생성합니다.
외부 라이브러리 없이 커스텀 minimal JSON 생성기를 사용합니다.
($ref, anyOf, oneOf, allOf, enum, const, 모든 primitive type 지원)

출력 포맷: LLaMA-Factory sharegpt

사용법:
    python src/prepare_jsonschemabench_sft.py
    python src/prepare_jsonschemabench_sft.py --num-samples 2500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()

SYSTEM_PROMPT = (
    "You are a JSON generation assistant. "
    "Given a JSON Schema, generate a valid JSON object that strictly conforms to it. "
    "Output ONLY the raw JSON object — no explanation, no markdown, no code fences."
)

USER_TEMPLATE = "Generate a valid JSON object conforming to the following JSON Schema:\n\n{schema}"

TRIVIAL_SCHEMAS = ({}, {"type": "object"}, {"type": "object", "properties": {}})


# ---------------------------------------------------------------------------
# Minimal JSON generator (no external deps)
# ---------------------------------------------------------------------------

def _resolve_ref(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    node = root
    for part in parts.replace("~1", "/").replace("~0", "~") if False else parts:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    return node if isinstance(node, dict) else {}


def _generate(schema: dict, root: dict, depth: int = 0) -> object:
    if depth > 10:
        return {}

    if not isinstance(schema, dict):
        return {}

    # $ref 해소
    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], root)
        merged = {**resolved, **{k: v for k, v in schema.items() if k != "$ref"}}
        return _generate(merged, root, depth + 1)

    # const
    if "const" in schema:
        return schema["const"]

    # enum
    if "enum" in schema:
        return schema["enum"][0]

    # type 정규화
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0])

    # anyOf / oneOf
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for sub in schema[key]:
                try:
                    val = _generate(sub, root, depth + 1)
                    jsonschema.validate(instance=val, schema=sub)
                    return val
                except Exception:
                    continue
            return {}

    # allOf: 첫 번째 서브스키마 기준으로 생성
    if "allOf" in schema:
        merged: dict = {}
        for sub in schema["allOf"]:
            merged.update(sub if isinstance(sub, dict) else {})
        return _generate(merged, root, depth + 1)

    # if/then
    if "if" in schema and "then" in schema:
        return _generate(schema["then"], root, depth + 1)

    # --- primitive types ---
    if schema_type == "null":
        return None

    if schema_type == "boolean":
        return True

    if schema_type == "integer":
        val = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        if isinstance(val, bool):
            val = 0
        val = int(val)
        if schema.get("exclusiveMinimum") == val:
            val += 1
        maximum = schema.get("maximum")
        if maximum is not None and val > int(maximum):
            val = int(maximum)
        multiple = schema.get("multipleOf")
        if multiple and val % multiple != 0:
            val = (val // multiple + 1) * multiple
        return val

    if schema_type == "number":
        val = schema.get("minimum", schema.get("exclusiveMinimum", 0.0))
        if isinstance(val, bool):
            val = 0.0
        return float(val)

    if schema_type == "string":
        min_len = schema.get("minLength", 1)
        pattern = schema.get("pattern")
        fmt = schema.get("format", "")
        if fmt == "date":
            return "2024-01-01"
        if fmt in ("date-time", "datetime"):
            return "2024-01-01T00:00:00Z"
        if fmt == "email":
            return "user@example.com"
        if fmt == "uri":
            return "https://example.com"
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        if fmt == "ipv4":
            return "0.0.0.0"
        if fmt == "ipv6":
            return "::1"
        if fmt == "time":
            return "00:00:00"
        return "a" * max(min_len, 1)

    if schema_type == "array":
        min_items = schema.get("minItems", 0)
        prefix = schema.get("prefixItems", [])
        items_schema = schema.get("items", {})

        result = []
        for pi in prefix:
            result.append(_generate(pi, root, depth + 1))
        while len(result) < min_items:
            result.append(_generate(items_schema, root, depth + 1) if isinstance(items_schema, dict) else "item")
        return result

    if schema_type == "object" or "properties" in schema or "required" in schema:
        result = {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for prop in required:
            prop_schema = properties.get(prop, {})
            result[prop] = _generate(prop_schema, root, depth + 1)
        # required 없으면 첫 번째 property라도 채우기
        if not required and properties:
            first = next(iter(properties))
            result[first] = _generate(properties[first], root, depth + 1)
        return result

    return {}


def generate_minimal(schema_obj: dict) -> object | None:
    try:
        result = _generate(schema_obj, schema_obj)
        jsonschema.validate(instance=result, schema=schema_obj)
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JSONSchemaBench → SFT baseline 데이터 준비")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=2500)
    parser.add_argument("--output", default="data/sft/jsonschemabench_sft_baseline.jsonl")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"epfl-dlab/JSONSchemaBench ({args.split} split) 다운로드 중...")
    ds = load_dataset("epfl-dlab/JSONSchemaBench", split=args.split)
    print(f"총 {len(ds)}개 로드 완료\n")

    written = skipped_parse = skipped_trivial = skipped_generate = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="처리 중", total=len(ds)):
            if written >= args.num_samples:
                break

            schema_str: str = item["json_schema"]
            unique_id: str = item["unique_id"]

            try:
                schema_obj = json.loads(schema_str)
                if not isinstance(schema_obj, dict):
                    skipped_parse += 1
                    continue
            except (json.JSONDecodeError, TypeError):
                skipped_parse += 1
                continue

            if schema_obj in TRIVIAL_SCHEMAS:
                skipped_trivial += 1
                continue

            result = generate_minimal(schema_obj)
            if result is None:
                skipped_generate += 1
                continue

            record = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": USER_TEMPLATE.format(schema=schema_str)},
                    {"from": "gpt", "value": json.dumps(result, ensure_ascii=False)},
                ],
                "unique_id": unique_id,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n완료.")
    print(f"  저장:                  {written}개")
    print(f"  스킵 (스키마 파싱 실패): {skipped_parse}개")
    print(f"  스킵 (trivial 스키마):  {skipped_trivial}개")
    print(f"  스킵 (JSON 생성 실패):  {skipped_generate}개")
    print(f"  출력: {output_path}")

    dataset_name = "jsonschemabench_sft_baseline"
    print(f"\n[LLaMA-Factory 데이터셋 등록]")
    print(f"  cp {output_path} /LLaMA-Factory/data/{output_path.name}")
    print(json.dumps({
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system",
            },
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
