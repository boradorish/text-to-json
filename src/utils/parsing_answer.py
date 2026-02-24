from __future__ import annotations

import json
import re
from typing import TypedDict, Any


class ParsedDoc(TypedDict):
    json_obj: Any          # dict | list
    json_schema: dict


_SECTION_PATTERNS = {
    "report": r"===\s*REPORT\s*===",
    "json": r"===\s*JSON\s*===",
    "json_schema": r"===\s*JSON_SCHEMA\s*===",
}


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _extract_between(text: str, start_pat: str, end_pat: str | None) -> str:
    """
    Extract content between start marker and end marker (or end of text).
    """
    start = re.search(start_pat, text, flags=re.IGNORECASE)
    if not start:
        raise ValueError(f"Start section not found: {start_pat}")

    start_idx = start.end()

    if end_pat is None:
        chunk = text[start_idx:]
    else:
        end = re.search(end_pat, text[start_idx:], flags=re.IGNORECASE)
        if not end:
            raise ValueError(f"End section not found: {end_pat}")
        chunk = text[start_idx : start_idx + end.start()]

    return chunk.strip()


def _extract_json_from_chunk(chunk: str) -> Any:
    """
    Accepts either:
      - a ```json ... ``` code block
      - or raw JSON
    Returns parsed JSON (dict or list).
    """
    # Prefer fenced ```json ... ```
    m = re.search(r"```json\s*\n(.*?)\n```", chunk, flags=re.IGNORECASE | re.DOTALL)
    if m:
        json_text = m.group(1).strip()
    else:
        # Otherwise try to find first {...} or [...] block
        m2 = re.search(r"(\{.*\}|\[.*\])\s*$", chunk.strip(), flags=re.DOTALL)
        if not m2:
            # fallback: try anywhere
            m2 = re.search(r"(\{.*\}|\[.*\])", chunk, flags=re.DOTALL)
        if not m2:
            raise ValueError("JSON content not found in section.")
        json_text = m2.group(1).strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

def parse_json_and_schema(text: str) -> ParsedDoc:
    """
    Parse a document shaped like:
      === JSON ===
      ```json
      {...}
      ```
      === JSON_SCHEMA ===
      ```json
      {...}
      ```
    Returns:
      json_obj (dict|list), json_schema (dict)
    """
    s = _normalize_newlines(text)

    json_chunk = _extract_between(
        s,
        _SECTION_PATTERNS["json"],
        _SECTION_PATTERNS["json_schema"],
    )

    schema_chunk = _extract_between(
        s,
        _SECTION_PATTERNS["json_schema"],
        None,
    )

    json_obj = _extract_json_from_chunk(json_chunk)
    json_schema = _extract_json_from_chunk(schema_chunk)

    if not isinstance(json_schema, dict):
        raise ValueError("JSON_SCHEMA must be a JSON object (dict).")

    return {
        "json_obj": json_obj,
        "json_schema": json_schema,
    }


def replace_original_table_in_report(text:str, parsed_md:str) -> str:
    """
    보고서 텍스트의 '<original table>' 플레이스홀더를
    제공된 마크다운 테이블(`parsed_md`)로 바꿉니다.
    """
    return text.replace("<original_table>", parsed_md)