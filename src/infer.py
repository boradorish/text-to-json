"""
OpenAI API로 user_prompt → JSON 추론을 수행합니다.

사용법:
    python src/infer.py                          # data/user_prompt/ 전체 처리
    python src/infer.py --input data/user_prompt/data1.txt
    python src/infer.py --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()
SYSTEM_PROMPT = (PROJECT_ROOT / "prompt" / "json_SYSTEM_prompt.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------

def extract_json_from_output(text: str) -> Any | None:
    """
    모델 출력에서 JSON 객체를 파싱합니다.
    성공 시 dict/list 반환, 실패 시 None 반환.
    """
    content = text.strip()

    # ```json ... ``` 코드 블록 우선 시도
    m = re.search(r"```json\s*([\s\S]+?)\s*```", content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 첫 번째 { ... } 블록 추출
    m2 = re.search(r"(\{[\s\S]*\})", content)
    if m2:
        try:
            return json.loads(m2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def run_inference(client: OpenAI, model: str, user_text: str) -> dict:
    """
    user_prompt 텍스트 → OpenAI API → { "raw_output", "json_obj" } 반환.
    json_obj: 파싱된 dict 또는 None.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )

    raw_output = response.choices[0].message.content or ""
    json_obj = extract_json_from_output(raw_output)

    return {"raw_output": raw_output, "json_obj": json_obj}


# ---------------------------------------------------------------------------
# 파일 처리
# ---------------------------------------------------------------------------

def process_file(file_path: Path, client: OpenAI, model: str, output_dir: Path) -> None:
    stem = file_path.stem
    out_json = output_dir / f"{stem}.json"

    if out_json.exists():
        print(f"[SKIP] {stem}: 이미 존재함")
        return

    print(f"[PROCESSING] {file_path.name}")
    user_text = file_path.read_text(encoding="utf-8")
    result = run_inference(client, model, user_text)

    output_dir.mkdir(parents=True, exist_ok=True)

    if result["json_obj"] is not None:
        out_json.write_text(
            json.dumps(result["json_obj"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [OK] {out_json}")
    else:
        raw_dir = output_dir.parent / "json_infer_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{stem}.txt").write_text(result["raw_output"], encoding="utf-8")
        print(f"  [WARN] JSON 파싱 실패. raw 저장: {raw_dir / f'{stem}.txt'}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenAI API로 user_prompt → JSON 추론")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI 모델 이름")
    parser.add_argument("--input", default=None, help="txt 파일 또는 디렉토리 (기본: data/user_prompt/)")
    parser.add_argument("--output", default="data/json_infer", help="출력 디렉토리")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    output_dir = PROJECT_ROOT / args.output
    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "user_prompt"

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
    else:
        print(f"[ERROR] 경로를 찾을 수 없음: {input_path}")
        sys.exit(1)

    print(f"\n총 {len(files)}개 파일 처리 시작 (model: {args.model})\n")
    for file_path in files:
        process_file(file_path, client, args.model, output_dir)
    print("\n모든 파일 처리 완료.")


if __name__ == "__main__":
    main()
