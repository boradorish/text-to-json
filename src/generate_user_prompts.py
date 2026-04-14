"""
각 report를 읽고 OpenAI API를 통해 per-report user_prompt를 생성합니다.

생성 결과는 data/user_prompt/{stem}.txt 에 저장됩니다.
최종 학습용 user_prompt = 생성된 요청문 + report 내용 + json_schema
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from litellm import batch_completion

MODEL = "openai/gpt-4.1"

PROMPT_GENERATION_SYSTEM_KR = """너는 데이터 추출 요청문을 작성하는 전문가야.
아래에 어떤 문서에 대한 분석 보고서가 주어질 거야.
이 보고서를 읽고, 실제 사용자가 작성할 법한 자연스러운 요청문을 한국어로 만들어줘.

규칙:
- 1~3문장으로 작성
- 구체적인 문서 내용을 반영해서 작성 (예: 어떤 기관, 어떤 데이터인지 언급)
- 반드시 "주어진 스키마(형식/구조/양식)를 따라 JSON을 추출/생성해달라"는 내용을 포함해야 함
- 실제 사람이 쓸 법한 자연스러운 어투
- "JSON 스키마", "json_schema" 같은 기술 용어 대신 "형식", "구조", "양식" 등의 표현 사용
- 요청문만 출력하고 다른 말은 하지 마

예시:
- "국가기술표준원 2016년 2월 업무추진비 집행 내역을 아래 주어진 형식에 맞춰 JSON으로 추출해줘."
- "한국해양대학교 장학 프로그램 전공 정보를 주어진 구조에 따라 JSON으로 만들어줘." """

PROMPT_GENERATION_SYSTEM_EN = """You are an expert at writing data extraction requests.
You will be given an analysis report about a document.
Read the report and write a natural request that a real user might write, in English.

Rules:
- Write 1~3 sentences
- Reflect specific content from the document (e.g., mention the organization or type of data)
- Must include a request to extract/generate JSON following the given schema (format/structure)
- Use natural, human-like phrasing
- Use terms like "format", "structure", or "schema" instead of technical jargon like "json_schema"
- Output only the request, nothing else

Examples:
- "Please extract the February 2016 business expense records from the Korea Agency for Technology and Standards into JSON following the given format."
- "Generate JSON for the scholarship program major information from Korea Maritime University based on the provided structure." """

BATCH_SIZE = 20


def build_user_prompt(report: str, json_schema: dict) -> str:
    schema_str = json.dumps(json_schema, ensure_ascii=False, indent=2)
    return f"""=== Report ===
{report}

=== JSON Schema ===
{schema_str}"""


def get_file_id(p: Path) -> int:
    try:
        return int(p.stem.replace("data_", "").replace("data", ""))
    except ValueError:
        return -1


def generate_user_prompts(project_root: Path, min_id: int = 0, max_id: int = 999999):
    report_dir = project_root / "data" / "report"
    schema_dir = project_root / "data" / "json_schema"
    output_dir = project_root / "data" / "user_prompt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # report와 json_schema가 모두 있고, 숫자 범위 안에 있고, 아직 생성되지 않은 파일만 처리
    report_files = sorted(report_dir.glob("*.txt"))
    candidates = [
        p for p in report_files
        if (schema_dir / f"{p.stem}.json").exists()
        and min_id <= get_file_id(p) <= max_id
        and not (output_dir / f"{p.stem}.txt").exists()
    ]

    print(f"처리할 파일: {len(candidates)}개 (이미 생성된 파일 제외)")

    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start:batch_start + BATCH_SIZE]

        # OpenAI에게 줄 메시지: 요청문 생성용
        messages = []
        for p in batch:
            report_text = p.read_text(encoding="utf-8")
            system_prompt = PROMPT_GENERATION_SYSTEM_EN if get_file_id(p) >= 20000 else PROMPT_GENERATION_SYSTEM_KR
            messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report_text},
            ])

        print(f"\n배치 {batch_start // BATCH_SIZE + 1}: {[p.name for p in batch]}")
        responses = batch_completion(model=MODEL, messages=messages)

        for p, resp in zip(batch, responses):
            try:
                generated_prompt = resp.choices[0].message.content.strip()

                # 최종 user_prompt = 생성된 요청문 + report + json_schema
                report_text = p.read_text(encoding="utf-8")
                schema_text = (schema_dir / f"{p.stem}.json").read_text(encoding="utf-8")
                json_schema = json.loads(schema_text)

                final_user_prompt = f"""{generated_prompt}

{build_user_prompt(report_text, json_schema)}"""

                out_path = output_dir / f"{p.stem}.txt"
                out_path.write_text(final_user_prompt, encoding="utf-8")
                print(f"  저장: {out_path.name}")
            except Exception as e:
                print(f"  오류 ({p.name}): {e}")


if __name__ == "__main__":
    load_dotenv()

    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name in ("src", "notebooks"):
        PROJECT_ROOT = PROJECT_ROOT.parent

    MIN_ID = 0
    MAX_ID = 999999999

    generate_user_prompts(PROJECT_ROOT, min_id=MIN_ID, max_id=MAX_ID)
