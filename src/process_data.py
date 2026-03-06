import os
import google.genai as genai
from dotenv import load_dotenv
from pathlib import Path
import importlib
import utils.parsing as parsing
importlib.reload(parsing)
from utils.parsing import parse_workbook_all_sheets_to_markdown
from utils.prompt_loader import load_json_generator_prompts, load_report_generator_prompts
from google import genai
from utils.parsing_answer import parse_json_and_schema ,replace_original_table_in_report
import json

def process_data(file_path: Path):
    """지정된 엑셀 파일 경로를 받아 마크다운 변환, LLM 처리, 결과 저장을 수행합니다."""
    md_all = parse_workbook_all_sheets_to_markdown(
        file_path,
        max_total_chars=300_000,
        max_rows_per_sheet=2000,
        save_combined_markdown=False, # 디버깅 시 True로 변경하여 중간 결과 확인 가능
    )

    json_generator_prompts = load_json_generator_prompts()
    report_generator_prompts = load_report_generator_prompts() 

    report_prompt = report_generator_prompts + f'''=======INPUT MARKDOWN======{md_all}'''
    json_prompt = json_generator_prompts + f'''=======INPUT MARKDOWN======{md_all}''' 

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp_report = client.models.generate_content(
        model="gemini-2.0-flash", #gemini-3.0-flash #multi processing
        contents=report_prompt
    )
    resp_jsons = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=json_prompt
    )

    data = parse_json_and_schema(resp_jsons.text)
    report = replace_original_table_in_report(resp_report.text, md_all)

    # 파일명에서 확장자를 제거하여 결과 파일명으로 사용
    output_file_stem = file_path.stem

    # 보고서(txt) 저장
    p_report = PROJECT_ROOT / "data" / "report" / f"{output_file_stem}.txt"
    p_report.parent.mkdir(parents=True, exist_ok=True)
    p_report.write_text(report.replace("\r\n", "\n"), encoding="utf-8")

    # JSON 객체 저장
    p_json = PROJECT_ROOT / "data" / "json" / f"{output_file_stem}.json"
    p_json.parent.mkdir(parents=True, exist_ok=True)
    p_json.write_text(json.dumps(data["json_obj"], ensure_ascii=False, indent=2), encoding="utf-8")

    # JSON 스키마 저장
    p_schema = PROJECT_ROOT / "data" / "json_schema" / f"{output_file_stem}.json"
    p_schema.parent.mkdir(parents=True, exist_ok=True)
    p_schema.write_text(json.dumps(data["json_schema"], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv() 
    google_key = os.getenv("GEMINI_API_KEY")

    MODEL = "model/gemini-2.5-flash"
    
    # 프로젝트 루트 디렉토리 설정 (notebooks 폴더에서 실행해도 동일하게 동작)
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

    # 처리할 파일들이 있는 디렉토리
    input_dir = PROJECT_ROOT / "downloads"

    # 디렉토리에서 .xlsx 파일 목록 가져오기 (엑셀 임시 파일 제외)
    # missing = [9, 11, 12, 14, 17, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 107, 108, 109, 110, 112, 113, 117, 119, 121, 124, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 148, 151, 152, 153, 155, 156, 159, 160, 161, 163, 164, 165, 166, 167, 168, 170, 174, 176, 177, 179, 180, 181, 182, 183, 186, 187, 189, 190, 191, 192, 193, 194, 197, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 222, 223, 224, 225, 226, 228, 229, 232, 233, 236, 237, 239, 241, 243, 244, 246, 247, 248, 250, 251, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 281, 282, 283, 285, 286, 287, 289, 290, 291, 293, 294, 297, 298, 299, 305, 309, 323, 325, 326, 336, 337, 339, 344, 345, 346, 349, 351, 352, 353, 355, 356, 357, 358, 360, 361, 362, 365, 367, 374, 375, 376, 377, 378, 379, 385, 389, 391, 392, 393, 397, 398, 400, 401, 402, 403, 404, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 420, 421, 422, 425, 432, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 454, 455, 457, 458, 461, 462, 463, 467, 468, 475, 480, 481, 482, 484, 486, 488, 489, 490, 492, 493, 496, 501, 502, 505, 506, 507, 509, 515, 523, 524, 526, 529, 530, 532, 533, 534, 536, 537, 538, 539, 540, 541, 544, 545, 546, 548, 550, 551, 552, 553, 554, 557, 558, 559, 561, 562, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578, 579, 580, 585, 590, 591, 596, 597, 598, 599, 605, 607, 609, 610, 625, 626, 627, 628]
    excel_files = sorted([p for p in input_dir.glob("*.xlsx") if p.is_file() and not p.name.startswith("~$")])
    # excel_files = [f"data_{p}.xlsx" for p in missing]

    print(f"총 {len(excel_files)}개의 파일을 처리합니다.")
    for file_name in excel_files:
        file_path = input_dir / file_name
        try:
            print(f"--- 처리 시작: {file_path.name} ---")
            process_data(file_path)
            print(f"--- 처리 완료: {file_path.name} ---")
        except Exception as e:
            print(f"!!! 오류 발생 ({file_path.name}): {e} !!!")