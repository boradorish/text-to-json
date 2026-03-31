import os
from litellm import batch_completion
from dotenv import load_dotenv
from pathlib import Path
import importlib
import utils.parsing as parsing
importlib.reload(parsing)
from utils.parsing import parse_workbook_all_sheets_to_markdown
from utils.prompt_loader import load_json_generator_prompts, load_report_generator_prompts
from utils.parsing_answer import parse_json_and_schema, replace_original_table_in_report
import json
import litellm

MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")  # 예: LLM_MODEL=gpt-4o

def process_batch(file_paths: list, project_root: Path):
    """파일 배치를 한 번에 처리합니다 (최대 10개)."""
    json_generator_prompts = load_json_generator_prompts()
    report_generator_prompts = load_report_generator_prompts()

    # 각 파일의 마크다운 변환
    md_all_list = []
    for file_path in file_paths:
        md_all = parse_workbook_all_sheets_to_markdown(
            file_path,
            max_total_chars=300_000,
            max_rows_per_sheet=2000,
            save_combined_markdown=False,
        )
        md_all_list.append(md_all)

    # 배치 메시지 구성: [report_0, json_0, report_1, json_1, ...]
    messages = []
    for md_all in md_all_list:
        report_prompt = report_generator_prompts + f'''=======INPUT MARKDOWN======{md_all}'''
        json_prompt = json_generator_prompts + f'''=======INPUT MARKDOWN======{md_all}'''
        messages.append([{"role": "user", "content": report_prompt}])
        messages.append([{"role": "user", "content": json_prompt}])

    # batch_completion으로 한 번에 처리
    responses = batch_completion(model=MODEL, messages=messages)

    # 결과 저장
    for i, file_path in enumerate(file_paths):
        try:
            resp_report = responses[i * 2]
            resp_jsons = responses[i * 2 + 1]
            md_all = md_all_list[i]

            if isinstance(resp_report, Exception) or isinstance(resp_jsons, Exception):
                err = resp_report if isinstance(resp_report, Exception) else resp_jsons
                print(f"!!! API 오류 ({file_path.name}): {type(err).__name__}: {err} !!!")
                continue

            report_text = resp_report.choices[0].message.content
            json_text = resp_jsons.choices[0].message.content

            data = parse_json_and_schema(json_text)
            report = replace_original_table_in_report(report_text, md_all)

            output_file_stem = file_path.stem

            p_report = project_root / "data" / "report" / f"{output_file_stem}.txt"
            p_report.parent.mkdir(parents=True, exist_ok=True)
            p_report.write_text(report.replace("\r\n", "\n"), encoding="utf-8")

            p_json = project_root / "data" / "json" / f"{output_file_stem}.json"
            p_json.parent.mkdir(parents=True, exist_ok=True)
            p_json.write_text(json.dumps(data["json_obj"], ensure_ascii=False, indent=2), encoding="utf-8")

            p_schema = project_root / "data" / "json_schema" / f"{output_file_stem}.json"
            p_schema.parent.mkdir(parents=True, exist_ok=True)
            p_schema.write_text(json.dumps(data["json_schema"], ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"--- 처리 완료: {file_path.name} ---")
        except Exception as e:
            print(f"!!! 파일 오류 ({file_path.name}): {e} !!!")


if __name__ == "__main__":
    load_dotenv()
    print(f"사용 모델: {MODEL}")

    # import litellm
    # litellm._turn_on_debug()

    BATCH_SIZE = 5

    # 프로젝트 루트 디렉토리 설정 (notebooks 폴더에서 실행해도 동일하게 동작)
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

    # 처리할 파일들이 있는 디렉토리
    # input_dir = PROJECT_ROOT / "download_processed"
    input_dir = PROJECT_ROOT / "sheetpedia_xlsx"
    #7014부터 모델 바꿈
    #9491
    # 디렉토리에서 .xlsx 파일 목록 가져오기 (엑셀 임시 파일 제외)
    # missing = [9, 11, 12, 14, 17, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 107, 108, 109, 110, 112, 113, 117, 119, 121, 124, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 148, 151, 152, 153, 155, 156, 159, 160, 161, 163, 164, 165, 166, 167, 168, 170, 174, 176, 177, 179, 180, 181, 182, 183, 186, 187, 189, 190, 191, 192, 193, 194, 197, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 222, 223, 224, 225, 226, 228, 229, 232, 233, 236, 237, 239, 241, 243, 244, 246, 247, 248, 250, 251, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 281, 282, 283, 285, 286, 287, 289, 290, 291, 293, 294, 297, 298, 299, 305, 309, 323, 325, 326, 336, 337, 339, 344, 345, 346, 349, 351, 352, 353, 355, 356, 357, 358, 360, 361, 362, 365, 367, 374, 375, 376, 377, 378, 379, 385, 389, 391, 392, 393, 397, 398, 400, 401, 402, 403, 404, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 420, 421, 422, 425, 432, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 454, 455, 457, 458, 461, 462, 463, 467, 468, 475, 480, 481, 482, 484, 486, 488, 489, 490, 492, 493, 496, 501, 502, 505, 506, 507, 509, 515, 523, 524, 526, 529, 530, 532, 533, 534, 536, 537, 538, 539, 540, 541, 544, 545, 546, 548, 550, 551, 552, 553, 554, 557, 558, 559, 561, 562, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578, 579, 580, 585, 590, 591, 596, 597, 598, 599, 605, 607, 609, 610, 625, 626, 627, 628]
    MIN_ID = 0  # 이 숫자 이상인 파일만 처리 (0이면 전체)

    def get_file_id(p: Path) -> int:
        try:
            return int(p.stem.replace("data_", "").replace("data", ""))
        except ValueError:
            return -1

    excel_files = sorted([
        p for p in input_dir.glob("*.xlsx")
        if p.is_file()
        and not p.name.startswith("~$")
        and not (PROJECT_ROOT / "data" / "json" / f"{p.stem}.json").exists()
        and get_file_id(p) >= MIN_ID
    ])
    # excel_files = [f"data_{p}.xlsx" for p in missing]

    print(f"총 {len(excel_files)}개의 파일을 처리합니다.")

    for batch_start in range(0, len(excel_files), BATCH_SIZE):
        batch = excel_files[batch_start:batch_start + BATCH_SIZE]
        print(f"\n=== 배치 처리 시작: {[f.name for f in batch]} ===")
        try:
            process_batch(batch, PROJECT_ROOT)
        except Exception as e:
            print(f"!!! 배치 오류 발생: {e} !!!")
            for file_path in batch:
                print(f"  - {file_path.name}")