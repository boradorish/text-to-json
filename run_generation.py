# run_generation.py
import os
import argparse
import google.genai as genai
from prompt_loader import load_prompts

def run(xlsx_path: str, model: str = "gemini-2.0-flash"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GEMINI_API_KEY가 설정되어 있지 않습니다. 예) export GEMINI_API_KEY='...'" )

    client = genai.Client(api_key=api_key)

    # 1) 프롬프트 로드 (type 랜덤)
    system_prompt, developer_prompt, user_prompt_template, prompt_type = load_prompts()

    # 2) user 프롬프트 구성
    #    (xlsx raw 파일을 제공하므로, document_extracts는 비워두거나 간단 안내로 둠)
    #    USER_prompt.txt에 {document_extracts}가 없다면 아래 format은 제거해도 됨.
    try:
        user_prompt = user_prompt_template.format(document_extracts="")
    except KeyError:
        user_prompt = user_prompt_template

    full_prompt = f"""
[SYSTEM]
{system_prompt}

[DOCUMENT STYLE]
{developer_prompt}

[USER INSTRUCTION]
{user_prompt}
""".strip()

    # 3) 파일 업로드 (Files API)
    uploaded = client.files.upload(file=xlsx_path)

    # 4) 업로드 파일 + 프롬프트로 생성
    #    파일을 먼저 주고, 프롬프트를 뒤에 주는 게 보통 안정적입니다.
    resp = client.models.generate_content(
        model=model,
        contents=[
            uploaded,
            "\n\n",
            full_prompt,
        ],
    )

    output = resp.text or ""

    out_name = f"output_{prompt_type}.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ 생성 완료: {out_name} (type={prompt_type}, model={model})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="입력 엑셀(.xlsx) 파일 경로")
    parser.add_argument("--model", default="gemini-2.0-flash", help="사용할 Gemini 모델명")
    args = parser.parse_args()

    run(args.xlsx, model=args.model)

if __name__ == "__main__":
    main()
