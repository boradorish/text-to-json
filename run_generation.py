# run_generation.py
import os
import google.generativeai as genai
from prompt_loader import load_prompts

def load_document_extracts(path="input.txt") -> str:
    return open(path, "r", encoding="utf-8").read()

def run():

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    system_prompt, developer_prompt, user_prompt_template, prompt_type = load_prompts()

    document_extracts = load_document_extracts()

    user_prompt = user_prompt_template.format(
        document_extracts=document_extracts
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=system_prompt
    )

    full_prompt = f"""
[DOCUMENT STYLE]
{developer_prompt}

[USER INSTRUCTION]
{user_prompt}
""".strip()

    response = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 8192
        }
    )

    output = response.text

    out_name = f"output_{prompt_type}.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ 생성 완료: {out_name} (type={prompt_type})")

if __name__ == "__main__":
    run()
