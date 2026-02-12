# prompt_loader.py
import random
from pathlib import Path

PROMPT_DIR = Path("prompt")

def load_json_generator_prompts():
    system_prompt = (PROMPT_DIR / "json_SYSTEM_prompt.txt").read_text(encoding="utf-8")

    return system_prompt

def load_report_generator_prompts(prompt_type: str = 'report'):
    system_prompt = (PROMPT_DIR / "SYSTEM_prompt.txt").read_text(encoding="utf-8")
    user_prompt_template = (PROMPT_DIR / "USER_prompt.txt").read_text(encoding="utf-8")

    developer_prompt = list(PROMPT_DIR.glob(f"prompt_{prompt_type}.txt"))
    if not developer_prompt:
        raise FileNotFoundError(f"prompt_{prompt_type}.txt")
    
    full_prompt = f"""
    [SYSTEM]
    {system_prompt}

    [DOCUMENT STYLE]
    {developer_prompt}

    [USER INSTRUCTION]
    # {user_prompt_template}
    """.strip()

    return full_prompt