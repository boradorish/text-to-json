# prompt_loader.py
import random
from pathlib import Path

PROMPT_DIR = Path("prompt")

def load_prompts():
    system_prompt = (PROMPT_DIR / "SYSTEM_prompt.txt").read_text(encoding="utf-8")
    user_prompt_template = (PROMPT_DIR / "USER_prompt.txt").read_text(encoding="utf-8")

    type_prompts = list(PROMPT_DIR.glob("prompt_*.txt"))
    if not type_prompts:
        raise FileNotFoundError("prompt/ 디렉토리에 prompt_*.txt 파일이 없습니다.")

    chosen = random.choice(type_prompts)
    developer_prompt = chosen.read_text(encoding="utf-8")
    prompt_type = chosen.stem.replace("prompt_", "")

    return system_prompt, developer_prompt, user_prompt_template, prompt_type
