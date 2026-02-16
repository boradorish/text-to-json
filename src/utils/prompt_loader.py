# prompt_loader.py
import random
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Find project root by walking upward until pyproject.toml (preferred) or .git is found.
    """
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    raise FileNotFoundError("Project root not found (pyproject.toml or .git missing).")

PROJECT_ROOT = find_project_root()
PROMPT_DIR = PROJECT_ROOT / "prompt"

def load_json_generator_prompts():
    print(PROMPT_DIR)
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
    
    [USER INSTRUCTION]
    # {user_prompt_template}
    """.strip()

    return full_prompt