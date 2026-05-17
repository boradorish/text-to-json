from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Find the text-to-json project root by walking upward until pyproject.toml
    or .git is found.
    """
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path(__file__).resolve().parents[2]
