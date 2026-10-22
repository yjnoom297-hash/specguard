"""Load review policy and prompts from editable project data files."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent.parent / "config"


@lru_cache(maxsize=1)
def review_rules() -> dict:
    return json.loads((ROOT / "review_rules.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def prompt(name: str) -> str:
    return (ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8").strip()
