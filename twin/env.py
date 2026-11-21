"""Load API keys from the project's .env file (standard library only, no python-dotenv needed).

Usage:
    from twin.env import load_env
    load_env()
    key = os.environ["GEMINI_API_KEY"]

Values already set in the environment (e.g. via `export`) take priority over the file.
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path=ENV_FILE):
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if value and name not in os.environ:
            os.environ[name] = value

