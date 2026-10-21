#!/usr/bin/env python3
"""Create a local .env and generate an API secret if the example placeholder remains."""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"
PLACEHOLDERS = {"", "change-me", "replace-with-a-long-random-secret"}


def main() -> int:
    if not ENV_FILE.exists():
        if not EXAMPLE.is_file():
            print("ERROR: .env.example is missing", file=sys.stderr)
            return 2
        ENV_FILE.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created .env from .env.example")

    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    key_index = next((i for i, line in enumerate(lines) if line.startswith("API_KEY=")), None)
    if key_index is None:
        print("ERROR: .env must define API_KEY", file=sys.stderr)
        return 2
    key = lines[key_index].split("=", 1)[1].strip().strip("'\"")
    if key in PLACEHOLDERS:
        lines[key_index] = f"API_KEY={secrets.token_urlsafe(48)}"
        temp = ENV_FILE.with_suffix(".env.tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(temp, 0o600)
        temp.replace(ENV_FILE)
        print("Generated a fresh API_KEY and saved it in .env (not displayed).")
    elif len(key) < 32:
        print("ERROR: Existing API_KEY is shorter than 32 characters; set a strong secret in .env.", file=sys.stderr)
        return 2
    os.chmod(ENV_FILE, 0o600)
    print("Environment file is ready. Keep .env private and do not commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
