#!/usr/bin/env python3
"""Write a project-only .env for app.py.

Prompts for the OpenCode Zen API key with hidden input so it is not echoed to
the screen or the scrollback buffer, asks for the model id, and writes .env next
to this script. The key is never printed, not even masked.

Usage:
    python setup_env.py
    python setup_env.py --force        # overwrite an existing key
    python setup_env.py --env-file P   # write somewhere else (used by tests)

.env is gitignored. .env.example holds placeholders only.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ENV_KEY = "APP_OPENCODE_API_KEY"
ENV_MODEL = "APP_OPENCODE_MODEL"
DEFAULT_MODEL = "space-bunny-free"
DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"

KEY_PROMPT = "OpenCode Zen API key (input hidden, not echoed): "
MODEL_PROMPT = f"Model id [{DEFAULT_MODEL}]: "


class SetupError(Exception):
    """Setup could not complete."""


def read_existing(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SetupError(f"cannot read {path}: {exc}") from exc


def has_key(lines: Sequence[str]) -> bool:
    return any(line.strip().startswith(f"{ENV_KEY}=") and line.split("=", 1)[1].strip() for line in lines)


def write_env(path: Path, api_key: str, model: str = DEFAULT_MODEL) -> Path:
    """Write the two settings, keeping unrelated lines already in the file.

    The key value is never returned, logged, or printed.
    """
    if not api_key or not api_key.strip():
        raise SetupError(f"{ENV_KEY} must not be empty")
    if not model or not model.strip():
        raise SetupError(f"{ENV_MODEL} must not be empty")

    kept = [
        line
        for line in read_existing(path)
        if not line.strip().startswith(f"{ENV_KEY}=") and not line.strip().startswith(f"{ENV_MODEL}=")
    ]
    content = "\n".join(kept + [f"{ENV_KEY}={api_key.strip()}", f"{ENV_MODEL}={model.strip()}"]) + "\n"

    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise SetupError(f"cannot write {path}: {exc}") from exc
    restrict_permissions(path)
    return path


def restrict_permissions(path: Path) -> None:
    """Best effort: owner-only file mode where the platform supports it."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_env.py",
        description=f"Write a project-only .env with {ENV_KEY} and {ENV_MODEL}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing key instead of refusing",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="path to write (default: .env next to this script)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.env_file).resolve()

    try:
        if has_key(read_existing(path)) and not args.force:
            print(f"{ENV_KEY} is already set in {path}. Nothing was changed.", file=sys.stderr)
            print("Re-run with --force to replace it.", file=sys.stderr)
            return 1
    except SetupError as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 1

    try:
        api_key = getpass.getpass(KEY_PROMPT)
    except (EOFError, KeyboardInterrupt):
        print("\nsetup cancelled. Nothing was written.", file=sys.stderr)
        return 1

    try:
        answer = input(MODEL_PROMPT)
    except (EOFError, KeyboardInterrupt):
        print("\nsetup cancelled. Nothing was written.", file=sys.stderr)
        return 1

    model = answer.strip() or DEFAULT_MODEL

    try:
        written = write_env(path, api_key, model)
    except SetupError as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {written}")
    print(f"{ENV_KEY}: stored via hidden input, value not displayed")
    print(f"{ENV_MODEL}: {model}")
    print("This file is gitignored. Do not paste the key anywhere else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
