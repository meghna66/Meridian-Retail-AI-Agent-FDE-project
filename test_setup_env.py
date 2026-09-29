"""Tests for setup_env.py.

No network, no real prompt. Hidden input is monkeypatched and every run targets
a tmp_path, so the project's own .env is never touched.
"""

from __future__ import annotations

import builtins
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

import setup_env

BASE_DIR = Path(setup_env.__file__).resolve().parent
FAKE_KEY = "sk-ant-fake-0123456789-DO-NOT-USE"


def run_main(argv, key=FAKE_KEY, model=""):
    """Run setup_env.main with hidden input and input() stubbed out."""
    out, err = io.StringIO(), io.StringIO()
    prompts = []

    def fake_getpass(prompt=""):
        prompts.append(prompt)
        return key

    def fake_input(prompt=""):
        prompts.append(prompt)
        return model

    original_getpass = setup_env.getpass.getpass
    original_input = builtins.input
    setup_env.getpass.getpass = fake_getpass
    builtins.input = fake_input
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = setup_env.main(argv)
    finally:
        setup_env.getpass.getpass = original_getpass
        builtins.input = original_input

    return code, out.getvalue(), err.getvalue(), prompts


def test_env_file_is_written_with_key_and_default_model(tmp_path):
    env_path = tmp_path / ".env"
    code, out, err, prompts = run_main(["--env-file", str(env_path)])

    assert code == 0, err
    text = env_path.read_text(encoding="utf-8")
    assert f"{setup_env.ENV_KEY}={FAKE_KEY}" in text
    assert f"{setup_env.ENV_MODEL}={setup_env.DEFAULT_MODEL}" in text
    assert setup_env.DEFAULT_MODEL == "space-bunny-free"


def test_key_is_never_printed(tmp_path):
    env_path = tmp_path / ".env"
    code, out, err, prompts = run_main(["--env-file", str(env_path)])

    assert code == 0
    captured = out + err
    assert FAKE_KEY not in captured
    assert "not displayed" in out
    # prompts must not leak the value either
    assert FAKE_KEY not in "".join(prompts)


def test_key_input_is_hidden_not_echoed(tmp_path, monkeypatch):
    """The key must come from getpass, not from a visible input() prompt."""
    seen = {}

    def fake_getpass(prompt=""):
        seen["prompt"] = prompt
        return FAKE_KEY

    # input() is stubbed to return the model default only. If the key were read
    # with input() it would come back as "" and the .env would have no key.
    monkeypatch.setattr(setup_env.getpass, "getpass", fake_getpass)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")

    env_path = tmp_path / ".env"
    with redirect_stdout(io.StringIO()):
        assert setup_env.main(["--env-file", str(env_path)]) == 0

    assert "hidden" in seen["prompt"]
    assert f"{setup_env.ENV_KEY}={FAKE_KEY}" in env_path.read_text(encoding="utf-8")


def test_model_prompt_shows_the_default(tmp_path):
    env_path = tmp_path / ".env"
    _, _, _, prompts = run_main(["--env-file", str(env_path)], model="\n")
    assert any("space-bunny-free" in p for p in prompts)


def test_model_override_is_written(tmp_path):
    env_path = tmp_path / ".env"
    code, out, err, _ = run_main(["--env-file", str(env_path)], model="kimi-k3")
    assert code == 0
    assert f"{setup_env.ENV_MODEL}=kimi-k3" in env_path.read_text(encoding="utf-8")


def test_existing_key_is_not_clobbered_without_force(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(f"{setup_env.ENV_KEY}=sk-existing\n", encoding="utf-8")

    code, out, err, _ = run_main(["--env-file", str(env_path)])

    assert code == 1
    assert env_path.read_text(encoding="utf-8") == f"{setup_env.ENV_KEY}=sk-existing\n"
    assert "already set" in err
    assert FAKE_KEY not in out + err


def test_force_replaces_existing_key(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(f"{setup_env.ENV_KEY}=sk-existing\n", encoding="utf-8")

    code, out, err, _ = run_main(["--env-file", str(env_path), "--force"])

    assert code == 0
    text = env_path.read_text(encoding="utf-8")
    assert FAKE_KEY in text
    assert "sk-existing" not in text
    assert FAKE_KEY not in out + err


def test_unrelated_lines_are_preserved(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("# keep me\nSOMETHING_ELSE=1\n", encoding="utf-8")

    code, _, err = run_main(["--env-file", str(env_path)])[:3]

    assert code == 0, err
    text = env_path.read_text(encoding="utf-8")
    assert "# keep me" in text
    assert "SOMETHING_ELSE=1" in text
    assert FAKE_KEY in text
    assert text.count(f"{setup_env.ENV_KEY}=") == 1


def test_empty_key_is_rejected(tmp_path):
    env_path = tmp_path / ".env"
    with pytest.raises(setup_env.SetupError):
        setup_env.write_env(env_path, "   ")
    assert not env_path.exists()


def test_empty_model_is_rejected(tmp_path):
    env_path = tmp_path / ".env"
    with pytest.raises(setup_env.SetupError):
        setup_env.write_env(env_path, FAKE_KEY, "  ")
    assert not env_path.exists()


def test_cancelled_setup_writes_nothing(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"

    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(setup_env.getpass, "getpass", raise_eof)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = setup_env.main(["--env-file", str(env_path)])

    assert code == 1
    assert not env_path.exists()
    assert "cancelled" in err.getvalue()


def test_setup_env_never_prints_key_even_on_error_paths(tmp_path, monkeypatch):
    env_path = tmp_path / "nested" / "missing-dir" / ".env"
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = run_main(["--env-file", str(env_path)])[0]
    assert code == 1
    assert FAKE_KEY not in out.getvalue() + err.getvalue()


def test_env_example_is_placeholders_only_and_untouched():
    text = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
    assert f"{setup_env.ENV_KEY}=your-zen-api-key-here" in text
    assert f"{setup_env.ENV_MODEL}={setup_env.DEFAULT_MODEL}" in text
    for line in text.splitlines():
        if line.strip().startswith(f"{setup_env.ENV_KEY}="):
            value = line.split("=", 1)[1].strip()
            assert value == "your-zen-api-key-here"


def test_gitignore_covers_env_and_caches():
    text = (BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", "__pycache__/", ".venv/", ".pyenv/", ".pytest_cache/"):
        assert entry in text


def test_write_env_output_can_be_parsed_by_app(monkeypatch, tmp_path):
    """The written .env must satisfy the names app.py reads."""
    env_path = tmp_path / ".env"
    setup_env.write_env(env_path, FAKE_KEY)
    monkeypatch.setattr(
        setup_env, "DEFAULT_ENV_PATH", env_path, raising=False
    )
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        name, _, value = line.partition("=")
        assert name in (setup_env.ENV_KEY, setup_env.ENV_MODEL)
        assert value
