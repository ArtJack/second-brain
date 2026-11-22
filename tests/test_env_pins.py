"""The suite reads no `.env`, and every variable the code reads is pinned.

`SB_LLM_TIMEOUT_S` was added to config.py without a pin in conftest.py. On the
owner's machine `load_dotenv()` then filled it from the real `.env` (180), and
`test_the_llm_client_has_a_bounded_timeout` (at most 120) failed in every
checkout under that one, while CI, which has no `.env`, stayed green. The QA
nightly runs in such a checkout, so it failed every night from 2026-10-17.

Two guards against the next one:

- python-dotenv's own off switch is set, and the installed version honours it,
  so no `.env` is read at all.
- Every environment variable the source reads is in `PINNED_ENV`. The list of
  variables comes from reading the source, not from someone remembering to add
  a line to conftest.py.

Variable names are safe to print; values are not. As in test_isolation.py,
every assertion that touches the environment or `cfg` compares a pre-computed
boolean: pytest prints both sides of a failed comparison, and a failed
`os.environ.get(...) == ...` prints the whole environment.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([*(ROOT / "src" / "secondbrain").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])

# Functions whose first argument is a variable name: `os.getenv` and the web
# server's `_env` / `_enabled` helpers. `os.environ[...]` and
# `os.environ.get(...)` are matched separately.
_READ_FUNCTIONS = {"getenv", "_env", "_enabled"}
_ENVIRON_METHODS = {"get", "setdefault", "pop"}
# A decision function may take the environment as a mapping (`env.get(...)`),
# so `.get` on anything counts when the key has a variable's shape. `pop` on a
# copy of the environment is a write, not a read, and does not count.
_VARIABLE_SHAPE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")


def _is_environ(node: ast.expr) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _literal(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def env_names_read(source: str) -> set[str]:
    """Every environment variable `source` reads by a literal name."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript) and _is_environ(node.value):
            name = _literal(node.slice)
        elif isinstance(node, ast.Call) and node.args:
            func, first = node.func, _literal(node.args[0])
            if isinstance(func, ast.Name):
                reads = func.id in _READ_FUNCTIONS
            elif isinstance(func, ast.Attribute):
                reads = (
                    func.attr == "getenv"
                    or (func.attr in _ENVIRON_METHODS and _is_environ(func.value))
                    or (func.attr == "get" and first is not None and _VARIABLE_SHAPE.fullmatch(first) is not None)
                )
            else:
                reads = False
            name = first if reads else None
        else:
            name = None
        if name:
            names.add(name)
    return names


def test_the_scan_sees_each_way_the_code_reads_a_variable():
    """The canary: if the scan stops seeing reads, the next test passes vacuously."""
    source = """
import os
from os import environ, getenv

os.getenv("A_GETENV", "1")
getenv("A_BARE_GETENV")
os.environ.get("A_ENVIRON_GET")
os.environ["A_SUBSCRIPT"]
environ.get("A_BARE_ENVIRON")
_env("A_SERVER_ENV", "x")
_enabled("A_SERVER_FLAG")
env.get("A_MAPPING_GET", "")
settings.get("not_a_variable")
copied = os.environ.copy()
copied.pop("A_WRITE_NOT_A_READ", None)
"""
    assert env_names_read(source) == {
        "A_GETENV",
        "A_BARE_GETENV",
        "A_ENVIRON_GET",
        "A_SUBSCRIPT",
        "A_BARE_ENVIRON",
        "A_SERVER_ENV",
        "A_SERVER_FLAG",
        "A_MAPPING_GET",
    }


def test_every_variable_the_code_reads_is_pinned(pinned_env: dict[str, str]):
    read: set[str] = set()
    for path in SOURCES:
        read |= env_names_read(path.read_text())

    # One read of each kind the real code uses, so an empty scan cannot pass.
    assert {"SB_LLM_TIMEOUT_S", "SB_MCP_TOKEN", "SB_WEB_REQUIRE_AUTH", "SB_PROJECT_CONTEXT_BUDGET_S"} <= read

    unpinned = sorted(read - set(pinned_env))
    assert not unpinned, (
        f"the code reads {unpinned} but tests/conftest.py does not pin them, so the suite "
        "takes whatever the shell exported. Add each one to PINNED_ENV."
    )


def test_python_dotenv_reads_nothing_while_the_suite_runs(tmp_path, monkeypatch):
    """The installed python-dotenv must honour the switch conftest.py sets.

    A version without it would load the owner's `.env` again, silently.
    """
    switch_set = os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
    assert switch_set, "conftest.py did not set PYTHON_DOTENV_DISABLED=1"

    env_file = tmp_path / ".env"
    env_file.write_text("SB_DOTENV_CANARY=loaded\n")
    monkeypatch.delenv("SB_DOTENV_CANARY", raising=False)
    try:
        loaded = dotenv.load_dotenv(env_file)
        leaked = "SB_DOTENV_CANARY" in os.environ
    finally:
        os.environ.pop("SB_DOTENV_CANARY", None)

    read_nothing = loaded is False and not leaked
    assert read_nothing, "python-dotenv ignored PYTHON_DOTENV_DISABLED and loaded a .env file"


def test_config_takes_the_pinned_values(pinned_env: dict[str, str], sandbox_root: Path):
    from secondbrain.config import cfg

    # A timeout is harmless to print, so it is compared directly.
    assert cfg.llm_timeout_s == float(pinned_env["SB_LLM_TIMEOUT_S"]) == 60.0

    reference_pinned = cfg.reference_collection == pinned_env["SB_REFERENCE_COLLECTION"]
    assert reference_pinned, "cfg.reference_collection is not the pinned test collection"
    not_live = cfg.reference_collection != "second_brain_reference"
    assert not_live, "cfg.reference_collection is the live reference collection"

    roots_pinned = [Path(root) for root in cfg.ingest_roots] == [sandbox_root]
    assert roots_pinned, "cfg.ingest_roots is not the pinned sandbox, so MCP ingest could read real files"
