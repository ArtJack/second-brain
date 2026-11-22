"""A relative storage path in the environment is anchored to the checkout, not the shell.

`.env` sets `SB_STATE_DB=./data/artjeck.sqlite3`, and `config.py` used the value
as given, so it resolved against whichever directory a process started in. Run
from another folder, `sb list-tasks` opened an empty task database there, the
keyword index beside it was empty too, and keyword search fell back to the
legacy scan without logging anything (SB-F-59). The services escaped only
because launchd starts them in the repo.

These tests build `Config` the way import does, from the environment, and move
the working directory with `monkeypatch.chdir`. A relative value must land under
the package's own repo root, which keeps a worktree's `data/` its own. An
absolute value must come back exactly as given: that is how worktree code is
pointed at production state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from secondbrain.config import Config

# Worked out from this file rather than read from `config._ROOT`, so the tests
# check that value instead of trusting it.
REPO_ROOT = Path(__file__).resolve().parents[1]

PATH_SETTINGS = [
    ("SB_STATE_DB", "state_db"),
    ("SB_DATA", "persist_dir"),
    ("SB_MEMORY_DIR", "memory_dir"),
]


@pytest.mark.parametrize(("variable", "attribute"), PATH_SETTINGS)
def test_a_relative_path_resolves_under_the_repo_root_whatever_the_working_directory(
    variable: str, attribute: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(variable, "./data/relative-probe")

    for working_directory in (tmp_path, REPO_ROOT / "tests", REPO_ROOT):
        monkeypatch.chdir(working_directory)
        resolved = getattr(Config(), attribute)
        assert resolved == REPO_ROOT / "data" / "relative-probe", f"started in {working_directory}"


@pytest.mark.parametrize(("variable", "attribute"), PATH_SETTINGS)
def test_an_absolute_path_is_left_exactly_as_given(
    variable: str, attribute: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A worktree process, started in its own checkout, pointed at production's state.
    # The `..` survives only if nothing normalises or resolves the value: resolving
    # at import would stat a path that may sit on an unmounted volume.
    production = tmp_path / "production" / ".." / "production" / "data" / "probe"
    monkeypatch.setenv(variable, str(production))
    monkeypatch.chdir(REPO_ROOT)

    assert getattr(Config(), attribute) == production


@pytest.mark.parametrize(
    ("variable", "attribute", "default"),
    [
        ("SB_STATE_DB", "state_db", REPO_ROOT / "data" / "artjeck.sqlite3"),
        ("SB_DATA", "persist_dir", REPO_ROOT / "data" / "chroma"),
        ("SB_MEMORY_DIR", "memory_dir", REPO_ROOT / "data" / "memory"),
    ],
)
def test_an_empty_value_still_means_the_default(
    variable: str, attribute: str, default: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A set-but-empty value must not become `Path("")`, which is `.` — anchored, the repo root itself."""
    monkeypatch.setenv(variable, "")
    monkeypatch.chdir(tmp_path)

    assert getattr(Config(), attribute) == default


def test_the_default_ingest_roots_name_the_anchored_memory_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ingest allow-list names the memory dir, so it must name the anchored one.

    `mcp_server` resolves each root when it checks a path, so a second, unanchored
    reading of `SB_MEMORY_DIR` would admit a `data/...` under the server's working
    directory instead of the directory memories are written to.
    """
    monkeypatch.setenv("SB_MEMORY_DIR", "./data/relative-memory")
    monkeypatch.delenv("SB_INGEST_ROOTS", raising=False)
    monkeypatch.chdir(tmp_path)

    assert Config().ingest_roots == [str(REPO_ROOT), str(REPO_ROOT / "data" / "relative-memory")]
