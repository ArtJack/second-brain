"""`sb health` exits non-zero when a check fails, and the nightly still finishes.

The command only changed its panel from green to yellow, and exited 0 whatever
failed, so the nightly, and anything else that runs it, could not tell.

The checks themselves are covered in test_health.py and test_health_scope.py.
`run_health` builds them inline, so, as those files do, these tests replace
`_service_checks` by name: there is no seam to pass them in.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from secondbrain import cli, health
from secondbrain.config import cfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_health_cli(tmp_path, monkeypatch):
    """Run `sb health` with the given checks; reports go to tmp_path."""
    monkeypatch.setattr(cfg, "state_db", tmp_path / "state.sqlite3")

    def run(*checks: health.HealthCheck):
        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: list(checks))
        return CliRunner().invoke(cli.app, ["health"])

    return run


def test_a_failing_check_exits_1_after_saving_the_report(run_health_cli, tmp_path):
    result = run_health_cli(health.HealthCheck("llm", "pass", "ok"), health.HealthCheck("qdrant", "fail", "down"))

    assert result.exit_code == 1
    assert len(list((tmp_path / "health").glob("*-health.md"))) == 1


def test_all_checks_passing_exits_0(run_health_cli):
    result = run_health_cli(health.HealthCheck("llm", "pass", "ok"), health.HealthCheck("qdrant", "pass", "ok"))

    assert result.exit_code == 0


def test_a_skipped_check_is_not_a_failure(run_health_cli):
    result = run_health_cli(health.HealthCheck("llm", "pass", "ok"), health.HealthCheck("project app", "skip", "none"))

    assert result.exit_code == 0


def _run_nightly(tmp_path: Path, health_exit: int) -> tuple[int, list[str]]:
    """Run a copy of deploy/run-nightly.sh with `uv` replaced by a recorder.

    `uv` is an exported shell function, which bash prefers to any `uv` on the
    PATH, and the copy sits outside the repository, so even a real `uv` would
    find no project here. Nothing in this test can start a real nightly step.
    """
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    script = deploy / "run-nightly.sh"
    shutil.copy(ROOT / "deploy" / "run-nightly.sh", script)
    calls = tmp_path / "calls.log"
    driver = f"""
uv() {{ echo "$*" >> '{calls}'; if [ "$3" = health ]; then return {health_exit}; fi; return 0; }}
export -f uv
[ "$(bash -c 'type -t uv')" = function ] || exit 99
exec bash '{script}'
"""
    proc = subprocess.run(
        ["bash", "-c", driver],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 99, "could not shadow uv; refusing to run the nightly script"
    return proc.returncode, calls.read_text().splitlines()


def test_the_nightly_still_briefs_when_health_fails_and_ends_with_its_status(tmp_path):
    status, calls = _run_nightly(tmp_path, health_exit=1)

    assert [call.split()[2] for call in calls] == ["project-context", "overnight", "gc", "task-sync", "health", "morning"]
    assert status == 1


def test_the_nightly_exits_0_when_health_passes(tmp_path):
    status, calls = _run_nightly(tmp_path, health_exit=0)

    assert calls[-1] == "run sb morning"
    assert status == 0
