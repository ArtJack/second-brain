from secondbrain.health import HealthCheck, _project_command, render_health_report, run_health


def test_render_health_report_summarizes_statuses():
    markdown = render_health_report(
        stamp="2026-06-04_09-00-00",
        checks=[
            HealthCheck("ollama", "pass", "HTTP 200"),
            HealthCheck("project app", "fail", "tests failed", "uv run pytest", 123),
            HealthCheck("project docs", "skip", "no command"),
        ],
    )

    assert "- Passed: 1" in markdown
    assert "- Failed: 1" in markdown
    assert "- Skipped: 1" in markdown
    assert "FAIL project app: tests failed" in markdown
    assert "Command: `uv run pytest`" in markdown


def test_project_command_prefers_uv_pytest_for_python_tests(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='app'\n")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "tests").mkdir()

    assert _project_command(tmp_path) == ["uv", "run", "pytest"]


def test_project_command_uses_npm_build_when_no_test_script(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts":{"build":"tsc"}}')

    assert _project_command(tmp_path) == ["npm", "run", "build"]


def test_run_health_writes_report_with_patched_checks(tmp_path, monkeypatch):
    import secondbrain.health as health

    monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [HealthCheck("ollama", "pass", "ok")])
    monkeypatch.setattr(health, "_project_checks", lambda timeout_s: [HealthCheck("project app", "skip", "no command")])

    res = run_health(output_dir=tmp_path)

    assert res["passed"] == 1
    assert res["skipped"] == 1
    assert "Health Report" in res["markdown"]
    assert (tmp_path / res["path"].split("/")[-1]).exists()
