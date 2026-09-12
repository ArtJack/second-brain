"""Health checks probe what this system is configured to use, and nothing else.

Two things were wrong. The service probe asked every backend for Ollama's native
`/api/tags`, which a LiteLLM gateway answers with 404 — so the nightly reported a
failure every night for a service that was healthy, and a check that is always
red is a check nobody reads. And `sb health` discovered other repositories from
the ingest targets and ran their test and build commands, with this process's
environment (which `config.load_dotenv()` has already filled with the owner's
keys) inherited by the child. Running someone else's `npm test` is neither
read-only nor this program's business; QA belongs to the QA system.

Project checks stay available behind an explicit flag, because running them by
hand is a reasonable thing to want.
"""
from __future__ import annotations

from secondbrain.health import _service_checks, run_health


def test_the_llm_probe_asks_the_configured_endpoint(monkeypatch):
    from secondbrain import health

    asked: list[str] = []

    def fake_http_check(name, url, *, timeout_s):
        asked.append(url)
        return health.HealthCheck(name=name, status="pass", detail="ok")

    monkeypatch.setattr(health, "_http_check", fake_http_check)
    monkeypatch.setattr(health, "_launchd_check", lambda label: health.HealthCheck("launchd", "pass", "ok"))
    monkeypatch.setattr(health.cfg, "base_url", "http://gateway.invalid:4000/v1")
    monkeypatch.setattr(health.cfg, "store_backend", "qdrant")
    monkeypatch.setattr(health.cfg, "qdrant_url", "http://store.invalid:6333")

    _service_checks(timeout_s=1)

    assert asked == ["http://gateway.invalid:4000/v1/models", "http://store.invalid:6333/collections"]


def test_a_chroma_deployment_is_not_asked_about_qdrant(monkeypatch):
    from secondbrain import health

    monkeypatch.setattr(health, "_http_check", lambda name, url, *, timeout_s: health.HealthCheck(name, "pass", url))
    monkeypatch.setattr(health, "_launchd_check", lambda label: health.HealthCheck("launchd", "pass", "ok"))
    monkeypatch.setattr(health.cfg, "store_backend", "chroma")

    names = [check.name for check in _service_checks(timeout_s=1)]

    assert "qdrant" not in names


def test_other_projects_are_not_built_or_tested_unless_asked(tmp_path, monkeypatch):
    from secondbrain import health

    ran: list[str] = []
    monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "pass", "ok")])
    monkeypatch.setattr(
        health,
        "_project_checks",
        lambda timeout_s: ran.append("project checks") or [health.HealthCheck("project app", "pass", "ok")],
    )

    res = run_health(output_dir=tmp_path)

    assert ran == []
    assert [check["name"] for check in res["checks"]] == ["llm"]

    health.run_health(output_dir=tmp_path, include_projects=True)

    assert ran == ["project checks"]


def test_a_project_subprocess_never_inherits_the_loaded_secrets(monkeypatch):
    from secondbrain import health

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-not-a-real-key")
    monkeypatch.setenv("SB_MCP_TOKEN", "mcp-not-a-real-token")

    env = health._clean_env()

    assert "OPENAI_API_KEY" not in env
    assert "QDRANT_API_KEY" not in env
    assert "SB_MCP_TOKEN" not in env
    assert env.get("PATH")


class TestTransitions:
    """A report of current state is a report nobody reads by the third night.

    `health` printed the same "2 failed" every morning for weeks. Both failures
    were by construction — a probe pointed at the wrong endpoint — and because
    the report never distinguished "still failing" from "just started failing",
    there was no way to notice the day something real broke. Alerting is about
    change; state belongs below the fold.
    """

    def test_a_first_run_alerts_on_nothing(self, tmp_path, monkeypatch):
        from secondbrain import health

        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "fail", "down")])
        res = health.run_health(output_dir=tmp_path)

        assert res["transitions"]["new_fail"] == [], "nothing is 'newly' anything on a first run"
        assert "no previous run" in res["markdown"].lower()

    def test_a_failure_that_persists_is_not_re_alerted(self, tmp_path, monkeypatch):
        from secondbrain import health

        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "fail", "down")])
        health.run_health(output_dir=tmp_path)
        res = health.run_health(output_dir=tmp_path)

        assert res["transitions"]["new_fail"] == []
        assert res["transitions"]["unchanged"] == ["llm"]

    def test_a_newly_failing_check_leads_the_report(self, tmp_path, monkeypatch):
        from secondbrain import health

        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "pass", "ok")])
        health.run_health(output_dir=tmp_path)
        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "fail", "gone")])
        res = health.run_health(output_dir=tmp_path)

        assert res["transitions"]["new_fail"] == ["llm"]
        assert "1 check(s) newly failing" in res["markdown"]

    def test_a_recovery_is_reported_too(self, tmp_path, monkeypatch):
        from secondbrain import health

        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "fail", "down")])
        health.run_health(output_dir=tmp_path)
        monkeypatch.setattr(health, "_service_checks", lambda timeout_s: [health.HealthCheck("llm", "pass", "ok")])
        res = health.run_health(output_dir=tmp_path)

        assert res["transitions"]["recovered"] == ["llm"]


class TestDeployedLabels:
    def test_the_launchd_labels_come_from_the_plists_this_repo_ships(self):
        """A hardcoded label can be green while the service that exists is down."""
        from secondbrain.health import deployed_labels

        labels = deployed_labels()

        assert "com.secondbrain.mcp" in labels, labels
        assert any("sb-web" in label for label in labels), labels
