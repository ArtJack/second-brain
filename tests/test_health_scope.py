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
    import secondbrain.health as health

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
    import secondbrain.health as health

    monkeypatch.setattr(health, "_http_check", lambda name, url, *, timeout_s: health.HealthCheck(name, "pass", url))
    monkeypatch.setattr(health, "_launchd_check", lambda label: health.HealthCheck("launchd", "pass", "ok"))
    monkeypatch.setattr(health.cfg, "store_backend", "chroma")

    names = [check.name for check in _service_checks(timeout_s=1)]

    assert "qdrant" not in names


def test_other_projects_are_not_built_or_tested_unless_asked(tmp_path, monkeypatch):
    import secondbrain.health as health

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
    import secondbrain.health as health

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-not-a-real-key")
    monkeypatch.setenv("SB_MCP_TOKEN", "mcp-not-a-real-token")

    env = health._clean_env()

    assert "OPENAI_API_KEY" not in env
    assert "QDRANT_API_KEY" not in env
    assert "SB_MCP_TOKEN" not in env
    assert env.get("PATH")
