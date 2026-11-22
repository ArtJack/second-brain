"""A running server picks up a changed .env on its next tool call (`config.refresh`).

`config` read .env once, at import, and the MCP server lives as long as the
Claude session that started it. On 2026-10-16 fourteen open sessions still held
the settings they had started with: five sent the lab gateway's master key,
rotated three days before, and got a 401 on every call; nine still answered
from the lab's store after the cut-over to Oracle, and a `learn` there wrote to
the old share. Each needed its process stopped by hand.

Hermetic: each test writes its own .env in tmp_path and has `config` treat it as
the file read at import. Every URL is the suite's dead port; nothing reaches a
model, a gateway or a store. `refresh()` changes os.environ, cfg and the cached
gateway client directly, so the fixture puts all three back.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from secondbrain import config, llm
from secondbrain.config import cfg, refresh


def _write(path: Path, values: dict[str, str], age_s: float) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    then = time.time() - age_s
    os.utime(path, (then, then))


class EnvFile:
    """The .env this process 'read at import', and the edits made to it since."""

    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = path
        self._monkeypatch = monkeypatch

    def loaded(self, **values: str) -> None:
        """Start from this file, as `load_dotenv()` would have left the process."""
        _write(self.path, values, age_s=60)
        os.environ.update(values)
        self._monkeypatch.setattr(config, "_ENV_FILE", str(self.path))
        self._monkeypatch.setattr(config, "_from_file", dict(values))
        self._monkeypatch.setattr(config, "_stamp", config._digest(self.path.read_text(encoding="utf-8")))
        cfg.__dict__.update(config.Config().__dict__)

    def edited(self, age_s: float = 10, **values: str) -> None:
        _write(self.path, values, age_s)


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    environ, settings = dict(os.environ), dict(cfg.__dict__)
    # Put back by monkeypatch at teardown, whatever _openai() rebuilt meanwhile.
    monkeypatch.setattr(llm, "_client", llm._client)
    monkeypatch.setattr(llm, "_client_for", llm._client_for)
    yield EnvFile(tmp_path / ".env", monkeypatch)
    for key in os.environ.keys() - environ.keys():
        del os.environ[key]
    os.environ.update(environ)
    cfg.__dict__.update(settings)


def test_a_changed_env_file_reaches_cfg_on_the_next_refresh(env_file):
    env_file.loaded(OPENAI_BASE_URL="http://127.0.0.1:9/v1", CHAT_MODEL="chat-before")
    env_file.edited(OPENAI_BASE_URL="http://127.0.0.1:9/v2", CHAT_MODEL="chat-after")

    assert refresh() is True
    assert cfg.base_url == "http://127.0.0.1:9/v2"
    assert cfg.chat_model == "chat-after"


def test_an_unchanged_file_is_not_applied_again(env_file):
    env_file.loaded(CHAT_MODEL="chat-before")

    assert refresh() is False
    assert cfg.chat_model == "chat-before"


def test_a_variable_the_process_inherited_keeps_winning_over_the_file(env_file):
    """`load_dotenv()`'s rule at import. It is what keeps conftest's pins, and a
    launchd job's own environment, in force."""
    env_file.loaded(CHAT_MODEL="chat-before")
    env_file.edited(CHAT_MODEL="chat-after", SB_STORE="qdrant")

    assert refresh() is True
    assert cfg.chat_model == "chat-after"
    assert cfg.store_backend == "chroma"


def test_a_variable_dropped_from_the_file_returns_to_its_default(env_file):
    env_file.loaded(CHAT_MODEL="chat-before", EMBED_MODEL="embed-kept")
    env_file.edited(EMBED_MODEL="embed-kept")

    assert refresh() is True
    assert cfg.chat_model == "llama3.1:8b"
    assert cfg.embed_model == "embed-kept"


def test_a_variable_the_process_set_itself_is_left_alone(env_file):
    env_file.loaded(CHAT_MODEL="chat-before")
    os.environ["CHAT_MODEL"] = "chat-set-at-runtime"
    env_file.edited(CHAT_MODEL="chat-after")

    refresh()
    assert cfg.chat_model == "chat-set-at-runtime"


def test_a_file_written_this_second_waits_for_a_later_refresh(env_file):
    """`sops -d … > .env` truncates the file before writing it; a half-written
    file would point the server at the defaults."""
    env_file.loaded(CHAT_MODEL="chat-before")
    env_file.edited(age_s=0, CHAT_MODEL="chat-after")

    assert refresh() is False
    assert cfg.chat_model == "chat-before"

    then = time.time() - 10
    os.utime(env_file.path, (then, then))
    assert refresh() is True
    assert cfg.chat_model == "chat-after"


def test_a_missing_file_keeps_the_current_settings(env_file):
    env_file.loaded(CHAT_MODEL="chat-before")
    env_file.path.unlink()

    assert refresh() is False
    assert cfg.chat_model == "chat-before"


def test_the_gateway_client_follows_a_refreshed_url_and_key(env_file):
    env_file.loaded(OPENAI_BASE_URL="http://127.0.0.1:9/v1", OPENAI_API_KEY="key-before")
    before = llm._openai()
    env_file.edited(OPENAI_BASE_URL="http://127.0.0.1:9/v2", OPENAI_API_KEY="key-after")

    refresh()
    after = llm._openai()
    assert after is not before
    assert str(after.base_url).rstrip("/") == "http://127.0.0.1:9/v2"
    assert after.api_key == "key-after"
    assert after.max_retries == 1


def test_a_keyword_index_opened_without_a_path_follows_a_refreshed_state_db(env_file, tmp_path):
    from secondbrain.keyword_index import KeywordIndex

    env_file.loaded(SB_STATE_DB=str(tmp_path / "before" / "state.sqlite3"))
    index = KeywordIndex()
    index.upsert_chunks("notes", [{"source": "/notes/a.md", "chunk": 0, "document": "written before"}])
    env_file.edited(SB_STATE_DB=str(tmp_path / "after" / "state.sqlite3"))

    refresh()
    assert index.path == tmp_path / "after" / "keyword-index.sqlite3"
    assert index.count("notes") == 0
    index.upsert_chunks("notes", [{"source": "/notes/b.md", "chunk": 0, "document": "written after"}])
    assert index.count("notes") == 1


def test_a_tool_call_reads_a_changed_env_before_the_tool_runs(env_file):
    """The incident itself: a server started on old settings answers on the new ones."""
    from secondbrain import mcp_server

    env_file.loaded(OPENAI_BASE_URL="http://127.0.0.1:9/v1", CHAT_MODEL="chat-before")
    env_file.edited(OPENAI_BASE_URL="http://127.0.0.1:9/v2", CHAT_MODEL="chat-after")

    result = asyncio.run(mcp_server.mcp.call_tool("status", {}))
    blocks = result[0] if isinstance(result, tuple) else result
    status = json.loads(blocks[0].text)
    assert status["backend"] == "http://127.0.0.1:9/v2"
    assert status["chat_model"] == "chat-after"


class TestTheSuiteStillReadsNoEnvFile:
    """`conftest.py` sets python-dotenv's off switch so that the suite opens no .env.

    `load_dotenv()` honours that switch; `find_dotenv()` and `dotenv_values()`,
    which the re-read is built on, do not. Without asking the switch itself,
    `config` found the owner's real .env at import, read it to take its digest,
    and read it again on every tool call: a change to that file in the middle of
    a run would have put its values into the test process.
    """

    def test_the_off_switch_decides_whether_config_looks_for_a_file(self, tmp_path, monkeypatch):
        found = str(tmp_path / ".env")

        monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
        assert config._env_file(lambda: found) == ""

        monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
        assert config._env_file(lambda: found) == found

    def test_this_process_named_no_env_file_at_import_so_refresh_reads_nothing(self):
        # Where a real .env sits above this checkout, this is the canary; in CI,
        # which has none, the test above is. Compared as booleans, as in
        # test_env_pins.py: a failed comparison would print the path.
        named_none = config._ENV_FILE == ""
        assert named_none, "config found a .env to re-read although PYTHON_DOTENV_DISABLED is set"
        assert refresh() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "t", "yes", "Y", "0", "false", "no", "on", "", " 1"])
    def test_config_reads_the_switch_exactly_as_python_dotenv_does(self, value, tmp_path, monkeypatch):
        """The switch is python-dotenv's, so its reading of a value is the answer."""
        import dotenv

        canary = tmp_path / ".env"
        canary.write_text("SB_DOTENV_CANARY=loaded\n", encoding="utf-8")
        monkeypatch.delenv("SB_DOTENV_CANARY", raising=False)
        monkeypatch.setenv("PYTHON_DOTENV_DISABLED", value)
        try:
            dotenv.load_dotenv(canary)
            library_read_nothing = "SB_DOTENV_CANARY" not in os.environ
        finally:
            os.environ.pop("SB_DOTENV_CANARY", None)

        assert config._dotenv_disabled() is library_read_nothing
