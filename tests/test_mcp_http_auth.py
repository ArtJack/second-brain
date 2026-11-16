"""The HTTP MCP transport fails closed, and its token check is constant-time.

With no `SB_MCP_TOKEN` the server logged one warning and served anyway, on
whatever address it was given: one renamed line in `.env` would have opened all
nine tools, `forget` and `ingest` included, to every device on the tailnet. The
token was compared with `!=`, where the web API in this repo uses
`compare_digest`.

`http_plan` is the decision and takes the environment as a plain mapping, so
most of this file needs no server and no patching.
"""

from __future__ import annotations

import asyncio

import pytest

from secondbrain.mcp_server import HttpPlan, MCPTokenRequired, _BearerAuthASGI, http_plan

TOKEN = "mcp-test-token"


class TestStartup:
    def test_no_token_refuses_to_start(self):
        with pytest.raises(MCPTokenRequired, match="SB_MCP_TOKEN"):
            http_plan({})

    def test_a_blank_token_is_no_token(self):
        with pytest.raises(MCPTokenRequired):
            http_plan({"SB_MCP_TOKEN": "   ", "SB_MCP_HOST": "127.0.0.1"})

    def test_a_token_serves_on_the_tailnet(self):
        plan = http_plan({"SB_MCP_TOKEN": TOKEN, "SB_MCP_HOST": "100.64.0.7", "SB_MCP_PORT": "9000"})

        assert plan == HttpPlan(host="100.64.0.7", port=9000, token=TOKEN)

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"])
    def test_the_override_serves_without_a_token_on_loopback(self, host):
        plan = http_plan({"SB_MCP_ALLOW_NO_TOKEN": "1", "SB_MCP_HOST": host})

        assert plan.token is None
        assert plan.host == host

    def test_the_override_works_on_the_default_host(self):
        assert http_plan({"SB_MCP_ALLOW_NO_TOKEN": "true"}).token is None

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "100.64.0.7", "192.168.1.20", "mini.example.ts.net"])
    def test_the_override_is_refused_off_loopback(self, host):
        with pytest.raises(MCPTokenRequired, match="loopback"):
            http_plan({"SB_MCP_ALLOW_NO_TOKEN": "1", "SB_MCP_HOST": host})

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_only_an_explicit_override_counts(self, value):
        with pytest.raises(MCPTokenRequired):
            http_plan({"SB_MCP_ALLOW_NO_TOKEN": value, "SB_MCP_HOST": "127.0.0.1"})


class TestMain:
    def test_the_http_transport_exits_before_serving_without_a_token(self, monkeypatch):
        import uvicorn

        from secondbrain import mcp_server

        def must_not_serve(*_args, **_kwargs):
            raise AssertionError("uvicorn.run was reached without a token")

        monkeypatch.setattr(uvicorn, "run", must_not_serve)
        monkeypatch.setenv("SB_MCP_TRANSPORT", "http")
        monkeypatch.setenv("SB_MCP_HOST", "100.64.0.7")
        monkeypatch.delenv("SB_MCP_TOKEN", raising=False)

        with pytest.raises(SystemExit) as exited:
            mcp_server.main()

        assert exited.value.code == 2

    def test_with_a_token_it_serves_behind_the_gate(self, monkeypatch):
        import uvicorn

        from secondbrain import mcp_server

        served = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: served.update(app=app, **kwargs))
        monkeypatch.setenv("SB_MCP_TRANSPORT", "http")
        monkeypatch.setenv("SB_MCP_TOKEN", TOKEN)

        mcp_server.main()

        assert isinstance(served["app"], _BearerAuthASGI)
        assert (served["host"], served["port"]) == ("127.0.0.1", 8848)


class _Recorder:
    """The app behind the gate: records whether a request got through."""

    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        self.calls.append(scope["type"])


def _call(gate, scope_type: str, authorization: bytes | None = None) -> list[dict]:
    headers = [] if authorization is None else [(b"authorization", authorization)]
    sent: list[dict] = []

    async def receive():
        return {"type": f"{scope_type}.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(gate({"type": scope_type, "headers": headers}, receive, send))
    return sent


class TestGate:
    def test_the_right_token_reaches_the_app(self):
        app = _Recorder()

        assert _call(_BearerAuthASGI(app, TOKEN), "http", f"Bearer {TOKEN}".encode()) == []
        assert app.calls == ["http"]

    @pytest.mark.parametrize(
        "authorization",
        [
            None,
            b"",
            f"Bearer {TOKEN[:-1]}x".encode(),  # same length, last byte differs
            f"Bearer {TOKEN}extra".encode(),
            TOKEN.encode(),
            b"Bearer \xff\xfe",  # not UTF-8: refused, not a server error
        ],
    )
    def test_anything_else_is_401_and_never_reaches_the_app(self, authorization):
        app = _Recorder()

        sent = _call(_BearerAuthASGI(app, TOKEN), "http", authorization)

        assert sent[0]["status"] == 401
        assert app.calls == []

    def test_a_websocket_handshake_without_the_token_is_refused(self):
        app = _Recorder()

        sent = _call(_BearerAuthASGI(app, TOKEN), "websocket")

        assert sent == [{"type": "websocket.close", "code": 1008}]
        assert app.calls == []

    def test_lifespan_events_pass_without_a_token(self):
        # The app's lifespan starts the session manager; gating it would stop
        # the server from starting at all.
        app = _Recorder()

        _call(_BearerAuthASGI(app, TOKEN), "lifespan")

        assert app.calls == ["lifespan"]

    @pytest.mark.parametrize("token", ["", "  "])
    def test_the_gate_will_not_guard_with_an_empty_token(self, token):
        # "Bearer " would otherwise be the password.
        with pytest.raises(ValueError):
            _BearerAuthASGI(_Recorder(), token)
