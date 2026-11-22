"""The task ledger reports its true size and can be read past its newest 50.

`TaskStore.list` capped every answer at 50, newest first, and its callers
reported the page as the ledger: the MCP `list_tasks` said `count: 50`, `/tasks`
the same, and the console said "open tasks: 50". With 177 open tasks, every
agent saw 50, and an older task could not be found even by its title.

Every test here runs against a real SQLite ledger of 125 tasks, 120 of them
open, in a temporary file: more than two default pages, and a last page that is
not full.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from secondbrain.config import cfg
from secondbrain.tasks import TaskStore

OPEN_IDS = list(range(125, 5, -1))  # ids 125..6, newest first; 1..5 are done


@pytest.fixture
def ledger(tmp_path, monkeypatch) -> TaskStore:
    """125 tasks with ids 1..125; the first five are done, the other 120 open.

    `cfg.state_db` points at the temporary file, so a `TaskStore()` built with
    no arguments inside the MCP tool, the web API or the console reads it too.
    """
    path = tmp_path / "tasks.sqlite3"
    monkeypatch.setattr(cfg, "state_db", path)
    store = TaskStore(path=path)
    for n in range(1, 126):
        store.add(f"task {n}")
    for task_id in range(1, 6):
        store.complete(task_id)
    return store


class TestTaskStore:
    def test_count_is_the_true_total_for_each_status(self, ledger):
        assert ledger.count("open") == 120
        assert ledger.count("done") == 5
        assert ledger.count("all") == 125

    def test_pages_cover_every_open_task_once_newest_first(self, ledger):
        pages = [ledger.list(status="open", limit=50, offset=offset) for offset in (0, 50, 100, 150)]

        assert [len(page) for page in pages] == [50, 50, 20, 0]
        assert [task["id"] for page in pages for task in page] == OPEN_IDS

    def test_the_default_page_is_still_the_newest_50(self, ledger):
        assert [task["id"] for task in ledger.list()] == OPEN_IDS[:50]

    @pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": -1}, {"offset": -1}])
    def test_a_page_that_cannot_exist_is_refused(self, ledger, kwargs):
        # SQLite reads LIMIT -1 as "no limit"; refusing it keeps a page a page.
        with pytest.raises(ValueError):
            ledger.list(**kwargs)


class TestMcpListTasks:
    def test_reports_the_total_beside_the_page(self, ledger):
        from secondbrain import mcp_server as m

        out = asyncio.run(m.list_tasks())

        assert out["total"] == 120
        assert out["showing"] == 50
        assert out["offset"] == 0
        assert out["count"] == 50  # the old field: the page size, as before
        assert [task["id"] for task in out["tasks"]] == OPEN_IDS[:50]

    def test_an_agent_following_the_docstring_reaches_every_task(self, ledger):
        from secondbrain import mcp_server as m

        seen, offset = [], 0
        while True:
            page = asyncio.run(m.list_tasks(limit=50, offset=offset))
            seen += [task["id"] for task in page["tasks"]]
            if page["offset"] + page["showing"] >= page["total"]:
                break
            offset = page["offset"] + page["showing"]

        assert seen == OPEN_IDS

    def test_all_and_done_report_their_own_totals(self, ledger):
        from secondbrain import mcp_server as m

        assert asyncio.run(m.list_tasks(status="all"))["total"] == 125
        done = asyncio.run(m.list_tasks(status="done"))
        assert done["total"] == done["showing"] == 5

    @pytest.mark.parametrize(
        "kwargs",
        [{"status": "opne"}, {"limit": 0}, {"limit": 201}, {"offset": -1}],
    )
    def test_a_bad_request_is_an_error_not_an_empty_ledger(self, ledger, kwargs):
        from secondbrain import mcp_server as m

        with pytest.raises(ValueError):
            asyncio.run(m.list_tasks(**kwargs))

    def test_agents_are_offered_the_paging_arguments(self):
        from secondbrain import mcp_server as m

        tool = {t.name: t for t in asyncio.run(m.mcp.list_tools())}["list_tasks"]

        assert {"status", "limit", "offset"} <= set(tool.inputSchema["properties"])


class TestWebTasks:
    @pytest.fixture
    def owner(self, monkeypatch):
        from secondbrain import server

        monkeypatch.delenv("SB_WEB_REQUIRE_AUTH", raising=False)
        monkeypatch.setenv("SB_WEB_OWNER_TOKEN", "owner-test-token")
        server._SESSIONS.clear()
        server._rate_hits.clear()
        client = TestClient(server.app)
        return lambda query="": client.get(f"/tasks{query}", headers={"Authorization": "Bearer owner-test-token"})

    def test_reports_the_total_beside_the_page(self, ledger, owner):
        resp = owner()

        assert resp.status_code == 200
        body = resp.json()
        assert (body["total"], body["showing"], body["offset"], body["count"]) == (120, 50, 0, 50)

    def test_the_last_page_is_reachable(self, ledger, owner):
        body = owner("?limit=50&offset=100").json()

        assert body["total"] == 120
        assert [task["id"] for task in body["tasks"]] == OPEN_IDS[100:]

    @pytest.mark.parametrize("query", ["?limit=0", "?limit=201", "?offset=-1"])
    def test_a_page_that_cannot_exist_is_refused(self, ledger, owner, query):
        assert owner(query).status_code == 422


class TestConsoleAgent:
    def test_the_task_list_says_how_many_it_left_out(self, ledger):
        from secondbrain.agent import task_list_node

        lines = task_list_node({"argument": ""})["answer"].splitlines()

        assert lines[0] == "OPEN #125 task 125"
        assert len(lines) == 51
        assert lines[-1] == "Showing the newest 50 of 120."

    def test_a_short_list_has_no_such_line(self, tmp_path, monkeypatch):
        from secondbrain.agent import task_list_node

        monkeypatch.setattr(cfg, "state_db", tmp_path / "tasks.sqlite3")
        TaskStore().add("only task")

        assert task_list_node({"argument": ""})["answer"] == "OPEN #1 only task"

    def test_status_counts_every_open_task(self, ledger, monkeypatch):
        from secondbrain import agent

        def store_down(*_args, **_kwargs):
            raise RuntimeError("store unreachable in this test")

        # The chunk count needs a vector store and this test is about tasks.
        # agent.py builds its store inline, so there is no seam to pass one in.
        monkeypatch.setattr(agent, "Store", store_down)

        answer = agent.status_node({"argument": ""})["answer"]

        assert "open tasks: 120" in answer.splitlines()
