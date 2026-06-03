"""LangGraph workflow for one Artjeck console turn."""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .ask import ask as ask_memory
from .ingest import ingest_paths
from .memory import learn as learn_memory
from .store import Store
from .tasks import TaskStore

Action = Literal["ask", "learn", "ingest", "task_add", "task_list", "task_done", "status", "help", "exit"]


class AgentState(TypedDict, total=False):
    user_input: str
    action: Action
    argument: str
    answer: str
    sources: list[dict]
    invalid_citations: list[int]
    exit: bool


def _route_text(text: str) -> tuple[Action, str]:
    q = text.strip()
    if q in {"/exit", "/quit"}:
        return "exit", ""
    if q == "/help":
        return "help", ""
    if q == "/status":
        return "status", ""
    if q in {"/tasks", "tasks"}:
        return "task_list", "open"
    if q in {"/tasks all", "tasks all"}:
        return "task_list", "all"
    if q.startswith("/learn "):
        return "learn", q.removeprefix("/learn ").strip()
    if q.startswith("remember "):
        return "learn", q.removeprefix("remember ").strip()
    if q.startswith("/ingest "):
        return "ingest", q.removeprefix("/ingest ").strip()
    if q.startswith("/task "):
        return "task_add", q.removeprefix("/task ").strip()
    if q.startswith("task:"):
        return "task_add", q.removeprefix("task:").strip()
    if q.startswith("/done "):
        return "task_done", q.removeprefix("/done ").strip()
    return "ask", q


def route(state: AgentState) -> AgentState:
    action, argument = _route_text(state["user_input"])
    return {**state, "action": action, "argument": argument}


def ask_node(state: AgentState) -> AgentState:
    res = ask_memory(state["argument"])
    return {
        **state,
        "answer": res["answer"],
        "sources": res["sources"],
        "invalid_citations": res.get("invalid_citations", []),
    }


def learn_node(state: AgentState) -> AgentState:
    if not state["argument"]:
        return {**state, "answer": "Provide a fact after /learn."}
    res = learn_memory(state["argument"], source="agent")
    return {**state, "answer": f"Learned {res['chunks']} chunk(s): {res['path']}"}


def ingest_node(state: AgentState) -> AgentState:
    path = state["argument"]
    if not path:
        return {**state, "answer": "Provide a file or folder path after /ingest."}
    files = chunks = 0
    for _, n in ingest_paths(path):
        files += 1
        chunks += n
    total = Store().count()
    return {
        **state,
        "answer": f"Ingested {files} file(s), {chunks} chunk(s); collection has {total} chunk(s).",
    }


def task_add_node(state: AgentState) -> AgentState:
    if not state["argument"]:
        return {**state, "answer": "Provide a task title after /task."}
    task = TaskStore().add(state["argument"])
    return {**state, "answer": f"Added task #{task['id']}: {task['title']}"}


def task_list_node(state: AgentState) -> AgentState:
    status = state.get("argument") or "open"
    tasks = TaskStore().list(status=status)
    if not tasks:
        label = "tasks" if status == "all" else f"{status} tasks"
        return {**state, "answer": f"No {label}."}
    lines = []
    for task in tasks:
        label = "DONE" if task["status"] == "done" else "OPEN"
        lines.append(f"{label} #{task['id']} {task['title']}")
    return {**state, "answer": "\n".join(lines)}


def task_done_node(state: AgentState) -> AgentState:
    try:
        task_id = int(state["argument"])
        task = TaskStore().complete(task_id)
    except ValueError:
        return {**state, "answer": "Use /done <task-id>."}
    except KeyError as exc:
        return {**state, "answer": str(exc)}
    return {**state, "answer": f"Completed task #{task['id']}: {task['title']}"}


def status_node(state: AgentState) -> AgentState:
    try:
        chunks = str(Store().count())
    except RuntimeError as exc:
        chunks = f"ERROR: {exc}"
    open_tasks = len(TaskStore().list(status="open"))
    answer = f"chunks: {chunks}\nopen tasks: {open_tasks}"
    return {**state, "answer": answer}


def help_node(state: AgentState) -> AgentState:
    answer = "\n".join(
        [
            "Ask normally: What do you know about my AI Lab?",
            "Teach memory: /learn <fact>",
            "Teach memory: remember <fact>",
            "Ingest files: /ingest <file-or-folder>",
            "Add task: /task <task>",
            "List tasks: /tasks",
            "Complete task: /done <task-id>",
            "Show status: /status",
            "Exit: /exit",
        ]
    )
    return {**state, "answer": answer}


def exit_node(state: AgentState) -> AgentState:
    return {**state, "answer": "bye", "exit": True}


def choose_node(state: AgentState) -> str:
    return {
        "ask": "ask",
        "learn": "learn",
        "ingest": "ingest",
        "task_add": "task_add",
        "task_list": "task_list",
        "task_done": "task_done",
        "status": "status",
        "help": "help",
        "exit": "exit",
    }[state["action"]]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("ask", ask_node)
    graph.add_node("learn", learn_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("task_add", task_add_node)
    graph.add_node("task_list", task_list_node)
    graph.add_node("task_done", task_done_node)
    graph.add_node("status", status_node)
    graph.add_node("help", help_node)
    graph.add_node("exit", exit_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", choose_node)
    for node in ["ask", "learn", "ingest", "task_add", "task_list", "task_done", "status", "help", "exit"]:
        graph.add_edge(node, END)
    return graph.compile()


_GRAPH = build_graph()


def run_turn(user_input: str) -> AgentState:
    return _GRAPH.invoke({"user_input": user_input})
