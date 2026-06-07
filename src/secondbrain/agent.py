"""LangGraph workflow for one Artjeck console turn."""
from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .ask import ask as ask_memory
from .ingest import ingest_paths
from .memory import learn as learn_memory
from .store import Store
from .system_info import format_system_status
from .tasks import TaskStore
from .weather import format_weather

Action = Literal[
    "ask",
    "learn",
    "ingest",
    "task_add",
    "task_list",
    "task_done",
    "status",
    "system_status",
    "weather",
    "chat",
    "help",
    "exit",
]


class AgentState(TypedDict, total=False):
    user_input: str
    action: Action
    argument: str
    answer: str
    sources: list[dict]
    invalid_citations: list[int]
    exit: bool


def _weather_location(text: str) -> str | None:
    q = text.strip()
    if re.fullmatch(r"/weather\b.*", q, flags=re.IGNORECASE):
        return re.sub(r"^/weather\b", "", q, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"(?:what(?:'s|s| is) )?(?:the )?(?:current )?weather", q.rstrip("?!.,;:"), flags=re.IGNORECASE):
        return ""
    match = re.search(r"\bweather\s+(?:in|for)\s+(.+?)\s*[?!.,;:]*$", q, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _route_text(text: str) -> tuple[Action, str]:
    q = text.strip()
    q_lower = q.lower()
    q_plain = q_lower.rstrip("?!.,;:").strip()
    if q in {"/exit", "/quit"}:
        return "exit", ""
    if q == "/help":
        return "help", ""
    if q == "/status":
        return "status", ""
    if q_lower in {"/system", "system status", "system info", "machine status"}:
        return "system_status", "all"
    if q_lower in {"/system memory", "system memory"} or any(
        phrase in q_lower
        for phrase in ("check system memory", "memory usage", "available memory", "free memory", "ram usage")
    ):
        return "system_status", "memory"
    if q_lower in {"/system storage", "storage space"} or any(
        phrase in q_lower for phrase in ("disk space", "free space", "storage space")
    ):
        return "system_status", "storage"
    weather_location = _weather_location(q)
    if weather_location is not None:
        return "weather", weather_location
    if q_plain in {
        "hi",
        "hello",
        "hey",
        "how are you",
        "how are you doing",
        "how's it going",
        "hows it going",
        "are you ok",
        "are you ready",
        "you good",
    }:
        return "chat", q_plain
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
    if q == "/task":
        return "task_add", ""
    if q.startswith("/task "):
        return "task_add", q.removeprefix("/task ").strip()
    if q.startswith("task:"):
        return "task_add", q.removeprefix("task:").strip()
    if q == "/done":
        return "task_done", ""
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


def system_status_node(state: AgentState) -> AgentState:
    return {**state, "answer": format_system_status(state.get("argument", "all"))}


def weather_node(state: AgentState) -> AgentState:
    return {**state, "answer": format_weather(state.get("argument", ""))}


def chat_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": "I'm here and ready. Ask me about your notes, teach me with /learn, or use /status if you want the brain stats.",
    }


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
            "Show system stats: /system, /system memory, /system storage",
            "Show weather: /weather <location>",
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
        "system_status": "system_status",
        "weather": "weather",
        "chat": "chat",
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
    graph.add_node("system_status", system_status_node)
    graph.add_node("weather", weather_node)
    graph.add_node("chat", chat_node)
    graph.add_node("help", help_node)
    graph.add_node("exit", exit_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", choose_node)
    for node in [
        "ask",
        "learn",
        "ingest",
        "task_add",
        "task_list",
        "task_done",
        "status",
        "system_status",
        "weather",
        "chat",
        "help",
        "exit",
    ]:
        graph.add_edge(node, END)
    return graph.compile()


_GRAPH = build_graph()


def run_turn(user_input: str) -> AgentState:
    return _GRAPH.invoke({"user_input": user_input})
