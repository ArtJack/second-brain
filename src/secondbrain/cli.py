"""`sb` — the command-line interface to your second brain."""
from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from .agent import run_turn
from .ask import ask as ask_fn
from .config import cfg
from .ingest import ingest_paths
from .memory import learn as learn_memory
from .store import Store

app = typer.Typer(add_completion=False, help="second-brain — ingest your stuff, ask, get cited answers.")
console = Console()


AGENT_NAME = "Artjeck"


def _print_answer(res: dict, show_distance: bool = True) -> None:
    console.print(Panel(res["answer"], title="answer", border_style="cyan"))
    for s in res["sources"]:
        if show_distance:
            console.print(f"  [cyan][{s['n']}][/] {s['source']} [dim](dist {s['distance']:.3f})[/]")
        else:
            console.print(f"  [cyan][{s['n']}][/] {s['source']}", style="dim")


def _ingest_path(path: str, reset: bool = False) -> dict:
    files = chunks = 0
    for f, n in ingest_paths(path, reset=reset):
        files += 1
        chunks += n
        console.print(f"  [green]+[/] {f} [dim]({n} chunks)[/]")
    return {"files": files, "chunks": chunks, "total": Store().count()}


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or folder to ingest"),
    reset: bool = typer.Option(False, "--reset", help="Wipe the collection before ingesting"),
):
    """Ingest a file or folder into your second brain."""
    try:
        with console.status("[cyan]embedding…[/]"):
            res = _ingest_path(path, reset=reset)
    except RuntimeError as exc:
        console.print(f"[red]ingest failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        Panel(
            f"Ingested [bold]{res['files']}[/] files / [bold]{res['chunks']}[/] chunks. "
            f"Collection now holds [bold]{res['total']}[/] chunks.",
            title="ingest complete",
            border_style="green",
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    k: int = typer.Option(cfg.top_k, "--k", help="Chunks to retrieve"),
):
    """Ask a question and get an answer cited to your sources."""
    try:
        res = ask_fn(question, k=k)
    except RuntimeError as exc:
        console.print(f"[red]ask failed:[/] {exc}")
        raise typer.Exit(1) from exc
    _print_answer(res)


@app.command()
def learn(
    text: list[str] = typer.Argument(..., help="Fact, preference, note, or decision to remember"),
    source: str = typer.Option("user", "--source", help="Who/what taught this memory"),
):
    """Teach second-brain a durable memory and ingest it immediately."""
    memory_text = " ".join(text).strip()
    try:
        res = learn_memory(memory_text, source=source)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]learn failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        Panel(
            f"Learned [bold]{res['chunks']}[/] chunk(s).\nsource : {res['path']}",
            title="memory saved",
            border_style="green",
        )
    )


@app.command()
def chat():
    """Interactive question loop over your second brain."""
    console.print("[dim]Ask away — Ctrl-C to quit.[/]")
    while True:
        try:
            q = console.input("[bold cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye 👋")
            break
        if not q:
            continue
        try:
            res = ask_fn(q)
        except RuntimeError as exc:
            console.print(f"[red]ask failed:[/] {exc}")
            continue
        _print_answer(res, show_distance=False)


@app.command()
def agent():
    """Interactive self-learning agent loop.

    Normal text asks a question. `/learn <fact>` writes a durable memory and ingests it.
    """
    console.print(
        f"[bold cyan]{AGENT_NAME}[/] [dim]is ready. Ask questions, use /learn, /ingest, /task, /tasks, /done, /status, /help, or /exit.[/]"
    )
    while True:
        try:
            q = console.input(f"[bold cyan]{AGENT_NAME.lower()} ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break
        if not q:
            continue
        try:
            res = run_turn(q)
        except (RuntimeError, ValueError) as exc:
            console.print(f"[red]turn failed:[/] {exc}")
            continue
        if res.get("action") == "ask":
            _print_answer({"answer": res.get("answer", ""), "sources": res.get("sources", [])}, show_distance=False)
        else:
            title = f"{AGENT_NAME} commands" if res.get("action") == "help" else AGENT_NAME
            console.print(Panel(res.get("answer", ""), title=title, border_style="cyan"))
        if res.get("exit"):
            break


@app.command()
def status():
    """Show the active backend, models, and how much is stored."""
    try:
        chunks = str(Store().count())
    except RuntimeError as exc:
        chunks = f"ERROR: {exc}"
    console.print(
        Panel(
            f"backend : {cfg.base_url}\n"
            f"embed   : {cfg.embed_model}\n"
            f"chat    : {cfg.chat_model}\n"
            f"store   : {cfg.store_backend}\n"
            f"chroma  : {cfg.persist_dir}\n"
            f"qdrant  : {cfg.qdrant_url}\n"
            f"memory  : {cfg.memory_dir}\n"
            f"state   : {cfg.state_db}\n"
            f"chunks  : {chunks}",
            title="second-brain status",
        )
    )


if __name__ == "__main__":
    app()


def artjeck_main() -> None:
    """Console entrypoint: `artjeck` starts the conversational agent directly."""
    if len(sys.argv) == 1:
        agent()
    else:
        app()
