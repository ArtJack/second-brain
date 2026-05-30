"""`sb` — the command-line interface to your second brain."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .ask import ask as ask_fn
from .config import cfg
from .ingest import ingest_paths
from .store import Store

app = typer.Typer(add_completion=False, help="second-brain — ingest your stuff, ask, get cited answers.")
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or folder to ingest"),
    reset: bool = typer.Option(False, "--reset", help="Wipe the collection before ingesting"),
):
    """Ingest a file or folder into your second brain."""
    files = chunks = 0
    with console.status("[cyan]embedding…[/]"):
        for f, n in ingest_paths(path, reset=reset):
            files += 1
            chunks += n
            console.print(f"  [green]+[/] {f} [dim]({n} chunks)[/]")
    console.print(
        Panel(
            f"Ingested [bold]{files}[/] files / [bold]{chunks}[/] chunks. "
            f"Collection now holds [bold]{Store().count()}[/] chunks.",
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
    res = ask_fn(question, k=k)
    console.print(Panel(res["answer"], title="answer", border_style="cyan"))
    for s in res["sources"]:
        console.print(f"  [cyan][{s['n']}][/] {s['source']} [dim](dist {s['distance']:.3f})[/]")


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
        res = ask_fn(q)
        console.print(Panel(res["answer"], border_style="cyan"))
        for s in res["sources"]:
            console.print(f"  [cyan][{s['n']}][/] {s['source']}", style="dim")


@app.command()
def status():
    """Show the active backend, models, and how much is stored."""
    console.print(
        Panel(
            f"backend : {cfg.base_url}\n"
            f"embed   : {cfg.embed_model}\n"
            f"chat    : {cfg.chat_model}\n"
            f"store   : {cfg.persist_dir}\n"
            f"chunks  : {Store().count()}",
            title="second-brain status",
        )
    )


if __name__ == "__main__":
    app()
