"""`sb` — the command-line interface to your second brain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import run_turn
from .ask import ask as ask_fn
from .config import cfg
from .evals import DEFAULT_BENCHMARK, load_benchmark, resolve_corpus_paths, run_benchmark, select_cases
from .ingest import ingest_paths
from .intake import PRIVATE_EVAL_DIR, build_private_artifacts, reset_session, run_intake
from .memory import learn as learn_memory
from .store import Store
from .tracing import write_trace_export

app = typer.Typer(add_completion=False, help="second-brain — ingest your stuff, ask, get cited answers.")
console = Console()


AGENT_NAME = "Artjeck"


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _decimal(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_answer(res: dict, show_distance: bool = True) -> None:
    console.print(Panel(res["answer"], title="answer", border_style="cyan"))
    for s in res["sources"]:
        if show_distance:
            console.print(f"  [cyan][{s['n']}][/] {s['source']} [dim](dist {s['distance']:.3f})[/]")
        else:
            console.print(f"  [cyan][{s['n']}][/] {s['source']}", style="dim")
    invalid = res.get("invalid_citations") or []
    if invalid:
        refs = ", ".join(f"[{n}]" for n in invalid)
        console.print(f"  [yellow]⚠ ungrounded citation(s) {refs}: no matching source — answer may be unreliable[/]")


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


@app.command("eval")
def evaluate(
    benchmark: Path = typer.Argument(DEFAULT_BENCHMARK, help="Benchmark JSON file"),
    k: int = typer.Option(cfg.top_k, "--k", help="Chunks to retrieve per case"),
    answers: bool = typer.Option(False, "--answers", help="Also call the chat model and run answer heuristics"),
    ingest_corpus: bool = typer.Option(False, "--ingest-corpus", help="Ingest the benchmark's configured corpus first"),
    tag: list[str] | None = typer.Option(None, "--tag", help="Run cases with this tag; repeat for more tags"),
    trace_output: Path | None = typer.Option(None, "--trace-output", help="Write standalone trace/span/trajectory JSON"),
    json_output: bool = typer.Option(False, "--json", help="Print the full JSON report"),
):
    """Run the local retrieval benchmark and optional grounded-answer checks."""
    try:
        loaded = load_benchmark(benchmark)
        corpus_ingest = None
        if ingest_corpus:
            corpus_paths = resolve_corpus_paths(loaded, benchmark)
            if not corpus_paths:
                raise ValueError("Benchmark does not configure a corpus.")
            corpus_collection = loaded.get("corpus_collection")
            if corpus_collection and cfg.collection != corpus_collection:
                raise ValueError(
                    f"Benchmark corpus requires SB_COLLECTION={corpus_collection}; "
                    f"current collection is {cfg.collection!r}."
                )
            files = chunks = 0
            for corpus_path in corpus_paths:
                if not corpus_path.exists():
                    raise ValueError(f"Corpus path not found: {corpus_path}")
                for _file, n in ingest_paths(corpus_path):
                    files += 1
                    chunks += n
            corpus_ingest = {"paths": [str(path) for path in corpus_paths], "files": files, "chunks": chunks}
        report = run_benchmark(select_cases(loaded, tag), top_k=k, include_answers=answers)
        if corpus_ingest:
            report["corpus_ingest"] = corpus_ingest
        if trace_output:
            report["trace_export"] = str(
                write_trace_export(trace_output, benchmark=report["benchmark"], traces=report["traces"])
            )
    except Exception as exc:
        console.print(f"[red]eval failed:[/] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        summary = report["summary"]
        if corpus_ingest:
            console.print(
                f"ingested benchmark corpus: [bold]{corpus_ingest['files']}[/] file(s), "
                f"[bold]{corpus_ingest['chunks']}[/] chunk(s)"
            )
        table = Table(title=f"evaluation: {report['benchmark']}")
        table.add_column("case")
        table.add_column("retrieval", justify="center")
        table.add_column("rank", justify="right")
        if answers:
            table.add_column("answer", justify="center")
        table.add_column("query")
        for case in report["cases"]:
            retrieval = case["retrieval"]
            row = [
                case["id"],
                "[dim]SKIP[/]"
                if not retrieval["scored"]
                else "[green]PASS[/]"
                if retrieval["passed"]
                else "[red]FAIL[/]",
                str(retrieval["first_relevant_rank"] or "-"),
            ]
            if answers:
                row.append("[green]PASS[/]" if case["answer"]["passed"] else "[red]FAIL[/]")
            row.append(case["query"])
            table.add_row(*row)
        console.print(table)
        console.print(
            f"retrieval [bold]{summary['retrieval_passed']}/{summary['retrieval_cases']}[/]  "
            f"hit-rate [bold]{_percent(summary['retrieval_hit_rate'])}[/]  "
            f"source recall [bold]{_percent(summary['mean_source_recall'])}[/]  "
            f"MRR [bold]{_decimal(summary['mrr'])}[/]  "
            f"duration [bold]{report['duration_ms']:.1f} ms[/]"
        )
        if answers:
            console.print(
                f"answer rubric [bold]{summary['answer_passed']}/{summary['cases']}[/] passed  "
                f"score [bold]{summary['answer_rubric_score']:.1%}[/]  "
                f"abstention [bold]{summary['abstention_passed']}/{summary['abstention_cases']}[/]"
            )
        trace_summary = report["trace_summary"]
        console.print(
            f"traces [bold]{trace_summary['traces']}[/]  "
            f"spans [bold]{trace_summary['spans']}[/]  "
            f"trajectory steps [bold]{trace_summary['trajectory_steps']}[/]"
        )
        if report.get("trace_export"):
            console.print(f"trace export [bold]{report['trace_export']}[/]")
        for case in report["cases"]:
            if not case["retrieval"]["passed"]:
                missing = set(case["retrieval"]["expected_sources"]) - set(
                    case["retrieval"]["matched_expected_sources"]
                )
                console.print(f"[red]FAIL[/] {case['id']}: missing source(s): {', '.join(sorted(missing))}")
            if answers and not case["answer"]["passed"]:
                checks = case["answer"]["checks"]
                failed = ", ".join(name for name, passed in checks.items() if not passed)
                console.print(f"[red]FAIL[/] {case['id']}: answer checks: {failed}")
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("eval-intake")
def eval_intake(
    output_dir: Path = typer.Option(PRIVATE_EVAL_DIR, "--output-dir", help="Private git-ignored intake directory"),
    reset: bool = typer.Option(False, "--reset", help="Start the private questionnaire over"),
    build_only: bool = typer.Option(False, "--build-only", help="Rebuild artifacts without asking questions"),
):
    """Collect or rebuild a private 100-question benchmark in the terminal."""
    if reset:
        reset_session(output_dir)
        console.print(f"[yellow]reset private intake:[/] {output_dir}")
    if build_only:
        summary = build_private_artifacts(output_dir)
        console.print(
            f"private benchmark: [bold]{summary['benchmark_cases']}[/] case(s), "
            f"[bold]{summary['remaining']}[/] prompt(s) remaining\n{summary['benchmark']}"
        )
        return
    run_intake(
        output_dir,
        input_fn=lambda prompt: console.input(f"[bold cyan]{prompt}[/]"),
        output_fn=console.print,
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
