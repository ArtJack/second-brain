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
from .ask import recall as recall_fn
from .browser_check import DEFAULT_BROWSER_DIR, capture_url
from .config import cfg
from .evals import DEFAULT_BENCHMARK, load_benchmark, resolve_corpus_paths, run_benchmark, select_cases
from .ingest import ingest_paths
from .intake import PRIVATE_EVAL_DIR, build_private_artifacts, reset_session, run_intake
from .health import run_health
from .memory import learn as learn_memory
from .morning import run_morning
from .overnight import run_overnight
from .project_context import DEFAULT_OUTPUT_DIR, write_project_context_notes
from .store import Store
from .task_sync import TaskCandidate, sync_tasks
from .tracing import write_trace_export
from .web_check import DEFAULT_WEB_DIR, check_url

app = typer.Typer(add_completion=False, help="second-brain — ingest your stuff, ask, get cited answers.")
console = Console()


AGENT_NAME = "Artjeck"
_COLLECTION: str | None = None


@app.callback()
def main_options(
    collection: str | None = typer.Option(
        None,
        "--collection",
        help="Override the vector collection for ask/recall/ingest/learn/status.",
    ),
) -> None:
    """Shared CLI options."""
    global _COLLECTION
    _COLLECTION = collection


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _decimal(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_answer(res: dict, show_distance: bool = True, show_sources: bool = True) -> None:
    console.print(Panel(res["answer"], title="answer", border_style="cyan"))
    if show_sources:
        for s in res["sources"]:
            if show_distance:
                console.print(f"  [cyan][{s['n']}][/] {s['source']} [dim](dist {s['distance']:.3f})[/]")
            else:
                console.print(f"  [cyan][{s['n']}][/] {s['source']}", style="dim")
    invalid = res.get("invalid_citations") or []
    if invalid:
        refs = ", ".join(f"[{n}]" for n in invalid)
        console.print(f"  [yellow]⚠ ungrounded citation(s) {refs}: no matching source — answer may be unreliable[/]")


def _ingest_path(path: str, reset: bool = False, collection: str | None = None) -> dict:
    files = chunks = 0
    selected_collection = collection if collection is not None else _COLLECTION
    for f, n in ingest_paths(path, reset=reset, collection=selected_collection):
        files += 1
        chunks += n
        console.print(f"  [green]+[/] {f} [dim]({n} chunks)[/]")
    return {"files": files, "chunks": chunks, "total": Store(collection=selected_collection).count()}


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
        res = ask_fn(question, k=k, collection=_COLLECTION)
    except RuntimeError as exc:
        console.print(f"[red]ask failed:[/] {exc}")
        raise typer.Exit(1) from exc
    _print_answer(res)


@app.command()
def recall(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(0, "--top-k", "--k", help="Chunks to retrieve (0 = configured default)"),
):
    """Retrieve raw matching chunks without calling the chat model."""
    try:
        res = recall_fn(query, top_k=top_k, collection=_COLLECTION)
    except RuntimeError as exc:
        console.print(f"[red]recall failed:[/] {exc}")
        raise typer.Exit(1) from exc
    table = Table(title=f"recall: {res['count']} hit(s)")
    table.add_column("source")
    table.add_column("distance", justify="right")
    table.add_column("text")
    for hit in res["hits"]:
        table.add_row(hit["source"], f"{hit['distance']:.4f}", hit["text"][:240])
    console.print(table)


@app.command()
def learn(
    text: list[str] = typer.Argument(..., help="Fact, preference, note, or decision to remember"),
    source: str = typer.Option("user", "--source", help="Who/what taught this memory"),
):
    """Teach second-brain a durable memory and ingest it immediately."""
    memory_text = " ".join(text).strip()
    try:
        res = learn_memory(memory_text, source=source, collection=_COLLECTION)
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
            res = ask_fn(q, collection=_COLLECTION)
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
            _print_answer(
                {"answer": res.get("answer", ""), "sources": res.get("sources", [])},
                show_distance=False,
                show_sources=False,
            )
        else:
            title = f"{AGENT_NAME} commands" if res.get("action") == "help" else AGENT_NAME
            console.print(Panel(res.get("answer", ""), title=title, border_style="cyan"))
        if res.get("exit"):
            break


@app.command()
def status():
    """Show the active backend, models, and how much is stored."""
    try:
        chunks = str(Store(collection=_COLLECTION).count())
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


@app.command()
def overnight(
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan and report without ingesting or updating file state"),
):
    """Run the safe overnight scan: ingest changed files and write a morning report."""
    try:
        res = run_overnight(dry_run=dry_run)
    except Exception as exc:
        console.print(f"[red]overnight failed:[/] {exc}")
        raise typer.Exit(1) from exc
    stats = res["stats"]
    chunks = "-" if res["total_chunks"] is None else str(res["total_chunks"])
    console.print(
        Panel(
            f"scanned  : {stats['scanned']}\n"
            f"changed  : {stats['changed']}\n"
            f"ingested : {stats['ingested']}\n"
            f"failed   : {stats['failed']}\n"
            f"chunks   : {chunks}\n"
            f"config   : {res['config']}\n"
            f"report   : {res['report']}",
            title="overnight complete",
            border_style="green" if stats["failed"] == 0 else "yellow",
        )
    )


@app.command()
def morning(
    rag: bool = typer.Option(False, "--rag", help="Include cited RAG summary from the indexed brain"),
    k: int = typer.Option(cfg.top_k, "--k", help="Chunks to retrieve for each briefing question"),
):
    """Create a morning briefing from overnight activity, tasks, and cited brain answers."""
    try:
        res = run_morning(include_rag=rag, k=k)
    except Exception as exc:
        console.print(f"[red]morning failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(Panel(res["markdown"], title="morning briefing", border_style="cyan"))
    console.print(f"[dim]saved: {res['path']}[/]")


@app.command("project-context")
def project_context(
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", help="Where generated project notes are written"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="Ingest generated notes into the active vector store"),
    limit: int = typer.Option(20, "--limit", help="Maximum project roots to summarize"),
):
    """Generate source-backed project notes for better morning briefings and RAG."""
    try:
        written = write_project_context_notes(output_dir=output_dir, limit=limit)
        ingest_res = None
        if ingest and written:
            ingest_res = _ingest_path(str(output_dir))
    except Exception as exc:
        console.print(f"[red]project-context failed:[/] {exc}")
        raise typer.Exit(1) from exc
    lines = [f"wrote {len(written)} project context note(s)", f"output: {output_dir}"]
    if ingest_res:
        lines.append(
            f"ingested {ingest_res['files']} file(s), {ingest_res['chunks']} chunk(s); "
            f"collection chunks: {ingest_res['total']}"
        )
    for path in written:
        lines.append(f"- {path}")
    console.print(Panel("\n".join(lines), title="project context", border_style="green"))


@app.command("task-sync")
def task_sync(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show tasks that would be added without writing"),
):
    """Import explicit project follow-ups into the task store with dedupe."""
    try:
        res = sync_tasks(dry_run=dry_run)
    except Exception as exc:
        console.print(f"[red]task-sync failed:[/] {exc}")
        raise typer.Exit(1) from exc
    added = res["added"]
    skipped = res["skipped"]
    lines = [
        f"mode: {'dry run' if dry_run else 'write'}",
        f"added: {len(added)}",
        f"skipped existing: {len(skipped)}",
    ]
    if added:
        lines.append("")
        lines.append("Added:")
        for item in added:
            if isinstance(item, TaskCandidate):
                lines.append(f"- {item.title} ({item.source})")
            else:
                lines.append(f"- #{item['id']}: {item['title']}")
    if skipped:
        lines.append("")
        lines.append("Skipped:")
        for item in skipped:
            lines.append(f"- {item.title}")
    console.print(Panel("\n".join(lines), title="task sync", border_style="green"))


@app.command("web-check")
def web_check(
    url: str = typer.Argument(..., help="HTTP(S) URL to fetch and save as a source-backed note"),
    output_dir: Path = typer.Option(DEFAULT_WEB_DIR, "--output-dir", help="Where web notes are written"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="Ingest the saved web note into the active vector store"),
    timeout: int = typer.Option(20, "--timeout", help="Fetch timeout in seconds"),
):
    """Fetch a web page, extract readable text, save a note, and optionally ingest it."""
    try:
        res = check_url(url, output_dir=output_dir, timeout_s=timeout)
        ingest_res = None
        if ingest:
            ingest_res = _ingest_path(str(res["path"]))
    except Exception as exc:
        console.print(f"[red]web-check failed:[/] {exc}")
        raise typer.Exit(1) from exc
    page = res["page"]
    lines = [
        f"title   : {page.title}",
        f"url     : {page.final_url}",
        f"status  : {page.status}",
        f"saved   : {res['path']}",
    ]
    if ingest_res:
        lines.append(
            f"ingest  : {ingest_res['files']} file(s), {ingest_res['chunks']} chunk(s); "
            f"collection chunks: {ingest_res['total']}"
        )
    console.print(Panel("\n".join(lines), title="web check", border_style="green"))


@app.command("browser-check")
def browser_check(
    url: str = typer.Argument(..., help="HTTP(S) URL to open in a real browser and save as a source-backed note"),
    output_dir: Path = typer.Option(DEFAULT_BROWSER_DIR, "--output-dir", help="Where browser notes and screenshots are written"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="Ingest the saved browser note into the active vector store"),
    screenshot: bool = typer.Option(True, "--screenshot/--no-screenshot", help="Save a full-page screenshot"),
    timeout: int = typer.Option(30, "--timeout", help="Navigation timeout in seconds"),
    wait_until: str = typer.Option("networkidle", "--wait-until", help="Playwright wait state: load, domcontentloaded, networkidle, commit"),
):
    """Open a page with Playwright, extract rendered text, save a note, and optionally ingest it."""
    try:
        res = capture_url(
            url,
            output_dir=output_dir,
            timeout_ms=timeout * 1000,
            wait_until=wait_until,
            screenshot=screenshot,
        )
        ingest_res = None
        if ingest:
            ingest_res = _ingest_path(str(res["path"]))
    except Exception as exc:
        console.print(f"[red]browser-check failed:[/] {exc}")
        raise typer.Exit(1) from exc
    page = res["page"]
    lines = [
        f"title     : {page.title}",
        f"url       : {page.final_url}",
        f"saved     : {res['path']}",
        f"screenshot: {page.screenshot or 'not captured'}",
    ]
    if ingest_res:
        lines.append(
            f"ingest    : {ingest_res['files']} file(s), {ingest_res['chunks']} chunk(s); "
            f"collection chunks: {ingest_res['total']}"
        )
    console.print(Panel("\n".join(lines), title="browser check", border_style="green"))


@app.command()
def health(
    timeout: int = typer.Option(45, "--timeout", help="Per-check timeout in seconds"),
):
    """Run read-only service and project health checks."""
    try:
        res = run_health(timeout_s=timeout)
    except Exception as exc:
        console.print(f"[red]health failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(Panel(res["markdown"], title="health", border_style="green" if res["failed"] == 0 else "yellow"))
    console.print(f"[dim]saved: {res['path']}[/]")


if __name__ == "__main__":
    app()


def artjeck_main() -> None:
    """Console entrypoint: `artjeck` starts the conversational agent directly."""
    if len(sys.argv) == 1:
        agent()
    else:
        app()
