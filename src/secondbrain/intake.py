"""Interactive private benchmark intake.

Answers are saved under git-ignored data/ after every prompt. Completed responses become
an isolated private corpus plus a benchmark that can exercise retrieval against it.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

PRIVATE_EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "private-eval"
PRIVATE_COLLECTION = "second_brain_private_eval"
SESSION_FILE = "intake.json"
BENCHMARK_FILE = "benchmark.json"

QUESTION_GROUPS = {
    "personal-context": (
        "What name should Artjeck use when addressing you?",
        "What city, region, or timezone should Artjeck assume for planning?",
        "Which languages are you comfortable using for work and daily communication?",
        "Which personal roles matter most in your life right now?",
        "Who are the important people or groups Artjeck should recognize by name or role?",
        "What parts of your background are most useful when giving you advice?",
        "Which current responsibilities take the most mental space?",
        "What recurring personal commitments should planning take into account?",
        "What does a genuinely good week look like for you?",
        "What personal context is commonly misunderstood unless you explain it explicitly?",
    ),
    "communication": (
        "How concise or detailed should Artjeck be by default?",
        "When should Artjeck challenge your assumptions instead of simply helping execute?",
        "What tone works best when you are focused and moving quickly?",
        "What tone works best when you are frustrated or overloaded?",
        "How should implementation summaries be structured?",
        "How should Artjeck present choices when there are several reasonable paths?",
        "Which writing habits or phrases should Artjeck avoid?",
        "When do you prefer a direct answer versus a step-by-step explanation?",
        "How should reminders or follow-ups be phrased so they are useful rather than annoying?",
        "What makes an assistant response feel especially helpful to you?",
    ),
    "work-style": (
        "What hours of the day are usually best for your deepest work?",
        "How do you prefer to break a large project into manageable steps?",
        "What kinds of tasks do you tend to postpone?",
        "What kinds of tasks give you energy?",
        "How should Artjeck help when priorities compete?",
        "What level of planning detail helps you act without feeling boxed in?",
        "How do you prefer to track decisions, tasks, and follow-ups?",
        "What signs indicate that you are taking on too much at once?",
        "What is your preferred way to review progress at the end of a week?",
        "Which recurring workflow friction would you most like to remove?",
    ),
    "career": (
        "What role or professional identity are you currently working toward?",
        "What compensation range or business outcome are you aiming for?",
        "Which skills are your strongest professional advantages?",
        "Which skill gaps deserve the most attention over the next six months?",
        "What kinds of work do you want more of?",
        "What kinds of work do you want less of?",
        "Which industries or customer problems interest you most?",
        "What would make a portfolio project feel credible to an employer or customer?",
        "Which professional milestones matter most this year?",
        "What is the biggest current obstacle between you and the next career step?",
    ),
    "projects": (
        "What active projects should Artjeck know about right now?",
        "Which project is the highest priority and why?",
        "What does success look like for the highest-priority project?",
        "What is the next concrete milestone for that project?",
        "Which project ideas should stay parked for later rather than distract from current work?",
        "Which technical or product decisions have already been made and should not be reopened casually?",
        "What risks could derail your current projects?",
        "Which stakeholders or users matter most for your current work?",
        "What artifacts should each serious project produce?",
        "How should Artjeck decide whether a new idea is worth pursuing now?",
    ),
    "technical-environment": (
        "Which computers, servers, or devices are part of your working environment?",
        "What is the preferred front door for model routing?",
        "Which local or hosted models do you use for coding, chat, and embeddings?",
        "Which data stores and databases are already part of your setup?",
        "What storage policy should Artjeck follow across your devices and disks?",
        "Which paths, services, or repositories are important enough to remember?",
        "Which deployment environments or hosting providers do you use?",
        "What backup policy should apply to your important project data?",
        "Which technical constraints most often affect architecture decisions?",
        "What parts of your environment should Artjeck never modify without confirmation?",
    ),
    "knowledge-management": (
        "What kinds of information should Artjeck remember durably?",
        "What kinds of information should Artjeck never learn automatically?",
        "Where do your notes and reference documents currently live?",
        "How should conflicting notes or outdated decisions be handled?",
        "Which documents or sources should be treated as authoritative?",
        "How should Artjeck distinguish a fact, a preference, a task, and an idea?",
        "What should happen when Artjeck cannot find enough evidence to answer?",
        "Which recurring questions would you most like your second brain to answer reliably?",
        "How should learned memories be reviewed, edited, or deleted?",
        "What would make you trust Artjeck with more of your knowledge over time?",
    ),
    "logistics-domain": (
        "Which trucking, logistics, or operations problems do you understand best?",
        "Which logistics workflow seems most promising for a first product?",
        "Who is the ideal first customer for that product?",
        "What manual work should the product reduce or eliminate?",
        "Which documents or data sources matter most in that workflow?",
        "What fields should be extracted from a Bill of Lading or similar document?",
        "What should an IFTA-related workflow calculate or organize?",
        "Which parts of the logistics workflow require human review?",
        "What would make a logistics MVP valuable enough for a customer to pay for?",
        "What should remain outside the first logistics MVP?",
    ),
    "planning-routines": (
        "Which recurring tasks should Artjeck help you remember?",
        "How far ahead do you prefer to plan your week?",
        "What should happen during a daily planning check-in?",
        "What should happen during a weekly review?",
        "Which deadlines or due-date patterns matter most in your work or personal life?",
        "How should overdue tasks be handled?",
        "How should Artjeck help when a task is vague or too large?",
        "What categories would make your task list easier to use?",
        "Which habits or routines are worth protecting on busy weeks?",
        "What planning mistakes do you want Artjeck to help prevent?",
    ),
    "assistant-boundaries": (
        "Which actions may Artjeck take without asking first?",
        "Which actions always require explicit confirmation?",
        "Which information should never be stored in the second brain?",
        "How should Artjeck handle secrets, tokens, passwords, and account numbers?",
        "When is it acceptable for Artjeck to use a paid model?",
        "When should Artjeck prefer local models even if they are slower or less polished?",
        "Which external systems may Artjeck read from in the future?",
        "Which external systems may Artjeck write to in the future?",
        "How should Artjeck surface uncertainty or conflicting evidence?",
        "What would make you stop trusting an assistant immediately?",
    ),
}

PROMPTS = tuple(
    {
        "id": f"{category}-{index:02d}",
        "category": category,
        "question": question,
    }
    for category, questions in QUESTION_GROUPS.items()
    for index, question in enumerate(questions, start=1)
)

if len(PROMPTS) != 100:  # Keep the terminal commitment explicit.
    raise RuntimeError(f"Private intake must contain exactly 100 prompts, found {len(PROMPTS)}.")


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_session(output_dir: str | Path = PRIVATE_EVAL_DIR) -> dict:
    path = Path(output_dir).expanduser() / SESSION_FILE
    if not path.exists():
        return {"schema_version": 1, "responses": {}}
    session = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(session, dict) or not isinstance(session.get("responses"), dict):
        raise ValueError(f"Invalid private intake session: {path}")
    return session


def save_session(output_dir: str | Path, session: dict) -> Path:
    path = Path(output_dir).expanduser() / SESSION_FILE
    session["schema_version"] = 1
    session["updated_at"] = _timestamp()
    _write_json(path, session)
    return path


def reset_session(output_dir: str | Path = PRIVATE_EVAL_DIR) -> None:
    root = Path(output_dir).expanduser()
    for path in [root / SESSION_FILE, root / BENCHMARK_FILE]:
        if path.exists():
            path.unlink()
    corpus_dir = root / "corpus"
    for category in QUESTION_GROUPS:
        path = corpus_dir / f"{category}.md"
        if path.exists():
            path.unlink()


def intake_status(session: dict) -> dict:
    responses = session.get("responses") or {}
    answered = sum(1 for response in responses.values() if response.get("answer"))
    skipped = sum(1 for response in responses.values() if response.get("skipped"))
    return {
        "answered": answered,
        "skipped": skipped,
        "completed": answered + skipped,
        "remaining": len(PROMPTS) - answered - skipped,
        "total": len(PROMPTS),
    }


def build_private_artifacts(output_dir: str | Path = PRIVATE_EVAL_DIR) -> dict:
    """Render private source documents and a retrieval benchmark from saved answers."""
    root = Path(output_dir).expanduser()
    corpus_dir = root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    session = load_session(root)
    responses = session["responses"]
    cases = []
    source_files = []

    for category in QUESTION_GROUPS:
        answered = [
            (prompt, responses[prompt["id"]]["answer"])
            for prompt in PROMPTS
            if prompt["category"] == category and responses.get(prompt["id"], {}).get("answer")
        ]
        path = corpus_dir / f"{category}.md"
        if not answered:
            if path.exists():
                path.unlink()
            continue
        lines = [
            f"# Private intake: {category.replace('-', ' ').title()}",
            "",
            "Private user-provided second-brain evaluation source. Keep local.",
            "",
        ]
        for prompt, answer in answered:
            lines.extend([f"## {prompt['question']}", "", answer, ""])
            cases.append(
                {
                    "id": prompt["id"],
                    "query": prompt["question"],
                    "expected_sources": [f"corpus/{category}.md"],
                    "reference_answer": answer,
                    "tags": ["private-intake", category],
                }
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        source_files.append(str(path))

    benchmark = {
        "name": "private-intake",
        "description": "Private user-provided benchmark. Keep local and do not commit.",
        "corpus": ["corpus"],
        "corpus_collection": PRIVATE_COLLECTION,
        "cases": cases,
    }
    benchmark_path = root / BENCHMARK_FILE
    _write_json(benchmark_path, benchmark)
    return {
        **intake_status(session),
        "benchmark": str(benchmark_path),
        "corpus_files": source_files,
        "benchmark_cases": len(cases),
    }


def run_intake(
    output_dir: str | Path = PRIVATE_EVAL_DIR,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict:
    """Collect private answers interactively, saving after every response."""
    root = Path(output_dir).expanduser()
    session = load_session(root)
    responses = session["responses"]
    output_fn("Private second-brain intake: 100 questions.")
    output_fn(f"Answers save after every prompt under {root}.")
    output_fn("Do not enter passwords, tokens, account numbers, or government IDs.")
    output_fn("Commands: /skip, /back, /status, /quit, /help")

    while True:
        pending = [prompt for prompt in PROMPTS if prompt["id"] not in responses]
        if not pending:
            output_fn("All 100 prompts are complete.")
            break
        prompt = pending[0]
        status = intake_status(session)
        output_fn("")
        output_fn(
            f"[{status['completed'] + 1}/{status['total']}] "
            f"{prompt['category']} | {prompt['question']}"
        )
        try:
            response = input_fn("answer> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            output_fn("Paused. Progress is saved.")
            break

        if response == "/quit":
            output_fn("Paused. Progress is saved.")
            break
        if response == "/status":
            status = intake_status(session)
            output_fn(
                f"answered={status['answered']} skipped={status['skipped']} "
                f"remaining={status['remaining']}"
            )
            continue
        if response == "/help":
            output_fn("Commands: /skip, /back, /status, /quit, /help")
            continue
        if response == "/back":
            completed = [prompt for prompt in PROMPTS if prompt["id"] in responses]
            if not completed:
                output_fn("Nothing to revisit yet.")
                continue
            previous = completed[-1]
            del responses[previous["id"]]
            save_session(root, session)
            output_fn(f"Reopened: {previous['question']}")
            continue
        if response == "/skip":
            responses[prompt["id"]] = {"skipped": True, "updated_at": _timestamp()}
        elif not response:
            output_fn("Enter an answer or use /skip.")
            continue
        else:
            responses[prompt["id"]] = {"answer": response, "updated_at": _timestamp()}
        save_session(root, session)

    summary = build_private_artifacts(root)
    output_fn(
        f"Saved: answered={summary['answered']} skipped={summary['skipped']} "
        f"remaining={summary['remaining']}"
    )
    if summary["benchmark_cases"]:
        output_fn(f"Private benchmark: {summary['benchmark']}")
    return summary
