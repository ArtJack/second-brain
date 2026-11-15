#!/usr/bin/env python3
"""Check a change against the structure rules of "How we build". Model-free.

The rules: a source file over 2,000 lines may not grow, a new source file stays under
about 400 lines, test doubles live in a shared `fakes.py`, new tests pass fakes in
rather than patch the code's own modules by name, and no handler only logs. Each of
those stays advice until something measures it: in the repositories this was written
for, the largest files grew in nearly every commit that touched them, under rules
that already said "smallest change".

This compares a base commit with the working tree (or with another commit) and
reports what the change did to those numbers. Only the first rule fails the check;
the rest are warnings, because each has honest exceptions a reviewer can accept.

The cap has a fix allowance of 10 lines, and the number is measured, not chosen.
Replayed over the last 100 merges of three repositories (2026-11-14), a cap with no
allowance failed up to a third of them, and in the busiest most of the growth came
from fixes of 10 lines or fewer (median 4): there, moving a function first would put
more at risk than the fix. The feature work that grew the same kind of files added a
median of 42 lines, which is what the rule is for.

Stdlib only, no network, well under a second on an ordinary diff:

    python3 check_structure.py                  # working tree against the default branch
    python3 check_structure.py --base origin/main --head HEAD
    python3 check_structure.py --list           # every code file over the cap
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

CAP = 2000        # no source file above this may grow; where one read of a file by an agent ends
ALLOWANCE = 10    # a fix may add this many lines to a file over the cap; more is an addition
NEW_FILE = 400    # a new source file past this probably holds two things

CODE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".swift", ".go", ".rs",
    ".rb", ".java", ".kt", ".sh",
}
# Only tracked files are ever looked at, so build output is already out. These are
# the folders where a repo commits code it did not write.
SKIPPED_DIRS = {
    "node_modules", "vendor", "site-packages", ".venv", "venv", "dist", ".next", "__pycache__",
}
TEST_DIRS = {"tests", "test", "__tests__", "e2e"}
SHARED_FAKE_DIRS = {"fakes"}
SHARED_FAKE_FILES = {"fakes.py", "conftest.py", "fakes.ts"}

# A test double, found by name: a Fake/Mock/Stub/Dummy/Spy prefix, or a stand-in named for
# the role it replaces (`LegacyStore`, `_CappedStore`, `EmailClientStub`): the prefix alone
# missed most of the store doubles in one suite. A pytest class (`TestStore`) is not one,
# and neither is a page object (`LoginPage`), which wraps the real page.
_DOUBLE = re.compile(
    r"^\s*(?:export\s+)?class\s+((?!Test)_?(?:(?:Fake|Mock|Stub|Dummy|Spy)[A-Za-z0-9_]*"
    r"|[A-Za-z0-9_]*?(?:Store|Client|Service|Gateway|Repository|Repo|Session|Browser"
    r"|Notifier|Sender|Clock|Stub|Mock|Fake|Spy|Double|Dummy)))\b"
)
_PATCH = re.compile(
    r"monkeypatch\.setattr\(|\bmock\.patch(?:\.object)?\(|@patch(?:\.object)?\("
    r"|\b(?:vi|jest)\.mock\("
)
_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


class SetupError(Exception):
    """The check could not run. Never reported as a pass."""


def git(repo: Path, *args: str, check: bool = True) -> bytes:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if check and done.returncode != 0:
        raise SetupError(f"git {' '.join(args)}: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout if done.returncode == 0 else b""


def is_code(path: str) -> bool:
    p = PurePosixPath(path)
    if p.suffix not in CODE_SUFFIXES or p.name.endswith((".min.js", ".d.ts")):
        return False
    return not (set(p.parts[:-1]) & SKIPPED_DIRS)


def is_test(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name
    if set(p.parts[:-1]) & TEST_DIRS:
        return True
    return (
        name.startswith("test_") or name == "conftest.py"
        or name.endswith(("_test.py", "_test.go", "Tests.swift"))
        or ".test." in name or ".spec." in name
    )


def is_shared_fake(path: str) -> bool:
    p = PurePosixPath(path)
    return p.name in SHARED_FAKE_FILES or bool(set(p.parts[:-1]) & SHARED_FAKE_DIRS)


def default_base(repo: Path) -> str:
    """The commit this branch left the default branch at."""
    candidates = []
    head_ref = git(repo, "symbolic-ref", "-q", "refs/remotes/origin/HEAD", check=False)
    if head_ref.strip():
        candidates.append(head_ref.decode().strip().removeprefix("refs/remotes/"))
    candidates += ["origin/main", "origin/master", "main", "master"]
    for ref in candidates:
        if git(repo, "rev-parse", "-q", "--verify", f"{ref}^{{commit}}", check=False):
            base = git(repo, "merge-base", ref, "HEAD", check=False).decode().strip()
            if base:
                return base
    raise SetupError("no default branch found to compare with; pass --base <ref>")


def content(repo: Path, ref: str | None, path: str) -> bytes | None:
    """A file's bytes at a commit, or in the working tree when ref is None."""
    if ref is None:
        target = repo / path
        return target.read_bytes() if target.is_file() else None
    done = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True, check=False
    )
    return done.stdout if done.returncode == 0 else None


def line_count(data: bytes | None) -> int:
    return 0 if data is None else len(data.splitlines())


def changes(repo: Path, base: str, head: str | None) -> list[tuple[str, str, str]]:
    """(status letter, path at base, path at head) for every file the change touches."""
    args = ["diff", "--name-status", "-M", "-z", base, *([head] if head else [])]
    fields = [f.decode(errors="replace") for f in git(repo, *args).split(b"\0") if f]
    out, i = [], 0
    while i < len(fields):
        status = fields[i][0]
        if status in "RC":
            out.append((status, fields[i + 1], fields[i + 2]))
            i += 3
        else:
            out.append((status, fields[i + 1], fields[i + 1]))
            i += 2
    if head is None:  # files git does not track yet are part of the change too
        for raw in git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
            if raw:
                path = raw.decode(errors="replace")
                out.append(("A", path, path))
    return out


def diff_lines(
    repo: Path, base: str, head: str | None, status: str, old: str, new: str
) -> tuple[list[str], list[str]]:
    """Lines the change added to and removed from one file."""
    if status == "A":
        data = content(repo, head, new) or b""
        return data.decode(errors="replace").splitlines(), []
    args = ["diff", "-U0", "-M", base, *([head] if head else []), "--", old, new]
    added, removed = [], []
    for line in git(repo, *args).decode(errors="replace").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    return added, removed


def quiet_handlers(source: bytes | None) -> int | None:
    """How many `except` blocks only pass or only log. None when the file cannot be parsed."""
    if source is None:
        return 0
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    def silent(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if not isinstance(stmt, ast.Expr):
            return False
        value = stmt.value
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.Call):
            func = value.func
            if isinstance(func, ast.Name):
                return func.id == "print"
            return isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS
        return False

    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and all(silent(s) for s in node.body)
    )


def inventory(repo: Path, cap: int) -> tuple[int, list[str]]:
    rows = []
    for raw in git(repo, "ls-files", "-z").split(b"\0"):
        path = raw.decode(errors="replace")
        if not path or not is_code(path):
            continue
        n = line_count(content(repo, None, path))
        if n > cap:
            rows.append((n, path))
    rows.sort(reverse=True)
    source = [r for r in rows if not is_test(r[1])]
    lines = [
        f"code files over {cap} lines in {repo.name}: "
        f"{len(source)} source, {len(rows) - len(source)} test"
    ]
    lines += [f"  {n:>6}  {'test  ' if is_test(path) else 'source'}  {path}" for n, path in rows]
    return 0, lines


def check(repo: Path, base: str, head: str | None, cap: int, new_file: int, strict: bool,
          allowance: int = ALLOWANCE) -> tuple[int, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    patch_net = 0
    touched = 0

    for status, old, new in changes(repo, base, head):
        if status == "D" or not is_code(new):
            continue
        touched += 1
        after = line_count(content(repo, head, new))
        before = 0 if status == "A" else line_count(content(repo, base, old))
        test = is_test(new)

        if after > cap and after > before:
            what = "grew past the cap" if before > cap else "crossed the cap"
            line = f"{new}  {before} → {after} (+{after - before})"
            if test:
                warnings.append(
                    f"test file {what}: {line}. Split it along the same lines as its source."
                )
            elif before > cap and after - before <= allowance:
                notes.append(f"within the {allowance}-line fix allowance: {line}")
            else:
                failures.append(
                    f"{what}: {line}\n"
                    "      Put the new code in a new file, or extract the part you are changing "
                    "first as its own `refactor:` commit (How we build, in AGENTS.md)."
                )
        elif status == "A" and not test and after > new_file:
            warnings.append(
                f"new source file over {new_file} lines: {new}  {after}. "
                "One file, one responsibility."
            )

        if test:
            added, removed = diff_lines(repo, base, head, status, old, new)
            patch_net += sum(1 for text in added if _PATCH.search(text))
            patch_net -= sum(1 for text in removed if _PATCH.search(text))
            if not is_shared_fake(new):
                doubles = [m.group(1) for text in added if (m := _DOUBLE.match(text))]
                if doubles:
                    warnings.append(
                        f"test double defined in a test file: {new}: {', '.join(doubles)}. "
                        "Use or extend the shared fake (How we build)."
                    )
        elif new.endswith(".py"):
            now = quiet_handlers(content(repo, head, new))
            was = 0 if status == "A" else quiet_handlers(content(repo, base, old))
            if now is None or was is None:
                warnings.append(f"not parsed, handlers not counted: {new}")
            elif now > was:
                warnings.append(
                    f"handlers that only log or pass: {new}  {was} → {now}. "
                    "A failure must reach a state, a report or a person (How we build)."
                )

    if patch_net > 0:
        warnings.append(
            f"patching by name: {patch_net} more `monkeypatch.setattr` / `patch(` / `mock(` "
            "lines in tests. Pass a fake in instead (How we build), or say in the report why "
            "there is no seam yet."
        )

    scope = f"{base[:9]} → {head[:9] if head else 'working tree'}"
    lines = [
        f"structure check: {repo.name}  {scope}  "
        f"({touched} code files changed; cap {cap}, new files {new_file})"
    ]
    lines += [f"FAIL  {f}" for f in failures]
    lines += [f"warn  {w}" for w in warnings]
    lines += [f"note  {n}" for n in notes]
    if failures or warnings:
        lines.append(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
    else:
        lines.append(
            "ok: no file crossed the cap or grew past the allowance, and nothing to warn about"
        )
    return (1 if failures or (strict and warnings) else 0), lines


def run(argv: list[str]) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=".", help="repository to check (default: here)")
    parser.add_argument(
        "--base", help="commit to compare with (default: where this branch left the default branch)"
    )
    parser.add_argument(
        "--head", help="commit to check (default: the working tree, uncommitted work included)"
    )
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument(
        "--allowance", type=int, default=ALLOWANCE,
        help="lines a fix may add to a file already over the cap (default: %(default)s)",
    )
    parser.add_argument("--new-file", type=int, default=NEW_FILE)
    parser.add_argument("--strict", action="store_true", help="warnings fail too")
    parser.add_argument(
        "--list", action="store_true", help="list every code file over the cap and exit"
    )
    args = parser.parse_args(argv)
    try:
        top = git(Path(args.repo).expanduser(), "rev-parse", "--show-toplevel").decode().strip()
        repo = Path(top)
        if args.list:
            return inventory(repo, args.cap)
        base = (
            git(repo, "rev-parse", "--verify", f"{args.base}^{{commit}}").decode().strip()
            if args.base else default_base(repo)
        )
        head = (
            git(repo, "rev-parse", "--verify", f"{args.head}^{{commit}}").decode().strip()
            if args.head else None
        )
        return check(repo, base, head, args.cap, args.new_file, args.strict, args.allowance)
    except SetupError as problem:
        return 2, [f"structure check could not run: {problem}"]


if __name__ == "__main__":
    code, report = run(sys.argv[1:])
    print("\n".join(report))
    sys.exit(code)
