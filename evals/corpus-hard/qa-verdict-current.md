# QA routing (current)

Effective 2026-08-28. All four active projects — Sales, ifta-agent,
artjeck-technology, second-brain — route testing judgment to the **verdict**
plugin agent, not to `qa-tester`.

State of record is `~/.claude/verdict/<key>/`, one directory per project, held
locally. The share's history for each project is a frozen read-only archive and
must not be written to.

`qa-tester` is retired for project QA. It is kept only for ad-hoc ISTQB study
questions. A new project needing QA gets a verdict profile written for it; the
old routing is not revived.
