# The nightly scan

Runs at 03:15. Read-only with respect to the source files: it hashes, decides
what changed, and ingests.

It is bounded by a per-run file cap so a newly added target cannot turn one
night into a multi-hour ingest. The cap bounds work, not sight: files beyond it
are deferred to the next run and reported as deferred, never silently dropped.

Anything an exclusion rule rejects is reported by name with the rule that
rejected it. A rule that quietly eats a whole target is the failure this
reporting exists to make loud.
