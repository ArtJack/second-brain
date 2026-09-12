# Offsite backup policy (current)

Effective 2026-08-01. Supersedes the 2026-05 draft.

restic pushes to the offsite repository at **03:30 daily**. Retention is 7
daily, 5 weekly, 12 monthly. The repository password lives in the vault, never
on disk in plaintext.

What is covered: the brain's memory directory, the Qdrant snapshots, the
engineering docs, and the per-project QA state. What is not covered: anything
under a `node_modules` or `.venv` directory, and any file the scan's exclusion
rules reject.

A restore is verified quarterly by restoring one snapshot into a scratch
directory and diffing it. An unverified backup is a hope, not a backup.
