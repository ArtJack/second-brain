# Synthetic fixture: second-brain operations

This file is regression-test data, not a learned user memory. The hosts named here are the
invented ones from `ai-lab-routing.md`, not anyone's real machines.

Artjeck learns only when the user explicitly teaches a fact. Each durable learned fact is
written as an inspectable Markdown file and then ingested through the cited RAG pipeline.
The assistant must not silently save model guesses. Tasks and small assistant state live in
a local SQLite database on `workstation-01`. Heavy vectors live in Qdrant on `gpu-01`.
