# Synthetic fixture: second-brain operations

This file is regression-test data, not a learned user memory.

Artjeck learns only when the user explicitly teaches a fact. Each durable learned fact is
written as an inspectable Markdown file and then ingested through the cited RAG pipeline.
The assistant must not silently save model guesses. Tasks and small assistant state live in
a local SQLite database on the Mac mini. Heavy vectors live in Qdrant on the Alienware.
