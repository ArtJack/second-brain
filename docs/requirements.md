# second-brain — Requirements (SDD)

> Spec-driven development artifact: *what* the system must do and *why*. See [design.md](design.md) for *how*.

## 1. Purpose
A local-first RAG assistant that answers questions **only from the user's own notes, docs, and
code**, with **every claim cited**, and whose answer quality is **graded by an eval harness**.

## 2. Users
- The owner, asking questions over their own knowledge (CLI + MCP).
- Any AI agent that connects over MCP and needs memory/recall tools.

## 3. Functional requirements
- **FR-1** Ingest notes/docs/code into a searchable vector index.
- **FR-2** Answer questions grounded strictly in ingested content; **cite every claim** to its source.
- **FR-3** `recall` — return raw retrieved evidence with **no model call** (deterministic retrieval).
- **FR-4** `learn` — persist new facts the user teaches it.
- **FR-5** Lightweight task memory: `add_task` / `list_tasks` / `complete_task`.
- **FR-6** Expose all capabilities over **MCP** (stdio + HTTP) so any agent can use them.
- **FR-7** Report `status` (index size, health).

## 4. Non-functional requirements
- **NFR-1 Local-first / free by default.** Inference + embeddings run on local models through an
  OpenAI-compatible gateway; no per-query cloud cost unless explicitly escalated.
- **NFR-2 Grounding.** No answer without a citation; refuse rather than hallucinate when evidence is thin.
- **NFR-3 Evaluability.** Answer quality measured against a graded eval set (see EVALUATION.md).
- **NFR-4 Privacy.** Personal knowledge never leaves the user's machines; reachable only over Tailscale.

## 5. Out of scope
- Multi-user / cloud SaaS hosting. Web-scale crawling.

## 6. Acceptance criteria
- Every answer carries source citations. ✓
- `recall` returns evidence without invoking a model. ✓
- Eval harness produces graded scores over a fixed question set. ✓
- Usable as an MCP server by an external agent. ✓
