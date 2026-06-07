# second-brain — Design (SDD)

> How the system meets [requirements.md](requirements.md). See also [EVALUATION.md](EVALUATION.md), [MCP.md](MCP.md).

## 1. Architecture
```
 question ─▶ retrieve (vector search) ─▶ assemble cited context ─▶ answer (local LLM)
                  │                                                      │
            Qdrant (vectors)                                      cite every claim
            nomic-embed embeddings                                 or refuse
                  ▲
            ingest: notes / docs / code
```
All model + embedding calls go through an **OpenAI-compatible gateway (LiteLLM)** — free local
models (`chat`/`embed`) by default, optional paid escalation behind a budget cap.

## 2. Key design decisions
1. **Retrieval is separable from generation.** `recall` returns raw evidence with no model call,
   so retrieval quality can be inspected and tested independently of the LLM.
2. **Citations are mandatory.** The answer layer must attach sources; thin evidence → refuse, not guess.
3. **Gateway indirection.** The app calls one endpoint, never a specific model/host — free-local by
   default, escalation is one flag, hosts can fail over.
4. **MCP-native.** Capabilities are exposed as MCP tools (`ask`, `recall`, `ingest`, `learn`,
   `list_tasks`, `add_task`, `complete_task`, `status`) so any agent gains memory + recall.
5. **Eval as a first-class layer.** A graded question set measures answer quality over time.

## 3. Components
- **Ingestion** — chunk + embed notes/docs/code → Qdrant.
- **Retrieval** — vector search returns ranked evidence with source metadata.
- **Answering** — local LLM composes an answer constrained to retrieved evidence, with citations.
- **MCP server** — stdio + HTTP, token-protected over Tailscale, runs 24/7 under launchd.
- **Eval harness** — see EVALUATION.md.

## 4. Testing & evaluation
- Unit/integration tests on ingest + retrieval.
- Eval harness grades answers against a fixed set (coverage, grounding, citation correctness).
