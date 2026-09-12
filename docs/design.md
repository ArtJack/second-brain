# second-brain — Design (SDD)

> How the system meets [requirements.md](requirements.md). See also [EVALUATION.md](EVALUATION.md), [MCP.md](MCP.md).

## 1. Architecture
```
 question ─▶ hybrid retrieval (vector + local BM25) ─▶ assemble cited context ─▶ answer (local LLM)
                  │                                                                  │
            Qdrant (vectors) + local keyword scan                              cite every claim
            nomic-embed embeddings                                             or refuse
                  ▲
            ingest: notes / docs / code
```
All model + embedding calls go through an **OpenAI-compatible gateway (LiteLLM)** — free local
models (`chat`/`embed`) by default, optional paid escalation behind a budget cap.

## 2. Key design decisions
1. **Retrieval is separable from generation.** `recall` returns raw evidence with no model call,
   so retrieval quality can be inspected and tested independently of the LLM.
2. **Citations are mandatory.** The answer layer must attach sources; thin evidence → refuse, not guess.
3. **Hybrid retrieval is toggleable.** Vector search handles semantic matches; local BM25 keyword
   fusion rescues exact section/list lookups, expands to adjacent chunks, and can be disabled with
   `SB_HYBRID=0` for A/B evaluation.
4. **Gateway indirection.** The app calls one endpoint, never a specific model/host — free-local by
   default, escalation is one flag, hosts can fail over.
5. **MCP-native.** Capabilities are exposed as MCP tools (`ask`, `recall`, `ingest`, `learn`,
   `list_tasks`, `add_task`, `complete_task`, `status`) so any agent gains memory + recall.
6. **Eval as a first-class layer.** A graded question set measures answer quality over time.

## 3. Components
- **Ingestion** — chunk + embed notes/docs/code → Qdrant.
- **Retrieval** — hybrid retrieval (vector search + local BM25) returns fused,
  de-duplicated evidence with source metadata.
- **Answering** — local LLM composes an answer constrained to retrieved evidence, with citations.
- **MCP server** — stdio + HTTP, token-protected over Tailscale, runs 24/7 under launchd.
- **Eval harness** — see EVALUATION.md.

## 4. Testing & evaluation
- Unit/integration tests on ingest + retrieval.
- Eval harness grades answers against a fixed set (coverage, grounding, citation correctness).

## 5. Chunk identity, and how to migrate a collection to it

A chunk's id is a hash of its normalised text: whitespace collapsed, case left
alone, sha256 truncated to 32 hex characters behind a `c:` prefix. Location is
payload — `path`, plus `mtime`, `doc_type`, `ingested_at`, `title` — and `source`
remains as an alias of `path` so every existing read site keeps working.

Identity used to be the filesystem path captured at ingest time, which made
location and identity the same thing. Copying a file created a second corpus,
and the web endpoint, which wrote each request body into a temporary directory,
created a brand new document per request whose citation pointed at a path that
no longer existed. `sb gc` compounds that: those paths genuinely are missing, so
the sweep reads web-ingested material as deleted.

### Migrating an existing collection

Do **not** mutate the live collection in place. Ingest into a fresh one, verify,
then swap.

```bash
# 1. Snapshot first. It is one curl and it is the only way back.
sb health --json | grep -i qdrant

# 2. Full re-ingest into a new collection.
SB_COLLECTION=second_brain_v2 sb ingest ~/Projects --reset

# 3. Verify: the count should be at or below the old one, never above.
sb --collection second_brain status
sb --collection second_brain_v2 status

# 4. Spot-check a query that has a known answer.
sb --collection second_brain_v2 ask "where does the gateway run?"

# 5. Rebuild the keyword index for the new collection.
SB_COLLECTION=second_brain_v2 sb keyword-reindex

# 6. Swap by changing SB_COLLECTION in the service environment, then restart
#    the service. Keep the old collection until the next nightly has run clean.
```

A count that comes back *higher* than the old collection means identity is not
collapsing duplicates as intended. Stop and find out why before swapping.

Re-ingest is also required by the contextual chunk headers, since those change
what is embedded. The two migrations are the same operation and should be done
once, together.
