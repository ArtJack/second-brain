# second-brain

A personal RAG assistant: point it at your notes, docs, and code, then ask questions and
get **answers cited to your own sources**. Runs on the home AI lab — free local models by
default, one env-flip to Claude quality.

```bash
uv sync
cp .env.example .env
artjeck                    # start talking to Artjeck directly
sb ingest ~/Notes            # or any file/folder (md, txt, pdf, code…)
sb ask "what did I decide about the gateway routing?"
sb learn "My preferred invoice format is PDF with line items."
sb agent                    # ask questions, teach it with /learn <fact>
sb chat                      # interactive
sb status
```

## Why it's built this way (the interesting decisions)

- **OpenAI-compatible everywhere.** `llm.py` talks to one OpenAI-style endpoint. Default is
  the GTX Ollama directly (free, no key, both models warm). Switch to `.env.gateway` and the
  *same code* routes through the LiteLLM gateway — free local embeddings + Claude for the
  answer, with a budget cap. Develop for free, ship with quality, zero code change.
- **Citations are mandatory.** The system prompt forces the model to answer *only* from
  retrieved context and cite `[n]`; the CLI prints the source files + distances. No source,
  no claim — that's the trust signal that separates a real RAG product from a demo.
- **Learning is explicit and inspectable.** `sb learn` and `/learn` in agent mode write
  Markdown memory files under `SB_MEMORY_DIR`, then ingest them through the same cited RAG
  pipeline. The model does not silently save its own guesses as truth.
- **Agent workflow is explicit.** Artjeck routes each console turn through a LangGraph
  workflow that can answer, learn, ingest files, and manage tasks.
- **Tasks are structured.** `SB_STATE_DB` is a SQLite database for durable tasks and
  assistant state. It is separate from semantic memory.
- **Lean, not framework-soup.** Small codebase, the OpenAI SDK + Chroma/Qdrant, no LangChain. You can
  read every step of the pipeline (chunk → embed → store → retrieve → ground → cite), which
  is the point — it shows you understand RAG, not just how to import it.
- **Local-first, lab-ready vector store.** Chroma on disk is the default zero-infra path.
  Set `SB_STORE=qdrant` to use the AI Lab's hosted Qdrant service without changing CLI code.

## Architecture

```
files ──▶ ingest.py ──chunk──▶ llm.embed ──▶ Store (Chroma or Qdrant)
                                                  │
learn ──▶ memory.py ──markdown──▶ ingest.py ──────┤
                                                  │
question ──▶ LangGraph ──ask──▶ Store.query ──top-k──▶ llm.answer ──▶ cited answer
              │
              ├─learn──▶ memory.py
              ├─ingest─▶ ingest.py
              └─task───▶ SQLite (`SB_STATE_DB`)
```

## Roadmap (each item maps to a JD checkbox)

- [x] Swap Chroma → lab **Qdrant** (hosted vector DB)
- [x] Explicit learned memory (`sb learn`, `/learn` in agent mode)
- [x] LangGraph turn workflow + SQLite task store
- [ ] **Hybrid search** (vector + BM25 keyword) + reranking
- [ ] Wrap as an **MCP server** so Claude Desktop / Claude Code can query your brain directly
- [ ] **Eval harness**: a golden Q→expected-source set, measure retrieval hit-rate
- [ ] Web UI (Next.js) + deploy → the public-URL portfolio piece

## Config

`.env` (see `.env.example` / `.env.gateway.example`): `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
`EMBED_MODEL`, `CHAT_MODEL`, `SB_STORE`, `SB_COLLECTION`, `SB_MEMORY_DIR`, `SB_STATE_DB`,
and retrieval knobs `SB_CHUNK_SIZE` / `SB_CHUNK_OVERLAP` / `SB_TOP_K`.

### Chroma default

```bash
SB_STORE=chroma
SB_DATA=./data/chroma
SB_MEMORY_DIR=./data/memory
SB_STATE_DB=./data/artjeck.sqlite3
```

### Qdrant on the AI Lab

```bash
SB_STORE=qdrant
QDRANT_URL=http://192.168.1.159:6333
QDRANT_API_KEY=<from /opt/ai-lab/.env on the Alienware>
```

After switching stores, ingest again because Chroma and Qdrant keep separate collections.

## Storage Policy

On ArtJack's machines, the Mac Mini should stay light:

- Mac Mini keeps the Artjeck code and small SQLite state DB.
- Shared Disk E keeps learned Markdown memories, large source files, exports, and backups.
- Alienware Qdrant should hold the heavy vector index once `QDRANT_API_KEY` is configured.

Current recommended local `.env` shape:

```bash
SB_MEMORY_DIR=/Volumes/DISK/AI/artjeck/memory
SB_STATE_DB=./data/artjeck.sqlite3
QDRANT_URL=http://192.168.1.159:6333
```

Put large documents under `/Volumes/DISK/AI/artjeck/inbox`, then ingest from there:

```bash
artjeck
/ingest /Volumes/DISK/AI/artjeck/inbox
```

## Learning

Teach a fact directly:

```bash
sb learn "ArtJack prefers short implementation summaries with exact file paths."
sb ask "How does ArtJack prefer implementation summaries?"
```

Use agent mode:

```bash
artjeck
/learn The AI Lab gateway is the preferred front door for shared model routing.
What is the preferred front door for shared model routing?
/ingest ~/Notes
/task Review new lease documents
/tasks
/done 1
/exit
```

Learned memories are normal Markdown files in `SB_MEMORY_DIR`, so you can inspect, edit,
delete, or re-ingest them. This keeps learning controlled: the system learns from what you
teach it, not from unverified model guesses.

`sb agent` starts the same conversation loop. The `artjeck` command is just the named
shortcut intended for daily use.
