# second-brain

A personal RAG assistant: point it at your notes, docs, and code, then ask questions and
get **answers cited to your own sources**. Runs on the home AI lab — free local models by
default, one env-flip to Claude quality.

```bash
uv sync
cp .env.example .env
sb ingest ~/Notes            # or any file/folder (md, txt, pdf, code…)
sb ask "what did I decide about the gateway routing?"
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
- **Lean, not framework-soup.** ~250 lines, the OpenAI SDK + Chroma, no LangChain. You can
  read every step of the pipeline (chunk → embed → store → retrieve → ground → cite), which
  is the point — it shows you understand RAG, not just how to import it.
- **Local-first vector store.** Chroma on disk for v1 (zero infra). The `Store` interface is
  small on purpose so swapping to the lab's Qdrant is a drop-in.

## Architecture

```
files ──▶ ingest.py ──chunk──▶ llm.embed ──▶ Store (Chroma)
                                                  │
question ──▶ llm.embed ──▶ Store.query ──top-k──▶ ask.py ──ground+cite──▶ llm.answer ──▶ cited answer
```

## Roadmap (each item maps to a JD checkbox)

- [ ] Swap Chroma → lab **Qdrant** (hosted vector DB)
- [ ] **Hybrid search** (vector + BM25 keyword) + reranking
- [ ] Wrap as an **MCP server** so Claude Desktop / Claude Code can query your brain directly
- [ ] **Eval harness**: a golden Q→expected-source set, measure retrieval hit-rate
- [ ] Web UI (Next.js) + deploy → the public-URL portfolio piece

## Config

`.env` (see `.env.example` / `.env.gateway.example`): `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
`EMBED_MODEL`, `CHAT_MODEL`, and retrieval knobs `SB_CHUNK_SIZE` / `SB_CHUNK_OVERLAP` / `SB_TOP_K`.
