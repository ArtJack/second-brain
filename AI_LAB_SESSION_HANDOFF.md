# AI Lab Session Handoff

Date: 2026-05-30

## What Was Done

- Turned `second-brain` into the named assistant `Artjeck`.
- Added the `artjeck` console command as the daily entrypoint.
- Added explicit learned memory:
  - `sb learn "..."`
  - `/learn <fact>` inside `artjeck`
  - `remember <fact>` inside `artjeck`
- Learned memories are Markdown files, so they are inspectable and editable.
- Added Qdrant support alongside Chroma:
  - `SB_STORE=chroma`
  - `SB_STORE=qdrant`
- Enabled live Qdrant in the local ignored `.env` using the key from Alienware `/opt/ai-lab/.env`.
- Added LangGraph turn routing for Artjeck.
- Added SQLite task storage:
  - `/task <task>`
  - `/tasks`
  - `/tasks all`
  - `/done <task-id>`
- Added `/ingest <path>` inside Artjeck.
- Added storage policy to keep the Mac Mini light:
  - code and small SQLite state stay on Mac
  - learned Markdown memories go to Disk E
  - vector index goes to Alienware Qdrant
- Added shared storage folders under `/Volumes/DISK/AI/artjeck`:
  - `memory/`
  - `inbox/`
  - `exports/`
  - `backups/`
- Added `/Volumes/DISK/AI/artjeck/README.md` outside this repo.

## Current Important Paths

- Project:
  `/Users/artjack/AI/projects/second-brain`
- Daily command wrapper:
  `/Users/artjack/.local/bin/artjeck`
- Additional wrapper:
  `/Users/artjack/AI/bin/artjeck`
- Learned memory:
  `/Volumes/DISK/AI/artjeck/memory`
- Large ingest inbox:
  `/Volumes/DISK/AI/artjeck/inbox`
- Local SQLite state:
  `/Users/artjack/AI/projects/second-brain/data/artjeck.sqlite3`
- Live vector DB:
  `http://192.168.1.159:6333` via Qdrant

## Current Local `.env` State

The live `.env` is ignored by git and contains secrets. It is configured like this:

```bash
SB_STORE=qdrant
SB_DATA=./data/chroma
SB_MEMORY_DIR=/Volumes/DISK/AI/artjeck/memory
SB_STATE_DB=./data/artjeck.sqlite3
SB_COLLECTION=second_brain
QDRANT_URL=http://192.168.1.159:6333
QDRANT_API_KEY=<set in ignored .env>
```

Do not commit `.env`.

## Verification Performed

- `uv run python -m py_compile src/secondbrain/*.py`
- `uv run artjeck status`
- Qdrant API key verified against `http://192.168.1.159:6333/collections`.
- Ingested `examples/lab-notes.md` into live Qdrant.
- Asked a cited question from Qdrant successfully.
- Learned a storage-policy memory to Disk E.
- Retrieved that learned memory through Qdrant.
- Confirmed `artjeck status` reports:
  - `store   : qdrant`
  - `memory  : /Volumes/DISK/AI/artjeck/memory`
  - `state   : data/artjeck.sqlite3`

## What Still Needs To Be Done

1. Add an eval harness for retrieval, citation behavior, learning, and task commands.
2. Add memory management commands:
   - list memories
   - show memory
   - edit memory
   - delete memory
   - reindex memory
3. Add hybrid search:
   - Qdrant vector search
   - keyword/BM25 exact search
   - optional reranking
4. Add safe tool permissions before Artjeck can modify files, send messages, or run shell commands.
5. Add reminders and task due dates.
6. Add calendar/email connectors later.
7. Add backup/restore for:
   - Qdrant collection
   - `data/artjeck.sqlite3`
   - `/Volumes/DISK/AI/artjeck/memory`
8. Add LangGraph checkpointing for long-running workflows.
9. Add case/document workflows:
   - timeline extraction
   - evidence index
   - PDF summaries with citations
   - draft letters
10. Consider LangSmith only after evals and multi-step traces have useful signal.

## Notes For Future Agents

- Prefer keeping heavy data off the Mac Mini SSD.
- Do not put live SQLite or Chroma on the SMB share; use Qdrant for vector storage.
- Do not auto-learn model guesses. Learning should remain explicit unless a confirmation flow is added.
- Do not print or commit `QDRANT_API_KEY`, `LITELLM_MASTER_KEY`, or other secrets.
- The GitHub repo may not have existed before this session; verify remotes before pushing.
