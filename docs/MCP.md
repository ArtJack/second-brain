# second-brain over MCP — cross-device guide

This exposes your second brain as an **MCP server** so any MCP client can query and teach
it: Claude Code / Claude Desktop on the Mac mini, the same over SSH from the iPad Air, and
native MCP clients on the iPad / MacBook Pro over Tailscale.

It reuses the existing engine (same `.env`, same LiteLLM-gateway routing, same Qdrant/Chroma
store), so it answers from the **free local models** by default — no extra cost. The only
models called are the ones the engine already uses.

## Tools

| Tool | Cost | Description |
|---|---|---|
| `ask(question, top_k=0)` | free (local chat) | Answer cited to your own sources. |
| `recall(query, top_k=0)` | free (embed only) | Raw matching chunks, **no chat model** — cheapest grounding. |
| `ingest(path)` | free (embed) | Chunk + embed + store a file or folder. |
| `learn(fact)` | free (embed) | Save a durable Markdown memory and ingest it. |
| `list_tasks(status="open")` | free | List tasks (`open` / `done` / `all`). |
| `add_task(text)` | free | Add a task. |
| `complete_task(task_id)` | free | Mark a task done. |
| `status()` | free | Backend, models, store, chunk count (no secrets). |

## Transports

Selected by `SB_MCP_TRANSPORT`:

- **`stdio`** (default) — the client launches `sb-mcp` as a subprocess. Best for local
  clients and for SSH sessions. No network, no port.
- **`http`** — a standalone Streamable-HTTP service at `/mcp` for networked / multi-device
  clients. Token-protected.

| Env (HTTP) | Default | Meaning |
|---|---|---|
| `SB_MCP_HOST` | `127.0.0.1` | Bind address. Use the Tailscale IP (`YOUR_TAILSCALE_IP`) to share across devices. |
| `SB_MCP_PORT` | `8848` | Port. |
| `SB_MCP_TOKEN` | _(unset)_ | If set, clients must send `Authorization: Bearer <token>`. Keep it in the git-ignored `.env`. |
| `SB_MCP_ALLOWED_HOSTS` | _(bind host + localhost)_ | Comma-separated extra `Host` headers to accept (e.g. a Tailscale MagicDNS name). DNS-rebinding protection stays on; the bind host and localhost are always allowed. |

## 1. Mac mini — Claude Code (stdio)

Already wired: the repo ships a project-scoped `.mcp.json`. Just run Claude Code in the repo:

```bash
cd ~/Projects/AI/projects/second-brain
claude        # the "second-brain" MCP server is auto-loaded
```

Then ask it things like *"use second-brain recall to find what I decided about gateway routing"*.

This config is **project-scoped** — it does not touch your global Claude config or the
`local-agent-lab` environment.

## 2. iPad Air M2 — over SSH (stdio, zero new software)

You already SSH into the Mac mini over Tailscale. From that session, run Claude Code exactly
as above — the MCP server runs on the mini, the iPad is just the terminal:

```bash
ssh <mac-mini-over-tailscale>
cd ~/Projects/AI/projects/second-brain && claude
```

## 3. Claude Desktop (any Mac) — stdio

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/artjack/Projects/AI/projects/second-brain", "sb-mcp"]
    }
  }
}
```

## 4. iPad / MacBook Pro — native MCP client over Tailscale (HTTP)

Run the HTTP service on the mini (see §5 for 24/7), then point any Streamable-HTTP MCP
client at it:

```
URL:    http://YOUR_TAILSCALE_IP:8848/mcp        (Tailscale — reachable from any of your devices)
Header: Authorization: Bearer <SB_MCP_TOKEN>
```

Because it listens on the **Tailscale** address and requires a token, it is reachable from
your devices anywhere but not from the public internet.

## 5. Run it 24/7 on the Mac mini (launchd)

The mini is always on, so run the HTTP service as a LaunchAgent:

```bash
./deploy/install-mcp-service.sh      # generates a token into .env, installs + starts the agent
```

This renders `deploy/com.secondbrain.mcp.plist` into `~/Library/LaunchAgents/`, binds to the
Tailscale IP on port 8848, and keeps it alive across reboots. Manage it with:

```bash
launchctl list | grep secondbrain
launchctl unload ~/Library/LaunchAgents/com.secondbrain.mcp.plist   # stop
tail -f ~/Library/Logs/secondbrain-mcp.log                          # logs
```

## Verify

```bash
# interactive tool tester
npx @modelcontextprotocol/inspector

# or a quick HTTP check (expect 401 without the token, 200 with it)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8848/mcp -d '{}'
```

## Security notes

- HTTP auth is a bearer token; transport privacy comes from **Tailscale** (encrypted mesh).
  Bind to the Tailscale IP, not `0.0.0.0`, unless you intend LAN-wide access.
- Secrets (`SB_MCP_TOKEN`, `QDRANT_API_KEY`, `LITELLM_MASTER_KEY`) live only in the
  git-ignored `.env` — never commit them.
- `ingest`/`learn` write to your store; `ask`/`recall`/`list_tasks`/`status` are read-only
  (annotated with `readOnlyHint` so clients can gate the rest).
