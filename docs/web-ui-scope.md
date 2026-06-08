# Web UI — Scope & Build Plan

**Status:** Scoping (not started). Author: Claude (Opus 4.8), 2026-06-07.
**Roadmap line this closes:** `README.md` — `- [ ] Web UI (Next.js) + deploy → the public-URL portfolio piece`.

This is a design/scope doc, not a work order. It records the decisions made with the owner,
the architecture, the phasing, and the load-bearing refactors — verified against the current code.

---

## 1. Goal

Ship a **standalone Next.js web client for second-brain** — the public, clickable proof that the
CLI/MCP RAG engine is real. It must do two jobs at once:

- **Recruiter/customer demo (anonymous):** ask questions, get answers *cited to sources*, on a
  curated public corpus. The citations are the point — "no source, no claim" made visible.
- **Owner daily-driver (authenticated):** the full interactive brain — ask / ingest / learn /
  tasks / eval traces — against the real `second_brain` collection.

### Decisions locked with the owner (2026-06-07)

| Decision | Choice |
|---|---|
| Public corpus | **Both** — curated *ArtJeck Lab* corpus (default) + a *neutral* (ISTQB/QA) corpus, switchable. |
| Interactivity | **Full interactive** (ask / ingest / learn / tasks). |
| Code placement | **Standalone app in the `second-brain` repo**, its own Vercel project. |
| Write-surface gating | **Both paths:** owner login → real brain; anonymous → ephemeral sandbox. Phase owner-mode first. |

The private `second_brain` collection is **never** exposed to anonymous visitors.

---

## 2. Architecture (mirrors the proven IFTA pattern on artjeck.com)

The site already runs a Mac-mini FastAPI backend behind a Cloudflare Tunnel
(`ifta-api.artjeck.com`), called from Vercel through a Next API route that holds the key
server-side, with per-IP rate limiting and Cloudflare Turnstile. The brain demo is the same shape.

```
Browser ─▶ brain.artjeck.com  (Vercel — standalone Next.js app: second-brain/web)
            │   chat UI · citations panel · corpus switcher · owner login · anon sandbox UI
            └─▶ Next API routes  /api/brain/*   (thin proxy)
                  · Turnstile verify (anon)   · per-IP + per-session rate limit
                  · backend bearer token held server-side   · input caps
                  └─▶ brain-api.artjeck.com   (Cloudflare Tunnel)
                        └─▶ FastAPI service on Mac mini  (second-brain/server)
                              · /ask /recall /ingest /learn /tasks /status /health
                              · SSE streaming for /ask
                              · collection routing: public | neutral | sandbox-{sid} | real
                              · auth: owner JWT  vs  anon session token (scope-checked)
                              · sandbox lifecycle: create · TTL reaper · wipe
                              └─▶ Qdrant + Ollama on Alienware  (+ LiteLLM gateway, $50/30d cap)
```

Why standalone (per owner choice) over folding into the artjeck.com site: a self-contained
repo artifact that checks the second-brain roadmap box. Cost: it gets its own Vercel project and
duplicates some infra (Turnstile keys, rate-limit code, tunnel hostname) the site already has —
acceptable, and the IFTA route is a copy-paste-grade reference.

---

## 3. Component breakdown

### A. Backend serving layer — `src/secondbrain/server.py` (FastAPI)
A REST sibling to `mcp_server.py` (which already wraps the same tools — lift its tool list).
Reuses `ask()`, `recall`, `ingest_paths`, `memory.learn`, `TaskStore`, `Store`, `health`.

| Endpoint | Method | Notes | Write? |
|---|---|---|---|
| `/health` | GET | lab liveness (drives the UI's "lab asleep" fallback) | — |
| `/status` | GET | chunk count, model, collection | — |
| `/ask` | POST | cited answer; **`/ask` SSE** streams tokens | — |
| `/recall` | POST | raw chunks, no model call | — |
| `/ingest` | POST | files/text → chunks | ✅ |
| `/learn` | POST | markdown memory → ingest | ✅ |
| `/tasks` | GET/POST | list / add | ✅ on POST |
| `/tasks/{id}/complete` | POST | | ✅ |

- **Collection routing:** every request resolves a `collection` to a real Qdrant collection:
  `public → second_brain_public`, `neutral → second_brain_neutral`,
  `sandbox → second_brain_sandbox_{session_id}`, `real → second_brain` (owner only).
  Read-only collections (`public`, `neutral`) reject writes with 403.
- **Auth & scope:** owner bearer (JWT/shared secret) unlocks `real` + all writes; anon session
  token (issued after Turnstile) is scoped to `public`/`neutral` (read) + its own `sandbox`.
- **Streaming:** add `llm.answer_stream()` (`stream=True` generator) and an SSE `/ask` path.
- **Sandbox lifecycle:** create `sandbox-{sid}` on session start; background TTL reaper drops
  `second_brain_sandbox_*` older than N hours; per-session caps (max docs / bytes / questions).

### B. Cloudflare Tunnel — `brain-api.artjeck.com`
New tunnel hostname → `localhost:<port>` on the Mac mini. Mirror the IFTA tunnel config.
Update the artjeck.com **privacy page** disclosure (already lists the Cloudflare tunnel). *(owner action)*

### C. Frontend — standalone Next.js app `second-brain/web` (own Vercel project, `brain.artjeck.com`)
- Stack: Next.js App Router + Tailwind; match ArtJeck brand tokens (calm-confident, no vanity copy).
- **Chat view:** streamed tokens, model badge (local vs Claude), latency.
- **Citations panel (centerpiece):** per-answer source list, expandable cited chunk text,
  file + distance, surfaces invalid citations.
- **Corpus switcher:** Lab (default) / Neutral / (owner: Real).
- **Suggested prompts** per corpus (e.g. Lab: "How does ArtJeck's hybrid retrieval work?").
- **Owner login:** NextAuth single-owner credential or GitHub OAuth restricted to `ArtJack`.
- **Anon sandbox UI:** "paste a doc / upload → ask about it"; clear "throwaway brain, wiped in N h".
- **Next API routes** proxy to brain-api: hold backend token, attach session cookie, Turnstile
  verify, per-IP rate limit (reuse IFTA `Map` pattern), input caps.
- **Availability fallback:** poll `/health`; if the lab is offline, render a graceful "lab is
  asleep" state backed by pre-captured example answers so the page is never dead. *(recruiter-critical)*

### D. Public corpora content — `scripts/seed_public_corpora.py`
- `second_brain_public` (Lab): curated clean docs — README, `docs/design.md`,
  `docs/requirements.md`, sanitized project write-ups. **Audit each doc for private data first.**
- `second_brain_neutral`: ISTQB syllabus / generic KB (ties to QA background).
- Reproducible seed script: `sb ingest --collection <name> <paths>` (needs CLI `--collection` flag — see §4).

### E. Abuse / cost controls
- Turnstile on first anon interaction → issues the session token.
- Per-IP + per-session rate limits (Next route; reuse IFTA).
- **Anon = free local model only** (no Claude for anonymous → cost containment). Owner may use the gateway.
- Sandbox caps: max docs, max bytes, max questions/session, TTL reaper.
- LiteLLM `$50/30d` cap = backstop. Reuse `tracing.py` spans for owner-only metrics.

### F. Eval / QA / docs
- Tests for the FastAPI layer: **auth scoping** (anon cannot write to `real`/`public`),
  **sandbox isolation**, **collection routing**, streaming shape.
- `docs/WEB_UI.md` usage guide; flip the README roadmap line.
- **Security review of the public write surface before go-live** (anon sandbox is the risk).

---

## 4. Load-bearing refactors & risks (verified against current code)

1. **Collection threading (small refactor — backbone of everything).** `Store(collection=…)`
   already exists (`store.py:253`, both backends). But `ask()` (`ask.py:28`), `ingest_paths`,
   and `memory.learn` call `Store()` with no arg → default `cfg.collection`. Thread an optional
   `collection` param through these + a CLI `--collection` flag. Bounded, not deep. **Do first.**
2. **Streaming is new.** `llm.answer()` (`llm.py:31`) is a single non-streaming
   `chat.completions.create()`. Add `answer_stream()` (`stream=True`) + SSE. Small but real.
3. **Availability.** The demo depends on Alienware (Qdrant+Ollama) and the Mac mini being up.
   Without the `/health` + cached-fallback path, the demo is dead when the lab sleeps — bad for
   recruiters. Treat the fallback as in-scope, not nice-to-have.
4. **Sandbox cleanup correctness.** Orphaned `sandbox-*` collections leak Qdrant storage on the
   Alienware. The TTL reaper must be reliable (and idempotent).
5. **Public-corpus data hygiene.** History once contained a comp band; audit every doc before it
   enters `second_brain_public`. One leaked private line in a public demo is the worst outcome.

---

## 5. Phasing (owner-mode first, sandbox later — per owner)

- **Phase 0 — Foundations.** Thread `collection` through `ask`/`ingest`/`learn` + CLI flag; add
  `llm.answer_stream()`; seed `second_brain_public` (audited). Tests stay green. *(backend-only)*
- **Phase 1 — Backend read API.** FastAPI `/health` `/status` `/ask`(+SSE) `/recall`; routing for
  `public`/`neutral`; no writes yet. Tunnel up on `brain-api.artjeck.com`. Tests.
- **Phase 2 — Frontend MVP (anonymous, read-only).** Standalone Next app: chat + citations panel +
  corpus switcher (Lab/Neutral) + suggested prompts + Turnstile + rate limit + availability
  fallback. Deploy `brain.artjeck.com`. **← recruiter-shippable milestone; flip roadmap to in-progress.**
- **Phase 3 — Owner mode.** NextAuth single-owner; expose ingest/learn/tasks/eval-traces against
  `real`; `collection=real` gated to owner. The daily-driver UI.
- **Phase 4 — Anonymous sandbox.** Ephemeral `sandbox-{sid}`; paste/upload ingest; learn/tasks
  in-session; TTL reaper + caps. The full public interactive loop.
- **Phase 5 — Hardening & launch.** Security review of the write surface; load/abuse test; polish;
  `docs/WEB_UI.md`; link from artjeck.com Lab section; flip roadmap to done.

The smallest thing that produces a live, linkable, honest demo is **Phases 0→2**. Everything after
is capability the owner gets; Phase 2 is the portfolio win.

---

## 6. Owner actions / open items (not code)
- Cloudflare: create the `brain-api.artjeck.com` tunnel hostname + the `brain.artjeck.com` site domain.
- Vercel: new project for `second-brain/web`; set backend URL + token + Turnstile keys as env.
- Decide owner-auth mechanism (single password vs GitHub OAuth restricted to `ArtJack`).
- Approve the public corpus doc list before seeding.

## 7. Rough effort (Opus-paced, excludes owner actions)
- Phase 0: ~½ day · Phase 1: ~1 day · Phase 2: ~1–2 days (UI polish dominates) ·
  Phase 3: ~1 day · Phase 4: ~1–2 days (lifecycle/abuse) · Phase 5: ~½–1 day.
- **To a live recruiter demo (0→2): ~2.5–3.5 days.** Full product (0→5): ~5–7 days.
