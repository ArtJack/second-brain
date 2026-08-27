# ArtJeck Technology — Site Concierge Knowledge Base

> This is the grounding corpus for the artjeck.com site assistant ("concierge").
> The assistant answers visitor questions ONLY from this document and cites it.
> Prices here are **anchors and typical ranges** — the assistant must never
> invent or finalize a price; for anything custom it collects the visitor's
> email for a quote. Keep this in sync with the live site (`artjeck-technology/
> src/lib/site-data.ts`) — the six "Starting at" anchors below must match the
> service cards exactly.

---

## About ArtJeck Technology

ArtJeck builds **production-grade AI systems with quality engineering built in
from day one**. The core idea: most AI projects die in the gap between a working
prototype and something that holds up under real users. ArtJeck treats AI
engineering, full-stack development, and software QA as **one build path**, not
three separate hand-offs — so what ships has evals, edge-case tests, and
regression checks already in it.

- **Founder/engineer:** Eugene (ArtJack).
- **Based in:** Sacramento, CA.
- **How work is delivered:** AI automation, web/iOS, and QA are done **remotely
  for clients across the US**. **Computer repair is local, Sacramento area only.**
- **Contact:** hello@artjeck.com · Telegram @artjeck · GitHub github.com/ArtJack

---

## Services & pricing

All project prices are **starting points**; final quotes follow a short scoping
conversation. The six anchors below match the website's service cards.

### 1. AI Automation Engineering — starting at **$8,000 / project**
AI agents, RAG systems, prompt engineering, workflow automation, model
evaluations, fine-tuning, and business-process automation — shipped production-
ready with evals and edge-case testing.
- Single-workflow automation — typically from **$2,500**
- RAG system / AI agent build — typically from **$8,000**
- AI feature + evaluation & QA harness — typically from **$12,000**
- Ongoing retainer (maintenance + iterations) — typically from **$1,500/mo**
- Advisory / consulting — **$150/hr**

### 2. Web & iOS Product Development — starting at **$1,500 / project**
Modern websites, web apps, dashboards, APIs, iOS apps, and MVPs — built for real
users and tested before they ship.
- Landing / brochure site — typically from **$1,500**
- Business site + CMS — typically from **$3,500**
- Web app / dashboard — typically from **$8,000**
- Web apps & MVPs — from **$7,500**
- Care & hosting plan — typically from **$75/mo**

### 3. Software Quality Engineering — starting at **$3,000 / project**
SDLC/STLC-based testing: functional, regression, smoke/sanity, test-design
techniques, edge-case validation, and release confidence.
- Test plan + regression suite — typically from **$3,000**
- Release-readiness audit — typically from **$1,500**
- Ongoing QA retainer — typically from **$1,000/mo**
- Hourly — **$90/hr**

### 4. Local Computer Repair — **Sacramento, CA area only**
Honest, flat-rate diagnostics and repair for homes and small businesses.

**No repair prices are published, and the assistant must not quote one.** Every
repair job is quoted individually, with a fixed price agreed before work starts
— this matches the website, where every repair page says the same and CI fails
the build if a dollar amount appears on one. Route any "how much is X" repair
question to a quote: take the problem and the visitor's email, and say Eugene
follows up with a fixed price, usually within one business day.

(A flat-rate list was briefly recorded here on 2026-08-26 and removed the same
day — it contradicted the published policy above. If you ever decide the
assistant may give ballpark repair pricing, that is a deliberate change to make
on the website first, not here.)

### 5. Data, Databases & Integrations — starting at **$1,500 / project**
Databases set up, migrated and connected; API and tool integrations; PDF and
document extraction; spreadsheets turned into real systems.
- Fixed scope — from **$1,500**
- Hourly after — **$90/hr**

### 6. SEO & Analytics — starting at **$1,000 / project**
Technical and local SEO, structured data, GA4 and Search Console setup,
analytics dashboards, Core Web Vitals.
- Audit + fixes — from **$1,000**
- Hourly after — **$90/hr**

---

## Proof of work (recent projects)

- **IFTA Agent** — a quarterly fuel-tax filing pipeline for trucking carriers.
  Ingests raw mileage/fuel files, computes a state-portal-ready return with exact
  CDTFA math, and runs a 17-tool AI agent to flag anomalies before filing. Python
  + Claude. **442 automated tests**; a real-data backtest matched a Kentucky
  carrier's filing to the penny. Live with a paying client.
  Case study: https://artjeck.com/work/ifta
- **Verdict** — an open-source QA agent for Claude Code, MIT licensed, at
  https://github.com/ArtJack/verdict. It keeps a baseline so a repeat run is a
  delta rather than a fresh audit, quarantines flaky tests only with an expiry
  date attached, and has no ability to edit the code it judges. Ships with its
  own published eval: 8 seeded defects across 5 failure classes, scored 8/8 on
  the first run — self-scored on a single run, which the repo states next to the
  number. Case study: https://artjeck.com/work/verdict
- **Greek-Scythian Society CIC** — a website for a UK heritage nonprofit, live at
  https://greek-scythian.org. Built twice on purpose: a bespoke WordPress phase
  first, then a static Next.js rebuild that kept the design and removed the
  servers the client would otherwise have to maintain. 23 pages, three runtime
  dependencies, every original URL preserved.
  Case study: https://artjeck.com/work/greek-scythian
- **second-brain** — a local-first RAG assistant that answers only from your own
  notes/docs/code, with every claim cited. Free local models by default. (This
  is the engine behind the demo and this very chat.)
- **lab-control** — an MCP control plane that gives an AI agent safe, gated
  "hands" on a private machine lab; runs 24/7.
- **Self-hosted email agent** — triages a mailbox with a local model and sends a
  Telegram digest, at $0 API cost.
- **Liora Studio** — a full Next.js e-commerce storefront with a 2FA admin CMS.
- **DM Express** — a fast, mobile-tuned trucking recruiting site (React/Vite,
  29 tests, sub-500KB first load).

---

## FAQ

**Do you work remotely?** Yes — AI automation, web/iOS, and QA are delivered
remotely for clients across the US. Computer repair is local to Sacramento only.

**Where are you based?** Sacramento, California.

**How do we get started?** Tell the assistant what you're trying to build or fix
and leave your email — Eugene follows up, usually within one business day, with a
fixed quote.

**How much will my project cost?** The starting points are above. The final price
depends on scope — share a few details and your email and you'll get a real quote.

**What makes ArtJeck different?** QA is built in, not bolted on. You get AI
features that are tested and production-ready, from someone who handles the AI,
the app, and the quality engineering as one path.

**Do you offer ongoing support?** Yes — retainers and care/hosting plans for AI
systems, web apps, and QA.

**Is this chat a real AI?** Yes — it runs on ArtJeck's own always-on home lab
(local models, $0 per-message API cost), grounded in this knowledge base. It's a
live example of the kind of system Eugene builds for clients.

**Is my conversation private?** The assistant runs on private, self-hosted
hardware. Don't share sensitive personal data in chat; for anything private,
email hello@artjeck.com.

**Can you build / integrate [specific tool]?** Probably — describe it and leave
your email, and Eugene will tell you exactly how he'd approach it.

---

## Answer policy (guardrails — also encode in the system prompt)

1. **Scope:** Only answer questions about ArtJeck — its services, pricing,
   process, and projects. For off-topic questions, briefly redirect to what
   ArtJeck does.
2. **Never invent or finalize prices.** Quote only the anchors/ranges above, and
   frame them as starting points. For custom scope, collect the visitor's email
   for a quote instead of guessing a number.
3. **Always move toward the next step:** capturing the visitor's email and what
   they need, so Eugene can follow up.
4. **Hand off gracefully:** for anything you can't answer or that needs a human,
   point to hello@artjeck.com or offer to take their email.
5. **Be honest about identity:** you are ArtJeck's AI assistant, not Eugene.
6. **Cite** the relevant section of this knowledge base when answering factual
   questions about services or pricing.
