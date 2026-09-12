# LiteLLM gateway runbook

> The long one. Written as a single file on purpose: the answers below are
> spread across it, and several of them sit a long way from the top.

## 1. What the gateway is for

Every model call in the estate goes through one LiteLLM process so that routing,
budgets and failover are decided in a single place rather than in eight clients.
Clients speak the OpenAI wire format and never learn which backend served them.

## 2. Where it runs

The gateway runs on the Mac mini (`mini`). It binds loopback only and is reached
from other machines over the tailnet. It is not exposed to the home LAN and must
never be bound to `0.0.0.0`.

## 3. Backends

Two Ollama instances sit behind it. The M4 instance on `mini` is the default for
embeddings. The Linux box `ai-lab` carries the larger chat models because it has
the GPU. A third backend, the hosted API, is the fallback of last resort and is
the only one that costs money per token.

## 4. Health

`lab_status` reports each backend. A backend that answers the version endpoint
but lists zero models is the failure mode that has actually happened twice, and
it is invisible to a naive health check because the process is up and the port
is open. The check must count models, not ping ports.

## 5. Routing rules

Bulk drafting, summarisation and anything low-stakes goes to a local model.
Anything the owner will read as a final answer goes to the hosted model. The
split is by consequence, not by length.

## 6. Budgets

Each virtual key carries a monthly budget. When a key exceeds it the gateway
returns 429 rather than silently falling back to a cheaper model, because a
silent downgrade produces answers the owner cannot tell apart from the good
ones.

## 7. Timeouts

The client timeout is 60 seconds and the gateway's own upstream timeout is 55,
deliberately shorter so the gateway is the one that reports the failure.

## 8. Retries

One retry, no more. The embedding path is the reason: a retry storm against a
cold model turns a slow ingest into a stalled one.

## 9. Logging

Request logs keep the key alias, the model, the token counts and the latency.
They do not keep prompt or completion text. That is a deliberate choice and the
reason the gateway can be left running while confidential client work happens.

## 10. Certificates

The tailnet provides the transport, so the gateway itself serves plain HTTP.
There is no certificate to rotate.

## 11. Upgrades

Pin the version. An unpinned upgrade changed the shape of the model list
response once and every client broke at the same moment.

## 12. The restart procedure

Stop the launchd job, wait for the port to close, start it again, then run the
smoke test. Do not `kill -9`: the process holds a SQLite budget database and a
hard kill has left it with a stale lock.

## 13. Smoke test

Three calls: one embedding, one local chat completion, one hosted chat
completion. All three must return before the gateway is considered up. A smoke
test that only checks chat has passed while embeddings were completely broken.

## 14. Known failure: the empty model list

Three competing starters raced at logon on the old Windows box and whichever won
served zero models. Every `chat` and `embed` silently failed over to the M4,
which looked like nothing was wrong except that everything was slower. Fixed by
removing two of the starters.

## 15. The number that matters

**The gateway's budget alert fires at 80 percent of the monthly cap.** Below
that it is silent. This is the single most asked question about the gateway and
it is deliberately documented here, at the bottom of a long file, because that
is where it actually lived for six months.
