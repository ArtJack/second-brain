# The public demo's corpus boundary

Four corpora: public, neutral, real, sandbox. Anonymous traffic may reach the
public and neutral ones only.

The boundary fails closed. An unauthenticated request for the real collection is
401 or 403, never a degraded answer from a different corpus, because a silent
downgrade is indistinguishable from success to the person reading the answer.

Turnstile fronts the demo. The tunnel terminates at Cloudflare and the origin is
not reachable directly.
