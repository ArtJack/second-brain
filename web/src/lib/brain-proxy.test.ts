/**
 * The proxy is the corpus boundary, and it had no tests at all. (Verdict SB-F-14.)
 *
 * Everything anonymous traffic is allowed to reach passes through this one
 * function. The CI job says as much in a comment: until this file existed, a
 * typecheck was the only automated statement about the code holding the gate.
 * A typecheck cannot tell you that an anonymous visitor is refused the real
 * corpus — only that the refusal, if written, has the right types.
 *
 * These tests are about the boundary failing *closed*. Every case below is one
 * where the wrong answer is not an error message but a quiet success: the demo
 * answering a stranger out of the owner's private notes, an owner token sent on
 * an anonymous request, a route nobody meant to expose being proxied because it
 * was not on a list.
 */
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSession = vi.hoisted(() => vi.fn());
vi.mock("next-auth", () => ({ getServerSession }));
vi.mock("@/lib/auth", () => ({ authOptions: {} }));

const { proxyBrain } = await import("@/lib/brain-proxy");

const OWNER = { user: { role: "owner" } };
const SIGNED_IN_NOT_OWNER = { user: { role: "visitor" } };

let fetchMock: ReturnType<typeof vi.fn>;
let ip = 0;

/** A fresh IP per request, so the module-level rate limiter cannot leak between tests. */
function request(url: string, init?: RequestInit & { ip?: string }): NextRequest {
  const headers = new Headers(init?.headers);
  headers.set("x-forwarded-for", init?.ip ?? `10.0.0.${++ip % 250}`);
  return new NextRequest(new Request(url, { ...init, headers }));
}

function post(url: string, body: unknown, init: RequestInit & { ip?: string } = {}) {
  return request(url, { ...init, method: "POST", body: JSON.stringify(body) });
}

beforeEach(() => {
  getServerSession.mockResolvedValue(null);
  process.env.BRAIN_API_URL = "https://brain.invalid";
  process.env.BRAIN_API_TOKEN = "public-token";
  process.env.BRAIN_OWNER_TOKEN = "owner-token";
  delete process.env.TURNSTILE_SECRET_KEY;
  fetchMock = vi.fn(async () => new Response('{"ok":true}', { headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("the corpus boundary", () => {
  it("refuses an anonymous visitor the real corpus, and does not call the brain", async () => {
    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "what?", corpus: "real" }), ["ask"], "POST");

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses a signed-in visitor who is not the owner", async () => {
    getServerSession.mockResolvedValue(SIGNED_IN_NOT_OWNER);

    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "what?", corpus: "real" }), ["ask"], "POST");

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allows the owner the real corpus", async () => {
    getServerSession.mockResolvedValue(OWNER);

    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "what?", corpus: "real" }), ["ask"], "POST");

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it.each(["public", "neutral"])("allows an anonymous visitor the %s corpus", async (corpus) => {
    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "what?", corpus }), ["ask"], "POST");

    expect(res.status).toBe(200);
  });

  it("refuses a corpus nobody defined rather than passing it through", async () => {
    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q", corpus: "sandbox" }), ["ask"], "POST");

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("checks the corpus on a GET query string too", async () => {
    const res = await proxyBrain(request("https://x/api/brain/status?corpus=real"), ["status"], "GET");

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses a non-string corpus instead of coercing it", async () => {
    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q", corpus: ["real"] }), ["ask"], "POST");

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("owner-only routes", () => {
  it.each([
    ["ingest", "POST"],
    ["learn", "POST"],
    ["tasks", "POST"],
    ["tasks", "GET"],
  ] as const)("refuses anonymous %s %s", async (endpoint, method) => {
    const req = method === "GET" ? request(`https://x/api/brain/${endpoint}`) : post(`https://x/api/brain/${endpoint}`, { text: "x", title: "x" });

    const res = await proxyBrain(req, [endpoint], method);

    expect(res.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses completing someone else's task anonymously", async () => {
    const res = await proxyBrain(post("https://x/api/brain/tasks/17/complete", {}), ["tasks", "17", "complete"], "POST");

    expect(res.status).toBe(401);
  });
});

describe("the route allow-list", () => {
  it.each([
    ["gc", "POST"],
    ["forget", "POST"],
    ["admin", "GET"],
    ["tasks/17/complete/../../../health", "GET"],
  ] as const)("404s %s rather than proxying it", async (endpoint, method) => {
    const res = await proxyBrain(request(`https://x/api/brain/${endpoint}`, { method }), endpoint.split("/"), method);

    expect(res.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("is a list of exact names, so a route that merely starts with a known one is refused", async () => {
    const res = await proxyBrain(post("https://x/api/brain/askanything", { question: "q" }), ["askanything"], "POST");

    expect(res.status).toBe(404);
  });
});

describe("which token leaves the building", () => {
  it("sends the public token for an anonymous request", async () => {
    await proxyBrain(post("https://x/api/brain/ask", { question: "q" }), ["ask"], "POST");

    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe("Bearer public-token");
  });

  it("sends the owner token only for an owner session", async () => {
    getServerSession.mockResolvedValue(OWNER);

    await proxyBrain(post("https://x/api/brain/ask", { question: "q" }), ["ask"], "POST");

    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe("Bearer owner-token");
  });

  it("fails closed with 503 when the brain is not configured, rather than calling nothing and returning 200", async () => {
    delete process.env.BRAIN_API_URL;

    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q" }), ["ask"], "POST");

    expect(res.status).toBe(503);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards the visitor's IP, because the origin otherwise sees one bucket for everyone", async () => {
    await proxyBrain(post("https://x/api/brain/ask", { question: "q" }, { ip: "203.0.113.9" }), ["ask"], "POST");

    expect(fetchMock.mock.calls[0][1].headers["X-Visitor-IP"]).toBe("203.0.113.9");
  });

  it("takes the first address from a forwarded chain, not the whole header", async () => {
    await proxyBrain(post("https://x/api/brain/ask", { question: "q" }, { ip: "203.0.113.9, 70.41.3.18" }), ["ask"], "POST");

    expect(fetchMock.mock.calls[0][1].headers["X-Visitor-IP"]).toBe("203.0.113.9");
  });
});

describe("Turnstile", () => {
  it("is required in production even when unconfigured, rather than silently skipped", async () => {
    const previous = process.env.NODE_ENV;
    vi.stubEnv("NODE_ENV", "production");

    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q" }), ["ask"], "POST");

    expect(res.status).toBe(503);
    vi.stubEnv("NODE_ENV", previous ?? "test");
  });

  it("rejects an ask with no token once a secret is configured", async () => {
    process.env.TURNSTILE_SECRET_KEY = "secret";

    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q" }), ["ask"], "POST");

    expect(res.status).toBe(403);
  });

  it("does not spend the token on recall, because the token is single-use and a question calls both", async () => {
    // The regression this encodes: verifying both burned the token on recall
    // and 403'd the ask that followed. Invisible under the always-pass test
    // keys, fatal under real ones.
    process.env.TURNSTILE_SECRET_KEY = "secret";

    const res = await proxyBrain(post("https://x/api/brain/recall", { query: "q" }), ["recall"], "POST");

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toContain("/recall");
  });

  it("does not challenge the owner", async () => {
    process.env.TURNSTILE_SECRET_KEY = "secret";
    getServerSession.mockResolvedValue(OWNER);

    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q" }), ["ask"], "POST");

    expect(res.status).toBe(200);
  });
});

describe("input limits", () => {
  it("rejects an empty question before spending a model call", async () => {
    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "   " }), ["ask"], "POST");

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects an oversized question", async () => {
    const res = await proxyBrain(post("https://x/api/brain/ask", { question: "x".repeat(4001) }), ["ask"], "POST");

    expect(res.status).toBe(400);
  });

  it("rejects a body that is not JSON at all", async () => {
    const req = request("https://x/api/brain/ask", { method: "POST", body: "not json" });

    const res = await proxyBrain(req, ["ask"], "POST");

    expect(res.status).toBe(400);
  });

  it("forwards only the fields it validated, never the caller's whole body", async () => {
    getServerSession.mockResolvedValue(OWNER);

    await proxyBrain(
      post("https://x/api/brain/ingest", { text: "note", source: "s.md", corpus: "real", extra: "smuggled" }),
      ["ingest"],
      "POST",
    );

    const sent = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(sent).toEqual({ text: "note", source: "s.md", corpus: "real" });
    expect(sent.extra).toBeUndefined();
  });

  it("pins an owner write to the real corpus whatever the caller asked for", async () => {
    getServerSession.mockResolvedValue(OWNER);

    await proxyBrain(post("https://x/api/brain/learn", { text: "fact", corpus: "public" }), ["learn"], "POST");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body).corpus).toBe("real");
  });
});

describe("rate limiting", () => {
  it("gives the expensive ask route a tighter budget than the cheap polls", async () => {
    const caller = "198.51.100.7";
    const statuses: number[] = [];
    for (let i = 0; i < 8; i++) {
      const res = await proxyBrain(post("https://x/api/brain/ask", { question: "q" }, { ip: caller }), ["ask"], "POST");
      statuses.push(res.status);
    }

    expect(statuses.filter((s) => s === 429).length).toBeGreaterThan(0);
    expect(statuses.slice(0, 6).every((s) => s === 200)).toBe(true);

    // The same caller is still well inside the budget for a cheap route.
    const health = await proxyBrain(request("https://x/api/brain/health", { ip: caller }), ["health"], "GET");
    expect(health.status).toBe(200);
  });

  it("counts each caller separately", async () => {
    for (let i = 0; i < 7; i++) {
      await proxyBrain(post("https://x/api/brain/ask", { question: "q" }, { ip: "198.51.100.8" }), ["ask"], "POST");
    }

    const other = await proxyBrain(post("https://x/api/brain/ask", { question: "q" }, { ip: "198.51.100.9" }), ["ask"], "POST");

    expect(other.status).toBe(200);
  });
});
