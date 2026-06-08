import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";

import { authOptions } from "@/lib/auth";

const RATE_LIMIT_MAX = 36;
const RATE_LIMIT_WINDOW_MS = 60_000;
const MAX_TEXT_LENGTH = 4000;
const MAX_INGEST_LENGTH = 100_000;
const MAX_LEARN_LENGTH = 20_000;
const recentRequests = new Map<string, number[]>();

const READ_GET = new Set(["health", "status"]);
const OWNER_GET = new Set(["tasks"]);
const READ_POST = new Set(["ask", "ask/stream", "recall"]);
const OWNER_POST = new Set(["ingest", "learn", "tasks"]);
const PUBLIC_CORPORA = new Set(["public", "neutral"]);
const OWNER_CORPORA = new Set(["public", "neutral", "real"]);

type JsonBody = Record<string, unknown>;

function clientIp(request: NextRequest): string {
  const fwd = request.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0]?.trim() || "unknown";
  return request.headers.get("x-real-ip")?.trim() || "unknown";
}

function isRateLimited(ip: string, route: string): boolean {
  const key = `${ip}:${route}`;
  const now = Date.now();
  const cutoff = now - RATE_LIMIT_WINDOW_MS;
  const recent = (recentRequests.get(key) ?? []).filter((time) => time > cutoff);
  if (recent.length >= RATE_LIMIT_MAX) {
    recentRequests.set(key, recent);
    return true;
  }
  recent.push(now);
  recentRequests.set(key, recent);
  return false;
}

async function verifyTurnstile(request: NextRequest): Promise<{ ok: true } | { ok: false; status: number; message: string }> {
  const secret = process.env.TURNSTILE_SECRET_KEY;
  if (!secret) {
    if (process.env.NODE_ENV === "production") {
      return { ok: false, status: 503, message: "Turnstile is not configured." };
    }
    return { ok: true };
  }

  const token = request.headers.get("x-turnstile-token");
  if (!token) {
    return { ok: false, status: 403, message: "Turnstile verification required." };
  }

  const body = new FormData();
  body.append("secret", secret);
  body.append("response", token);
  body.append("remoteip", clientIp(request));

  const response = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body,
  });
  const result = (await response.json().catch(() => null)) as { success?: boolean } | null;
  if (!result?.success) {
    return { ok: false, status: 403, message: "Turnstile verification failed." };
  }
  return { ok: true };
}

function unavailable() {
  return NextResponse.json(
    { error: "brain API unavailable" },
    { status: 503 },
  );
}

function isTaskComplete(endpoint: string) {
  return /^tasks\/[^/]+\/complete$/.test(endpoint);
}

function isKnownRoute(endpoint: string, method: "GET" | "POST") {
  if (method === "GET") return READ_GET.has(endpoint) || OWNER_GET.has(endpoint);
  return READ_POST.has(endpoint) || OWNER_POST.has(endpoint) || isTaskComplete(endpoint);
}

function isOwnerRoute(endpoint: string, method: "GET" | "POST") {
  if (method === "GET") return OWNER_GET.has(endpoint);
  return OWNER_POST.has(endpoint) || isTaskComplete(endpoint);
}

function authError() {
  return NextResponse.json({ error: "Owner session required." }, { status: 401 });
}

function validateCorpus(corpus: unknown, isOwner: boolean) {
  if (corpus === undefined) return null;
  if (typeof corpus !== "string") return { status: 400, message: "Corpus must be a string." };
  const allowed = isOwner ? OWNER_CORPORA : PUBLIC_CORPORA;
  if (!allowed.has(corpus)) {
    return {
      status: isOwner ? 400 : 403,
      message: isOwner ? "Unknown corpus." : "This demo only exposes the Lab and Neutral corpora.",
    };
  }
  return null;
}

function validateSearchCorpus(request: NextRequest, isOwner: boolean) {
  return validateCorpus(request.nextUrl.searchParams.get("corpus") ?? undefined, isOwner);
}

function validateReadBody(endpoint: string, body: JsonBody, isOwner: boolean) {
  const corpus = body.corpus;
  const corpusError = validateCorpus(corpus, isOwner);
  if (corpusError) {
    return corpusError;
  }
  const text = endpoint === "recall" ? body.query : body.question;
  if (typeof text !== "string" || text.trim().length === 0) {
    return { status: 400, message: endpoint === "recall" ? "Query is required." : "Question is required." };
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return { status: 400, message: `Please keep input under ${MAX_TEXT_LENGTH} characters.` };
  }
  return null;
}

function validateOwnerBody(endpoint: string, body: JsonBody): { ok: true; body: JsonBody } | { ok: false; message: string } {
  if (endpoint === "ingest") {
    if (typeof body.text !== "string" || body.text.trim().length === 0) {
      return { ok: false, message: "Text is required." };
    }
    if (body.text.length > MAX_INGEST_LENGTH) {
      return { ok: false, message: `Please keep ingest text under ${MAX_INGEST_LENGTH} characters.` };
    }
    if (body.source !== undefined && (typeof body.source !== "string" || body.source.trim().length === 0 || body.source.length > 120)) {
      return { ok: false, message: "Source must be a non-empty string under 120 characters." };
    }
    return {
      ok: true,
      body: {
        text: body.text,
        source: typeof body.source === "string" ? body.source : "web-ingest.md",
        corpus: "real",
      },
    };
  }

  if (endpoint === "learn") {
    if (typeof body.text !== "string" || body.text.trim().length === 0) {
      return { ok: false, message: "Text is required." };
    }
    if (body.text.length > MAX_LEARN_LENGTH) {
      return { ok: false, message: `Please keep learned memory under ${MAX_LEARN_LENGTH} characters.` };
    }
    if (body.source !== undefined && (typeof body.source !== "string" || body.source.trim().length === 0 || body.source.length > 120)) {
      return { ok: false, message: "Source must be a non-empty string under 120 characters." };
    }
    return {
      ok: true,
      body: {
        text: body.text,
        source: typeof body.source === "string" ? body.source : "web",
        corpus: "real",
      },
    };
  }

  if (endpoint === "tasks") {
    if (typeof body.title !== "string" || body.title.trim().length === 0) {
      return { ok: false, message: "Task title is required." };
    }
    if (body.title.length > 500) {
      return { ok: false, message: "Task title must stay under 500 characters." };
    }
    if (body.notes !== undefined && (typeof body.notes !== "string" || body.notes.length > 5000)) {
      return { ok: false, message: "Task notes must stay under 5000 characters." };
    }
    return {
      ok: true,
      body: {
        title: body.title,
        notes: typeof body.notes === "string" ? body.notes : "",
      },
    };
  }

  return { ok: false, message: "Unsupported owner write route." };
}

export async function proxyBrain(request: NextRequest, path: string[], method: "GET" | "POST") {
  const endpoint = path.join("/");
  if (!isKnownRoute(endpoint, method)) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  if (isRateLimited(clientIp(request), endpoint)) {
    return NextResponse.json({ error: "Too many requests. Please wait a minute." }, { status: 429 });
  }

  const session = await getServerSession(authOptions);
  const isOwner = session?.user?.role === "owner";
  const ownerRoute = isOwnerRoute(endpoint, method);
  if (ownerRoute && !isOwner) {
    return authError();
  }

  if (method === "GET") {
    const corpusError = validateSearchCorpus(request, isOwner);
    if (corpusError) {
      return NextResponse.json({ error: corpusError.message }, { status: corpusError.status });
    }
  }

  const baseUrl = process.env.BRAIN_API_URL?.replace(/\/$/, "");
  const token = isOwner ? (process.env.BRAIN_OWNER_TOKEN || process.env.BRAIN_API_TOKEN) : process.env.BRAIN_API_TOKEN;
  if (!baseUrl || !token) {
    return unavailable();
  }

  let bodyText: string | undefined;
  if (method === "POST") {
    if (ownerRoute) {
      if (isTaskComplete(endpoint)) {
        bodyText = undefined;
      } else {
        const body = (await request.json().catch(() => null)) as JsonBody | null;
        if (!body) {
          return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
        }
        const bodyResult = validateOwnerBody(endpoint, body);
        if (!bodyResult.ok) {
          return NextResponse.json({ error: bodyResult.message }, { status: 400 });
        }
        bodyText = JSON.stringify(bodyResult.body);
      }
    } else {
      if (!isOwner) {
        const turnstile = await verifyTurnstile(request);
        if (!turnstile.ok) {
          return NextResponse.json({ error: turnstile.message }, { status: turnstile.status });
        }
      }

      const body = (await request.json().catch(() => null)) as JsonBody | null;
      if (!body) {
        return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
      }
      const bodyError = validateReadBody(endpoint, body, isOwner);
      if (bodyError) {
        return NextResponse.json({ error: bodyError.message }, { status: bodyError.status });
      }
      bodyText = JSON.stringify(body);
    }
  }

  const search = method === "GET" ? request.nextUrl.search : "";
  const response = await fetch(`${baseUrl}/${endpoint}${search}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(bodyText ? { "Content-Type": "application/json" } : {}),
    },
    body: bodyText,
    cache: "no-store",
  });

  if (endpoint === "ask/stream") {
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") ?? "text/event-stream",
        "Cache-Control": "no-store",
      },
    });
  }

  const text = await response.text();
  return new Response(text, {
    status: response.status,
    headers: {
      "Content-Type": response.headers.get("content-type") ?? "application/json",
      "Cache-Control": "no-store",
    },
  });
}
