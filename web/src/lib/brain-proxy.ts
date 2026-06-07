import { NextRequest, NextResponse } from "next/server";

const RATE_LIMIT_MAX = 36;
const RATE_LIMIT_WINDOW_MS = 60_000;
const MAX_TEXT_LENGTH = 4000;
const recentRequests = new Map<string, number[]>();

const READ_GET = new Set(["health", "status"]);
const READ_POST = new Set(["ask", "ask/stream", "recall"]);
const ALLOWED_CORPORA = new Set(["public", "neutral"]);

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

function validateBody(endpoint: string, body: Record<string, unknown>) {
  const corpus = body.corpus;
  if (typeof corpus === "string" && !ALLOWED_CORPORA.has(corpus)) {
    return "This demo only exposes the Lab and Neutral corpora.";
  }
  const text = endpoint === "recall" ? body.query : body.question;
  if (typeof text !== "string" || text.trim().length === 0) {
    return endpoint === "recall" ? "Query is required." : "Question is required.";
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return `Please keep input under ${MAX_TEXT_LENGTH} characters.`;
  }
  return null;
}

export async function proxyBrain(request: NextRequest, path: string[], method: "GET" | "POST") {
  const endpoint = path.join("/");
  if ((method === "GET" && !READ_GET.has(endpoint)) || (method === "POST" && !READ_POST.has(endpoint))) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  if (isRateLimited(clientIp(request), endpoint)) {
    return NextResponse.json({ error: "Too many requests. Please wait a minute." }, { status: 429 });
  }

  const baseUrl = process.env.BRAIN_API_URL?.replace(/\/$/, "");
  const token = process.env.BRAIN_API_TOKEN;
  if (!baseUrl || !token) {
    return unavailable();
  }

  let bodyText: string | undefined;
  if (method === "POST") {
    const turnstile = await verifyTurnstile(request);
    if (!turnstile.ok) {
      return NextResponse.json({ error: turnstile.message }, { status: turnstile.status });
    }

    const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body) {
      return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
    }
    const bodyError = validateBody(endpoint, body);
    if (bodyError) {
      return NextResponse.json({ error: bodyError }, { status: 400 });
    }
    bodyText = JSON.stringify(body);
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
