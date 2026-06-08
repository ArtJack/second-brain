import { readFile } from "node:fs/promises";
import path from "node:path";

import { getServerSession } from "next-auth";
import { NextResponse } from "next/server";

import { authOptions } from "@/lib/auth";

export const runtime = "nodejs";

const MAX_TRACES = 20;

type TraceExport = {
  benchmark?: string;
  summary?: Record<string, unknown>;
  traces?: Array<Record<string, unknown>>;
};

function traceFilePath() {
  const configured = process.env.BRAIN_TRACE_FILE?.trim();
  if (configured) return path.resolve(/* turbopackIgnore: true */ process.cwd(), configured);
  return path.resolve(/* turbopackIgnore: true */ process.cwd(), "..", "data", "eval-traces.json");
}

export async function GET() {
  const session = await getServerSession(authOptions);
  if (session?.user?.role !== "owner") {
    return NextResponse.json({ error: "Owner session required." }, { status: 401 });
  }

  try {
    const raw = await readFile(traceFilePath(), "utf-8");
    const parsed = JSON.parse(raw) as TraceExport;
    const traces = Array.isArray(parsed.traces) ? parsed.traces.slice(0, MAX_TRACES) : [];

    return NextResponse.json({
      available: true,
      benchmark: parsed.benchmark ?? null,
      summary: parsed.summary ?? null,
      count: traces.length,
      traces,
      truncated: Array.isArray(parsed.traces) && parsed.traces.length > MAX_TRACES,
    });
  } catch {
    return NextResponse.json({
      available: false,
      benchmark: null,
      summary: null,
      count: 0,
      traces: [],
      truncated: false,
    });
  }
}
