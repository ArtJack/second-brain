"use client";

import {
  AlertTriangle,
  CheckCircle2,
  CheckSquare,
  FileUp,
  ListChecks,
  Loader2,
  Plus,
  RefreshCw,
  ScrollText,
  ShieldCheck,
} from "lucide-react";
import type { FormEvent } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";

type OwnerTab = "ingest" | "learn" | "tasks" | "traces";
type BusyState = OwnerTab | "complete" | null;

type BrainTask = {
  id: number;
  title: string;
  notes?: string;
  status: string;
  created_at?: string;
  completed_at?: string | null;
};

type TasksResponse = {
  count: number;
  tasks: BrainTask[];
};

type TraceStep = {
  name?: string;
  kind?: string;
  duration_ms?: number;
  status?: string;
};

type TraceRecord = {
  trace_id?: string;
  name?: string;
  status?: string;
  duration_ms?: number;
  attributes?: Record<string, unknown>;
  trajectory?: TraceStep[];
  spans?: TraceStep[];
};

type TracesResponse = {
  available: boolean;
  benchmark: string | null;
  summary: Record<string, unknown> | null;
  count: number;
  traces: TraceRecord[];
  truncated: boolean;
};

type ActionResult = {
  chunks?: number;
  total?: number;
  collection?: string;
  memory_file?: string;
};

const tabs: Array<{ key: OwnerTab; label: string; icon: typeof FileUp }> = [
  { key: "ingest", label: "Ingest", icon: FileUp },
  { key: "learn", label: "Learn", icon: ShieldCheck },
  { key: "tasks", label: "Tasks", icon: ListChecks },
  { key: "traces", label: "Traces", icon: ScrollText },
];

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  const data = (await response.json().catch(() => null)) as (T & { error?: string; detail?: string }) | null;
  if (!response.ok) {
    throw new Error(data?.error ?? data?.detail ?? "Request failed.");
  }
  if (!data) {
    throw new Error("Invalid response.");
  }
  return data;
}

function formatSummaryValue(value: unknown) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  return "n/a";
}

export function OwnerPanel({ onMutation }: { onMutation: () => void }) {
  const [activeTab, setActiveTab] = useState<OwnerTab>("ingest");
  const [busy, setBusy] = useState<BusyState>(null);
  const [notice, setNotice] = useState<{ tone: "ok" | "warn"; text: string } | null>(null);
  const [ingestSource, setIngestSource] = useState("owner-web-ingest.md");
  const [ingestText, setIngestText] = useState("");
  const [learnSource, setLearnSource] = useState("web");
  const [learnText, setLearnText] = useState("");
  const [taskTitle, setTaskTitle] = useState("");
  const [taskNotes, setTaskNotes] = useState("");
  const [tasks, setTasks] = useState<BrainTask[]>([]);
  const [traces, setTraces] = useState<TracesResponse | null>(null);

  const openTasks = useMemo(() => tasks.filter((task) => task.status !== "done"), [tasks]);

  const showResult = useCallback((label: string, result: ActionResult) => {
    const chunks = typeof result.chunks === "number" ? `${result.chunks} chunks` : "accepted";
    const total = typeof result.total === "number" ? `, ${result.total} total` : "";
    setNotice({ tone: "ok", text: `${label}: ${chunks}${total}` });
  }, []);

  const loadTasks = useCallback(async () => {
    setBusy("tasks");
    try {
      const result = await requestJson<TasksResponse>("/api/brain/tasks?status=open");
      setTasks(result.tasks);
      setNotice(null);
    } catch (error) {
      setNotice({ tone: "warn", text: error instanceof Error ? error.message : "Could not load tasks." });
    } finally {
      setBusy(null);
    }
  }, []);

  const loadTraces = useCallback(async () => {
    setBusy("traces");
    try {
      const result = await requestJson<TracesResponse>("/api/traces");
      setTraces(result);
      setNotice(null);
    } catch (error) {
      setNotice({ tone: "warn", text: error instanceof Error ? error.message : "Could not load traces." });
    } finally {
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      if (activeTab === "tasks") void loadTasks();
      if (activeTab === "traces") void loadTraces();
    }, 0);
    return () => window.clearTimeout(handle);
  }, [activeTab, loadTasks, loadTraces]);

  const submitIngest = async (event: FormEvent) => {
    event.preventDefault();
    if (!ingestText.trim()) return;
    setBusy("ingest");
    try {
      const result = await requestJson<ActionResult>("/api/brain/ingest", {
        method: "POST",
        body: JSON.stringify({ source: ingestSource, text: ingestText }),
      });
      setIngestText("");
      showResult("Ingested into Real", result);
      onMutation();
    } catch (error) {
      setNotice({ tone: "warn", text: error instanceof Error ? error.message : "Ingest failed." });
    } finally {
      setBusy(null);
    }
  };

  const submitLearn = async (event: FormEvent) => {
    event.preventDefault();
    if (!learnText.trim()) return;
    setBusy("learn");
    try {
      const result = await requestJson<ActionResult>("/api/brain/learn", {
        method: "POST",
        body: JSON.stringify({ source: learnSource, text: learnText }),
      });
      setLearnText("");
      showResult("Learned memory", result);
      onMutation();
    } catch (error) {
      setNotice({ tone: "warn", text: error instanceof Error ? error.message : "Learn failed." });
    } finally {
      setBusy(null);
    }
  };

  const submitTask = async (event: FormEvent) => {
    event.preventDefault();
    if (!taskTitle.trim()) return;
    setBusy("tasks");
    try {
      await requestJson<BrainTask>("/api/brain/tasks", {
        method: "POST",
        body: JSON.stringify({ title: taskTitle, notes: taskNotes }),
      });
      setTaskTitle("");
      setTaskNotes("");
      await loadTasks();
      setNotice({ tone: "ok", text: "Task added." });
    } catch (error) {
      setNotice({ tone: "warn", text: error instanceof Error ? error.message : "Task add failed." });
    } finally {
      setBusy(null);
    }
  };

  const completeTask = async (taskId: number) => {
    setBusy("complete");
    try {
      await requestJson<{ completed: boolean }>(`/api/brain/tasks/${taskId}/complete`, { method: "POST" });
      await loadTasks();
      setNotice({ tone: "ok", text: "Task completed." });
    } catch (error) {
      setNotice({ tone: "warn", text: error instanceof Error ? error.message : "Could not complete task." });
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="mt-3 rounded-md border border-[var(--border)] bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] px-3 py-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-[var(--accent)]" />
          <span className="text-sm font-semibold text-slate-950">Owner tools</span>
        </div>
        <span className="rounded-md bg-[var(--accent-soft)] px-2 py-1 text-xs font-medium text-[var(--accent)]">Real</span>
      </div>

      <div className="grid grid-cols-4 border-b border-[var(--border)] bg-[var(--surface-muted)]">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`flex h-10 items-center justify-center gap-2 border-r border-[var(--border)] text-xs font-semibold transition last:border-r-0 ${
                active ? "bg-white text-slate-950" : "text-slate-600 hover:bg-white/70 hover:text-slate-950"
              }`}
            >
              <Icon className="size-3.5" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {notice && (
        <div
          className={`mx-3 mt-3 flex items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium ${
            notice.tone === "ok"
              ? "border-emerald-200 bg-emerald-50 text-[var(--accent)]"
              : "border-amber-300 bg-[var(--amber-soft)] text-[var(--amber)]"
          }`}
        >
          {notice.tone === "ok" ? <CheckCircle2 className="size-3.5 shrink-0" /> : <AlertTriangle className="size-3.5 shrink-0" />}
          <span>{notice.text}</span>
        </div>
      )}

      {activeTab === "ingest" && (
        <form className="space-y-3 p-3" onSubmit={submitIngest}>
          <input
            value={ingestSource}
            onChange={(event) => setIngestSource(event.target.value)}
            className="h-9 w-full rounded-md border border-[var(--border)] px-3 text-sm outline-none focus:border-[var(--accent)]"
            placeholder="source-name.md"
            maxLength={120}
          />
          <textarea
            value={ingestText}
            onChange={(event) => setIngestText(event.target.value)}
            className="min-h-28 w-full resize-y rounded-md border border-[var(--border)] px-3 py-2 text-sm leading-6 outline-none focus:border-[var(--accent)]"
            placeholder="Paste text to chunk and index..."
            maxLength={100000}
          />
          <OwnerSubmitButton busy={busy === "ingest"} disabled={!ingestText.trim()} label="Ingest text" icon={FileUp} />
        </form>
      )}

      {activeTab === "learn" && (
        <form className="space-y-3 p-3" onSubmit={submitLearn}>
          <input
            value={learnSource}
            onChange={(event) => setLearnSource(event.target.value)}
            className="h-9 w-full rounded-md border border-[var(--border)] px-3 text-sm outline-none focus:border-[var(--accent)]"
            placeholder="memory source"
            maxLength={120}
          />
          <textarea
            value={learnText}
            onChange={(event) => setLearnText(event.target.value)}
            className="min-h-24 w-full resize-y rounded-md border border-[var(--border)] px-3 py-2 text-sm leading-6 outline-none focus:border-[var(--accent)]"
            placeholder="Write a durable memory..."
            maxLength={20000}
          />
          <OwnerSubmitButton busy={busy === "learn"} disabled={!learnText.trim()} label="Learn memory" icon={ShieldCheck} />
        </form>
      )}

      {activeTab === "tasks" && (
        <div className="space-y-3 p-3">
          <form className="grid gap-2" onSubmit={submitTask}>
            <input
              value={taskTitle}
              onChange={(event) => setTaskTitle(event.target.value)}
              className="h-9 rounded-md border border-[var(--border)] px-3 text-sm outline-none focus:border-[var(--accent)]"
              placeholder="Task title"
              maxLength={500}
            />
            <input
              value={taskNotes}
              onChange={(event) => setTaskNotes(event.target.value)}
              className="h-9 rounded-md border border-[var(--border)] px-3 text-sm outline-none focus:border-[var(--accent)]"
              placeholder="Notes"
              maxLength={5000}
            />
            <OwnerSubmitButton busy={busy === "tasks"} disabled={!taskTitle.trim()} label="Add task" icon={Plus} />
          </form>
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-600">{openTasks.length} open tasks</span>
            <button
              type="button"
              onClick={loadTasks}
              className="grid size-8 place-items-center rounded-md border border-[var(--border)] text-slate-600 transition hover:border-[var(--border-strong)] hover:text-slate-950"
              aria-label="Refresh tasks"
              title="Refresh tasks"
            >
              {busy === "tasks" ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
            </button>
          </div>
          <div className="max-h-48 divide-y divide-[var(--border)] overflow-y-auto rounded-md border border-[var(--border)]">
            {openTasks.length === 0 ? (
              <div className="px-3 py-8 text-center text-xs text-slate-500">No open tasks.</div>
            ) : (
              openTasks.map((task) => (
                <div key={task.id} className="grid grid-cols-[1fr_34px] gap-2 px-3 py-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-800">{task.title}</p>
                    {task.notes && <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-500">{task.notes}</p>}
                  </div>
                  <button
                    type="button"
                    onClick={() => completeTask(task.id)}
                    className="grid size-8 place-items-center rounded-md border border-[var(--border)] text-slate-600 transition hover:border-emerald-300 hover:bg-emerald-50 hover:text-[var(--accent)]"
                    aria-label={`Complete task ${task.id}`}
                    title="Complete task"
                  >
                    {busy === "complete" ? <Loader2 className="size-3.5 animate-spin" /> : <CheckSquare className="size-3.5" />}
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {activeTab === "traces" && (
        <div className="space-y-3 p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-600">{traces?.benchmark ?? "Local trace export"}</span>
            <button
              type="button"
              onClick={loadTraces}
              className="grid size-8 place-items-center rounded-md border border-[var(--border)] text-slate-600 transition hover:border-[var(--border-strong)] hover:text-slate-950"
              aria-label="Refresh traces"
              title="Refresh traces"
            >
              {busy === "traces" ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
            </button>
          </div>

          {traces?.available && traces.summary ? (
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(traces.summary)
                .slice(0, 3)
                .map(([key, value]) => (
                  <div key={key} className="rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-2 py-2">
                    <p className="truncate text-[11px] font-medium text-slate-500">{key}</p>
                    <p className="mt-1 font-mono text-sm font-semibold text-slate-800">{formatSummaryValue(value)}</p>
                  </div>
                ))}
            </div>
          ) : (
            <div className="rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-8 text-center text-xs text-slate-500">
              No trace export found.
            </div>
          )}

          {traces?.available && traces.traces.length > 0 && (
            <div className="max-h-56 divide-y divide-[var(--border)] overflow-y-auto rounded-md border border-[var(--border)]">
              {traces.traces.map((trace, index) => {
                const steps = trace.trajectory ?? trace.spans ?? [];
                return (
                  <div key={trace.trace_id ?? index} className="px-3 py-2">
                    <div className="flex items-center justify-between gap-3">
                      <p className="truncate text-sm font-medium text-slate-800">{trace.name ?? trace.trace_id ?? `Trace ${index + 1}`}</p>
                      <span className="shrink-0 rounded-md bg-[var(--surface-muted)] px-2 py-1 text-[11px] font-medium text-slate-600">
                        {steps.length} steps
                      </span>
                    </div>
                    {steps.length > 0 && (
                      <p className="mt-1 truncate text-xs text-slate-500">
                        {steps
                          .slice(0, 4)
                          .map((step) => step.name)
                          .filter(Boolean)
                          .join(" / ")}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function OwnerSubmitButton({
  busy,
  disabled,
  label,
  icon: Icon,
}: {
  busy: boolean;
  disabled: boolean;
  label: string;
  icon: typeof FileUp;
}) {
  return (
    <button
      type="submit"
      disabled={disabled || busy}
      className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-[var(--accent)] px-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-slate-300"
    >
      {busy ? <Loader2 className="size-4 animate-spin" /> : <Icon className="size-4" />}
      {label}
    </button>
  );
}
