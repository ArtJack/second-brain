"use client";

import {
  Activity,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  CircleUserRound,
  Copy,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Loader2,
  Moon,
  RefreshCw,
  Scale,
  Send,
  Settings2,
  Sparkles,
  TerminalSquare,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import fallbackJson from "@/data/demo-fallback.json";
import type { BrainAnswer, BrainSource, BrainStatus, ChatMessage, Corpus, FallbackData } from "@/types/brain";
import { TurnstileField } from "@/components/turnstile-field";

const fallback = fallbackJson as FallbackData;

const corpusMeta: Record<Corpus, { label: string; icon: typeof FlaskConical; description: string }> = {
  public: {
    label: "Lab",
    icon: FlaskConical,
    description: "ArtJeck project docs",
  },
  neutral: {
    label: "Neutral",
    icon: Scale,
    description: "Synthetic QA corpus",
  },
};

function nowTime() {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date());
}

function messageId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function answerFor(corpus: Corpus, prompt: string): BrainAnswer {
  const corpusFallback = fallback[corpus];
  return corpusFallback.answers[prompt] ?? Object.values(corpusFallback.answers)[0];
}

function initialMessages(corpus: Corpus): ChatMessage[] {
  const prompt = fallback[corpus].prompts[0];
  const answer = answerFor(corpus, prompt);
  return [
    {
      id: messageId(),
      role: "user",
      content: prompt,
      createdAt: "10:32:14",
    },
    {
      id: messageId(),
      role: "assistant",
      content: answer.answer,
      createdAt: "10:32:15",
      latencyMs: answer.latencyMs,
      model: answer.model,
      sources: answer.sources,
      invalidCitations: answer.invalid_citations,
      offline: true,
    },
  ];
}

function parseSseEvent(block: string) {
  let event = "message";
  const data: string[] = [];
  for (const raw of block.split("\n")) {
    const line = raw.trimEnd();
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return { event, data: data.join("\n") };
}

function sourceType(source: string) {
  const ext = source.split(".").pop()?.toLowerCase();
  if (ext === "md" || ext === "markdown") return "Markdown";
  if (ext === "py") return "Python";
  if (ext === "json") return "JSON";
  return "Text";
}

function normalizeSources(hits: Array<{ source: string; distance: number; text: string }>): BrainSource[] {
  return hits.map((hit, index) => ({
    n: index + 1,
    source: hit.source,
    type: sourceType(hit.source),
    distance: hit.distance,
    text: hit.text,
  }));
}

function relevanceBars(distance: number) {
  const score = Math.max(1, Math.min(5, Math.round((1 - Math.min(distance, 0.7)) * 5)));
  return Array.from({ length: 5 }, (_, index) => index < score);
}

function updateMessage(messages: ChatMessage[], id: string, patch: Partial<ChatMessage>) {
  return messages.map((message) => (message.id === id ? { ...message, ...patch } : message));
}

type StatusSnapshot =
  | { health: "online"; status: BrainStatus; lastUpdated: string }
  | { health: "offline"; status: null };

async function readStatus(corpus: Corpus): Promise<StatusSnapshot> {
  try {
    const [healthResponse, statusResponse] = await Promise.all([
      fetch("/api/brain/health", { cache: "no-store" }),
      fetch(`/api/brain/status?corpus=${corpus}`, { cache: "no-store" }),
    ]);
    if (!healthResponse.ok || !statusResponse.ok) throw new Error("offline");
    const statusJson = (await statusResponse.json()) as BrainStatus;
    return {
      health: "online",
      status: statusJson,
      lastUpdated: new Intl.DateTimeFormat("en-US", {
        hour: "numeric",
        minute: "2-digit",
      }).format(new Date()),
    };
  } catch {
    return { health: "offline", status: null };
  }
}

export function BrainApp() {
  const [corpus, setCorpus] = useState<Corpus>("public");
  const [messages, setMessages] = useState<ChatMessage[]>(() => initialMessages("public"));
  const [question, setQuestion] = useState("");
  const [selectedSource, setSelectedSource] = useState(0);
  const [isAsking, setIsAsking] = useState(false);
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [status, setStatus] = useState<BrainStatus | null>(null);
  const [turnstileToken, setTurnstileToken] = useState("");
  const [lastUpdated, setLastUpdated] = useState("10:31 AM");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const latestAssistant = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant"),
    [messages],
  );
  const sources = latestAssistant?.sources ?? [];
  const invalidCitations = latestAssistant?.invalidCitations ?? [];
  const activeSource = sources[Math.min(selectedSource, Math.max(sources.length - 1, 0))];
  const meta = corpusMeta[corpus];

  const applyStatusSnapshot = useCallback((snapshot: StatusSnapshot) => {
    setHealth(snapshot.health);
    setStatus(snapshot.status);
    if (snapshot.health === "online") {
      setLastUpdated(snapshot.lastUpdated);
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    applyStatusSnapshot(await readStatus(corpus));
  }, [applyStatusSnapshot, corpus]);

  useEffect(() => {
    let active = true;
    const check = async () => {
      const snapshot = await readStatus(corpus);
      if (active) applyStatusSnapshot(snapshot);
    };
    void check();
    const interval = window.setInterval(check, 30_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [applyStatusSnapshot, corpus]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const applyFallbackAnswer = useCallback(
    (prompt: string, assistantId: string) => {
      const fallbackAnswer = answerFor(corpus, prompt);
      setMessages((current) =>
        updateMessage(current, assistantId, {
          content: fallbackAnswer.answer,
          latencyMs: fallbackAnswer.latencyMs,
          model: fallbackAnswer.model,
          sources: fallbackAnswer.sources,
          invalidCitations: fallbackAnswer.invalid_citations,
          streaming: false,
          offline: true,
        }),
      );
      setSelectedSource(0);
    },
    [corpus],
  );

  const askQuestion = useCallback(
    async (prompt?: string) => {
      const text = (prompt ?? question).trim();
      if (!text || isAsking) return;

      const userMessage: ChatMessage = {
        id: messageId(),
        role: "user",
        content: text,
        createdAt: nowTime(),
      };
      const assistantId = messageId();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: nowTime(),
        streaming: true,
        model: status?.chat_model ?? "local model",
        sources: [],
      };

      setMessages((current) => [...current, userMessage, assistantMessage]);
      setQuestion("");
      setIsAsking(true);

      if (health !== "online") {
        window.setTimeout(() => {
          applyFallbackAnswer(text, assistantId);
          setIsAsking(false);
        }, 260);
        return;
      }

      const started = performance.now();
      try {
        const recallResponse = await fetch("/api/brain/recall", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-turnstile-token": turnstileToken,
          },
          body: JSON.stringify({ query: text, top_k: 5, corpus }),
        });
        if (recallResponse.ok) {
          const recall = (await recallResponse.json()) as { hits?: Array<{ source: string; distance: number; text: string }> };
          const recallSources = normalizeSources(recall.hits ?? []);
          setMessages((current) => updateMessage(current, assistantId, { sources: recallSources }));
          setSelectedSource(0);
        }

        const streamResponse = await fetch("/api/brain/ask/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-turnstile-token": turnstileToken,
          },
          body: JSON.stringify({ question: text, k: 5, corpus }),
        });
        if (!streamResponse.ok || !streamResponse.body) throw new Error("stream failed");

        const reader = streamResponse.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let answer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() ?? "";
          for (const block of blocks) {
            if (!block.trim()) continue;
            const parsed = parseSseEvent(block);
            if (parsed.event === "token") {
              answer += parsed.data;
              setMessages((current) =>
                updateMessage(current, assistantId, {
                  content: answer,
                  latencyMs: Math.round(performance.now() - started),
                }),
              );
            }
            if (parsed.event === "sources") {
              const streamSources = JSON.parse(parsed.data) as BrainSource[];
              setMessages((current) =>
                updateMessage(current, assistantId, {
                  sources: streamSources.map((source) => ({ ...source, type: sourceType(source.source) })),
                }),
              );
            }
            if (parsed.event === "done") {
              const doneData = JSON.parse(parsed.data) as { invalid_citations?: number[] };
              setMessages((current) =>
                updateMessage(current, assistantId, {
                  invalidCitations: doneData.invalid_citations ?? [],
                  streaming: false,
                  latencyMs: Math.round(performance.now() - started),
                  model: status?.chat_model ?? "local model",
                }),
              );
            }
          }
        }
        setMessages((current) =>
          updateMessage(current, assistantId, {
            streaming: false,
            latencyMs: Math.round(performance.now() - started),
            model: status?.chat_model ?? "local model",
          }),
        );
      } catch {
        setHealth("offline");
        applyFallbackAnswer(text, assistantId);
      } finally {
        setIsAsking(false);
      }
    },
    [applyFallbackAnswer, corpus, health, isAsking, question, status?.chat_model, turnstileToken],
  );

  const selectCorpus = useCallback((nextCorpus: Corpus) => {
    setCorpus(nextCorpus);
    setMessages(initialMessages(nextCorpus));
    setSelectedSource(0);
    setQuestion("");
    setHealth("checking");
  }, []);

  const prompts = fallback[corpus].prompts;
  const sourceCount = sources.length;
  const uniqueSources = new Set(sources.map((source) => source.source)).size;
  const modelLabel = latestAssistant?.model?.includes("claude") ? "Claude" : "local model";

  return (
    <main className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <header className="flex h-[68px] items-center justify-between border-b border-[var(--border)] bg-white/90 px-4 backdrop-blur md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid size-9 shrink-0 place-items-center rounded-md bg-[var(--accent)] font-mono text-sm font-bold text-white">
            AJ
          </div>
          <div className="flex min-w-0 items-center gap-3">
            <span className="text-lg font-semibold text-slate-950">ArtJeck</span>
            <span className="hidden h-6 w-px bg-[var(--border)] sm:block" />
            <span className="truncate text-sm font-semibold text-[var(--accent)]">second-brain</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-700">
          <IconButton label="Open console" icon={TerminalSquare} />
          <IconButton label="Docs" icon={BookOpen} />
          <IconButton label="Theme" icon={Moon} />
          <div className="ml-1 hidden items-center gap-2 rounded-md border border-[var(--border)] bg-white px-2 py-1.5 sm:flex">
            <CircleUserRound className="size-5 text-slate-500" />
            <span>Anonymous</span>
            <ChevronDown className="size-4 text-slate-500" />
          </div>
        </div>
      </header>

      <div className="grid min-h-[calc(100vh-124px)] grid-cols-1 lg:grid-cols-[38%_62%]">
        <section className="border-b border-[var(--border)] bg-white/72 p-4 lg:border-r lg:border-b-0 lg:p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h1 className="text-base font-semibold text-slate-950">Ask the lab</h1>
              <p className="mt-1 text-xs text-[var(--muted)]">{meta.description}</p>
            </div>
            <button
              type="button"
              onClick={refreshStatus}
              className="grid size-9 place-items-center rounded-md border border-[var(--border)] bg-white text-slate-600 transition hover:border-[var(--border-strong)] hover:text-slate-950"
              aria-label="Refresh status"
              title="Refresh status"
            >
              <RefreshCw className="size-4" />
            </button>
          </div>

          <div className="mb-3 grid grid-cols-2 gap-2">
            {(Object.keys(corpusMeta) as Corpus[]).map((key) => {
              const Icon = corpusMeta[key].icon;
              const active = corpus === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => selectCorpus(key)}
                  className={`flex h-10 items-center justify-center gap-2 rounded-md border text-sm font-medium transition ${
                    active
                      ? "border-[var(--accent)] bg-[var(--accent)] text-white shadow-sm"
                      : "border-[var(--border)] bg-white text-slate-600 hover:border-[var(--border-strong)] hover:text-slate-950"
                  }`}
                >
                  <Icon className="size-4" />
                  {corpusMeta[key].label}
                </button>
              );
            })}
          </div>

          <div ref={scrollRef} className="max-h-[calc(100vh-390px)] min-h-[360px] space-y-3 overflow-y-auto pr-1">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </div>

          <form
            className="mt-3 rounded-md border border-[var(--border)] bg-white"
            onSubmit={(event) => {
              event.preventDefault();
              askQuestion();
            }}
          >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about the lab..."
              className="min-h-24 w-full resize-none rounded-t-md border-0 bg-transparent px-3 py-3 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400"
              maxLength={4000}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  askQuestion();
                }
              }}
            />
            <div className="flex items-center justify-between border-t border-[var(--border)] px-3 py-2">
              <span className="text-xs text-[var(--muted)]">{question.length} / 4000</span>
              <div className="flex items-center gap-2">
                <IconButton label="Adjust retrieval" icon={Settings2} />
                <button
                  type="submit"
                  disabled={!question.trim() || isAsking}
                  className="grid size-10 place-items-center rounded-md bg-[var(--accent)] text-white shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-slate-300"
                  aria-label="Send question"
                  title="Send question"
                >
                  {isAsking ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
                </button>
              </div>
            </div>
          </form>

          <div className="mt-3 rounded-md border border-[var(--border)] bg-white">
            <div className="flex items-center justify-between border-b border-[var(--border)] px-3 py-2">
              <span className="text-sm font-semibold text-slate-950">Suggested prompts</span>
              <ChevronUp className="size-4 text-slate-500" />
            </div>
            <div className="divide-y divide-[var(--border)]">
              {prompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => askQuestion(prompt)}
                  className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-sm text-slate-700 transition hover:bg-[var(--surface-muted)]"
                >
                  <Sparkles className="size-4 shrink-0 text-[var(--accent)]" />
                  <span>{prompt}</span>
                </button>
              ))}
            </div>
          </div>

          <TurnstileField onToken={setTurnstileToken} onExpire={() => setTurnstileToken("")} />
        </section>

        <section className="bg-[var(--surface)] p-4 lg:p-5">
          <div className="mb-4 flex items-center justify-between border-b border-[var(--border)] pb-4">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-slate-950">Sources</h2>
              <span className="rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-2 py-0.5 text-sm font-medium text-slate-700">
                {sourceCount}
              </span>
            </div>
            <div className="flex items-center gap-3 text-sm text-slate-600">
              <span>{uniqueSources} unique sources</span>
              <Settings2 className="size-4" />
            </div>
          </div>

          <div className="overflow-hidden rounded-md border border-[var(--border)] bg-white">
            <div className="grid grid-cols-[44px_minmax(180px,1.5fr)_120px_120px_150px_36px] border-b border-[var(--border)] bg-[var(--surface-muted)] px-3 py-3 text-xs font-medium text-slate-500 max-xl:grid-cols-[44px_minmax(160px,1fr)_110px_110px_36px] max-md:hidden">
              <span>#</span>
              <span>Source</span>
              <span className="max-xl:hidden">Type</span>
              <span>Distance</span>
              <span>Relevance</span>
              <span />
            </div>
            <div className="divide-y divide-[var(--border)]">
              {sources.length === 0 ? (
                <div className="px-4 py-12 text-center text-sm text-slate-500">Ask a question to inspect source chunks.</div>
              ) : (
                sources.map((source, index) => (
                  <button
                    type="button"
                    key={`${source.source}-${source.n}`}
                    onClick={() => setSelectedSource(index)}
                    className={`grid w-full grid-cols-[44px_minmax(180px,1.5fr)_120px_120px_150px_36px] items-center px-3 py-4 text-left text-sm transition max-xl:grid-cols-[44px_minmax(160px,1fr)_110px_110px_36px] max-md:grid-cols-[34px_1fr] max-md:gap-y-2 ${
                      index === selectedSource ? "border-l-2 border-[var(--accent)] bg-[var(--accent-soft)]" : "hover:bg-[var(--surface-muted)]"
                    }`}
                  >
                    <span className="font-mono text-slate-700">{source.n}</span>
                    <span className="flex min-w-0 items-center gap-2 font-medium text-slate-800">
                      <FileText className="size-4 shrink-0 text-slate-500" />
                      <span className="truncate">{source.source}</span>
                      <ExternalLink className="size-3.5 shrink-0 text-slate-400" />
                    </span>
                    <span className="text-slate-600 max-xl:hidden max-md:hidden">{source.type ?? sourceType(source.source)}</span>
                    <span className="font-mono font-semibold text-[var(--accent)] max-md:col-start-2">{source.distance.toFixed(3)}</span>
                    <span className="flex items-center gap-1 max-md:col-start-2">
                      {relevanceBars(source.distance).map((filled, barIndex) => (
                        <span
                          key={barIndex}
                          className={`h-1.5 w-5 rounded-sm ${filled ? "bg-[var(--accent)]" : "bg-slate-300"}`}
                        />
                      ))}
                    </span>
                    <span className="text-slate-400 max-md:hidden">•••</span>
                  </button>
                ))
              )}
            </div>
          </div>

          <div className="mt-4 rounded-md border border-[var(--border)] bg-white">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] px-4 py-3">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-slate-950">Expanded citation</h3>
                {invalidCitations.length > 0 && (
                  <span className="inline-flex items-center gap-1 rounded-md border border-amber-300 bg-[var(--amber-soft)] px-2 py-1 text-xs font-medium text-[var(--amber)]">
                    <AlertTriangle className="size-3.5" />
                    Invalid citation
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <span>
                  Source {activeSource ? selectedSource + 1 : 0} of {sourceCount}
                </span>
                <IconButton
                  label="Previous source"
                  icon={ChevronUp}
                  onClick={() => setSelectedSource((value) => Math.max(0, value - 1))}
                />
                <IconButton
                  label="Next source"
                  icon={ChevronDown}
                  onClick={() => setSelectedSource((value) => Math.min(sourceCount - 1, value + 1))}
                />
              </div>
            </div>

            {activeSource ? (
              <div className="p-4">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <FileText className="size-4 shrink-0 text-slate-500" />
                    <span className="truncate text-sm font-semibold text-slate-800">{activeSource.source}</span>
                    <span className="rounded-md border border-[var(--border)] bg-white px-2 py-1 text-xs text-slate-600">
                      {activeSource.type ?? sourceType(activeSource.source)}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-sm text-slate-600">
                    <span>distance</span>
                    <span className="font-mono font-semibold text-[var(--accent)]">{activeSource.distance.toFixed(3)}</span>
                    <Copy className="size-4 text-slate-400" />
                  </div>
                </div>
                <pre className="min-h-[220px] max-h-[360px] overflow-auto rounded-md border border-[var(--border)] bg-[#fbfcfc] p-0 font-mono text-[13px] leading-6 text-slate-800">
                  {(activeSource.text ?? "Source text unavailable from the live API.").split("\n").map((line, index) => (
                    <div key={index} className="grid grid-cols-[44px_1fr] border-b border-slate-100 last:border-b-0">
                      <span className="select-none bg-slate-50 px-3 text-right text-slate-400">{index + 1}</span>
                      <code className="whitespace-pre-wrap px-3">{line || " "}</code>
                    </div>
                  ))}
                </pre>
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600">
                  <span>Lines 1-{Math.max(1, (activeSource.text ?? "").split("\n").length)} of source chunk</span>
                  <button className="rounded-md border border-[var(--border)] bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-[var(--border-strong)]">
                    Show more context
                  </button>
                </div>
              </div>
            ) : (
              <div className="px-4 py-14 text-center text-sm text-slate-500">Source text will appear here.</div>
            )}
          </div>
        </section>
      </div>

      <footer className="flex min-h-14 flex-wrap items-center justify-between gap-3 border-t border-[var(--border)] bg-white px-4 py-3 text-sm text-slate-600 md:px-6">
        <div className="flex flex-wrap items-center gap-5">
          <StatusPill health={health} />
          <span className="hidden sm:inline">All systems {health === "online" ? "nominal" : "using fallback answers"}</span>
          <span className="inline-flex items-center gap-2">
            <Activity className="size-4 text-slate-500" />
            Model: {status?.chat_model ?? modelLabel}
          </span>
          <span className="inline-flex items-center gap-2">
            <Zap className="size-4 text-slate-500" />
            {latestAssistant?.latencyMs ? `${latestAssistant.latencyMs} ms` : "pending"}
          </span>
          <span className="inline-flex items-center gap-2">
            <Database className="size-4 text-slate-500" />
            {sourceCount} sources
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden md:inline">Updated {lastUpdated}</span>
          <button
            type="button"
            className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium ${
              health === "online"
                ? "border-emerald-200 bg-emerald-50 text-[var(--accent)]"
                : "border-amber-300 bg-[var(--amber-soft)] text-[var(--amber)]"
            }`}
          >
            {health === "online" ? <CheckCircle2 className="size-4" /> : <Moon className="size-4" />}
            {health === "online" ? "Lab online" : "The lab is asleep"}
          </button>
        </div>
      </footer>
    </main>
  );
}

function IconButton({
  label,
  icon: Icon,
  onClick,
}: {
  label: string;
  icon: typeof Settings2;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="grid size-9 place-items-center rounded-md border border-[var(--border)] bg-white text-slate-600 transition hover:border-[var(--border-strong)] hover:text-slate-950"
      aria-label={label}
      title={label}
    >
      <Icon className="size-4" />
    </button>
  );
}

function StatusPill({ health }: { health: "checking" | "online" | "offline" }) {
  const tone =
    health === "online"
      ? "bg-emerald-50 text-[var(--accent)]"
      : health === "offline"
        ? "bg-[var(--amber-soft)] text-[var(--amber)]"
        : "bg-slate-100 text-slate-500";
  return (
    <span className={`inline-flex items-center gap-2 rounded-md px-2.5 py-1 font-medium ${tone}`}>
      <span className="size-2 rounded-full bg-current" />
      {health === "checking" ? "Checking lab" : health === "online" ? "Lab online" : "Fallback mode"}
    </span>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className="rounded-md border border-[var(--border)] bg-white p-3 shadow-[0_1px_0_rgba(15,23,42,0.03)]">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {isAssistant ? <Sparkles className="size-4 text-blue-600" /> : <span className="size-4 rounded-sm bg-[var(--accent)]" />}
          <span className={`text-sm font-semibold ${isAssistant ? "text-blue-700" : "text-[var(--accent)]"}`}>
            {isAssistant ? "Assistant" : "You"}
          </span>
        </div>
        <span className="text-xs text-slate-500">{message.createdAt}</span>
      </div>
      <div className="whitespace-pre-wrap text-sm leading-6 text-slate-800">
        {message.content || <span className="text-slate-400">Streaming answer...</span>}
      </div>
      {isAssistant && (
        <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-[var(--border)] pt-3 text-xs text-slate-600">
          <span className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 font-medium text-[var(--accent)]">
            <Database className="size-3.5" />
            {message.model ?? "local model"}
          </span>
          <span className="inline-flex items-center gap-1">
            <Zap className="size-3.5" />
            {message.latencyMs ? `${message.latencyMs} ms` : "measuring"}
          </span>
          <span>{message.sources?.length ?? 0} sources</span>
          {message.streaming && (
            <span className="inline-flex items-center gap-1 text-[var(--accent)]">
              <Loader2 className="size-3.5 animate-spin" />
              Streaming
            </span>
          )}
          {message.offline && <span className="text-[var(--amber)]">fallback answer</span>}
        </div>
      )}
    </article>
  );
}
