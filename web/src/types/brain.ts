export type Corpus = "public" | "neutral";

export type BrainSource = {
  n: number;
  source: string;
  type?: string;
  distance: number;
  retrieval?: string;
  text?: string;
};

export type BrainAnswer = {
  answer: string;
  latencyMs: number;
  model: string;
  sources: BrainSource[];
  invalid_citations?: number[];
};

export type CorpusFallback = {
  label: string;
  prompts: string[];
  answers: Record<string, BrainAnswer>;
};

export type FallbackData = Record<Corpus, CorpusFallback>;

export type BrainStatus = {
  corpus: Corpus;
  collection: string;
  store: string;
  backend: string;
  embed_model: string;
  chat_model: string;
  chunks: number | string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  latencyMs?: number;
  model?: string;
  sources?: BrainSource[];
  invalidCitations?: number[];
  streaming?: boolean;
  offline?: boolean;
};
