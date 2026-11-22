export type PublicCorpus = "public" | "neutral";
export type Corpus = PublicCorpus | "real";

export type BrainSource = {
  n: number;
  source: string;
  type?: string;
  distance: number;
  retrieval?: string;
  text?: string;
};

export type Grounding = {
  citations_present?: boolean;
  sources_retrieved?: number;
  refused?: boolean;
  unsupported?: boolean;
};

export type BrainAnswer = {
  answer: string;
  latencyMs: number;
  model: string;
  sources: BrainSource[];
  invalid_citations?: number[];
  grounding?: Grounding;
};

export type CorpusFallback = {
  label: string;
  prompts: string[];
  answers: Record<string, BrainAnswer>;
};

export type FallbackData = Record<PublicCorpus, CorpusFallback>;

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
  unsupported?: boolean;
  streaming?: boolean;
  offline?: boolean;
};
