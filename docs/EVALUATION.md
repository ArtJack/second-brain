# Evaluating second-brain

The first evaluation layer is intentionally local, cheap, and deterministic. A benchmark is
a versioned JSON file containing questions and the sources retrieval should find. Optional
answer checks add simple grounding heuristics without requiring a second model.

## Run the smoke benchmark

Ingest the demo note once, then run retrieval-only evaluation:

```bash
uv run sb ingest examples/lab-notes.md
uv run sb eval
```

Add local chat-model calls and answer heuristics:

```bash
uv run sb eval --answers
uv run sb eval --answers --json > data/eval-report.json
uv run sb eval --answers --trace-output data/eval-traces.json
```

`sb eval` exits non-zero when any case fails, so it can become a CI or deployment smoke
check once a test store is available.

## Run the regression benchmark

`evals/regression.json` contains 26 cases over a clearly labeled synthetic corpus:

- 18 focused single-source questions
- 4 multi-source questions
- 4 unsupported questions that should produce an explicit abstention

Use `--ingest-corpus` to load its fixture documents before running it:

```bash
SB_COLLECTION=second_brain_regression uv run sb eval evals/regression.json --ingest-corpus
SB_COLLECTION=second_brain_regression uv run sb eval evals/regression.json --answers
```

Filter by repeatable tags when a local answer-model run should stay small:

```bash
SB_COLLECTION=second_brain_regression uv run sb eval evals/regression.json --answers --tag rubric-smoke
SB_COLLECTION=second_brain_regression uv run sb eval evals/regression.json --answers --tag abstention
```

The synthetic corpus is for repeatable engineering checks. It should complement a private
benchmark built from real usage questions, not be mistaken for personal memory. The
configured `corpus_collection` guard prevents `--ingest-corpus` from writing fixtures into
the normal `second_brain` collection.

## Benchmark format

The starter dataset lives at `evals/retrieval.json`. The larger suite also declares a
corpus path relative to its JSON file:

```json
{
  "name": "my-benchmark",
  "corpus": ["corpus"],
  "corpus_collection": "second_brain_regression",
  "cases": [
    {
      "id": "gateway-url",
      "query": "What URL does the LiteLLM gateway use?",
      "expected_sources": ["examples/lab-notes.md"],
      "expected_answer_contains": ["http://127.0.0.1:4000"],
      "tags": ["gateway"]
    }
  ]
}
```

Source paths match by exact value or path suffix. This lets the same benchmark work when a
file was ingested as a relative path on one machine and an absolute path on another.

Multi-source cases list every required source. Unsupported questions omit
`expected_sources` and set `expect_abstain`:

```json
{
  "id": "abstain-coffee",
  "query": "What is the user's favorite coffee order?",
  "expect_abstain": true,
  "tags": ["abstention"]
}
```

## Metrics

- **Retrieval hit-rate**: fraction of cases where every expected source appears in top-k.
- **Mean source recall**: average fraction of expected sources retrieved per case.
- **MRR**: mean reciprocal rank of the first relevant source. Higher means useful evidence
  appears earlier.
- **Answer rubric** with `--answers`: require bounded citations to every expected source,
  check configured required phrases, and require explicit abstention for unsupported
  questions.
- **Rubric score**: the mean fraction of deterministic answer checks that passed.

Abstention cases do not have a relevant source, so retrieval metrics skip them. Answer mode
still retrieves context and grades whether the model refuses to invent an answer.

These rubric checks are guardrails, not semantic proof. They catch cheap failures while
keeping the default loop fast enough to run often.

## Trace, span, and trajectory

Every benchmark case emits one local JSON **trace** automatically. A trace is the
end-to-end execution record for that case. It contains nested timed **spans**:

| Span | Kind | Meaning |
|---|---|---|
| `eval.case` | workflow | Root span for the complete case. |
| `retrieve` | retrieval | Query the configured vector store. |
| `evaluate_retrieval` | evaluator | Score expected-source coverage and rank. |
| `answer_path` | rag | Run the grounded RAG answer path when `--answers` is enabled. |
| `evaluate_answer` | evaluator | Apply citation, source-coverage, phrase, and abstention checks. |

The normal `answer_path` also emits nested spans:

| Nested span | Kind | Meaning |
|---|---|---|
| `store_count` | store | Check whether any indexed chunks exist. |
| `embed_question` | embedding | Embed the question. |
| `retrieve_context` | retrieval | Fetch answer context from the vector store. |
| `generate_answer` | llm | Call the configured local or gateway-backed chat model. |

The trace also includes a **trajectory**: the ordered list of meaningful completed child
spans. Retrieval-only cases have two trajectory steps. Normal answer-mode cases include the
four nested RAG operations as well as their wrapping `answer_path` and the answer evaluator.
The root workflow span is omitted from the trajectory because it wraps the whole case rather
than representing an additional action.

The full `--json` report includes traces. Use `--trace-output` when you want a standalone
artifact:

```bash
SB_COLLECTION=second_brain_regression \
  uv run sb eval evals/regression.json \
  --answers --tag rubric-smoke \
  --trace-output data/regression-traces.json
```

Trace artifacts include questions, source paths, scores, and timings. They intentionally
exclude retrieved document bodies, but they may still contain private metadata. Keep them
local unless they have been reviewed.

## Evaluation layers

Use the terms carefully:

| Layer | Add now? | Role in second-brain |
|---|---|---|
| Benchmark | Yes | Versioned questions and expected evidence. Grow this from real usage. |
| Evaluator | Yes | The runner that executes benchmark cases and records metrics. |
| Heuristic | Yes | Cheap checks for retrieval, citation bounds, cited sources, and latency. |
| Trace / span | Yes | Local nested spans capture workflow, retrieval, generation, evaluator outcomes, and timings. |
| Trajectory | Yes | Each case records the ordered retrieval, generation, and evaluation path. Reuse the same recorder for future multi-step agent tools. |
| Rubric | Yes | Deterministic criteria for citations, source coverage, required phrases, and abstention. |
| LLM judge | Later, optional | Sampled semantic grading after heuristics. Prefer a separately configured stronger judge and calibrate it against human labels. |
| Reward function | Not yet | Useful only when optimizing a ranker, prompt, or policy automatically. A dashboard score is not automatically a reward function. |

For a mostly-local assistant, deterministic retrieval metrics should remain the everyday
signal. A judge model is a slower audit layer, not the foundation.

## Build a private benchmark

Use the terminal intake to create a private 100-question benchmark from your own context:

```bash
uv run sb eval-intake
```

Progress saves after every answer under `data/private-eval/`, which is covered by the
repository's ignored `data/` directory. Resume with the same command at any time.

Terminal commands:

| Command | Meaning |
|---|---|
| `/skip` | Leave a prompt out of the generated corpus and benchmark. |
| `/back` | Reopen the previous saved prompt. |
| `/status` | Show answered, skipped, and remaining counts. |
| `/quit` | Pause safely after writing the current private artifacts. |

The collector generates one private corpus document per category plus
`data/private-eval/benchmark.json`. Evaluate it in its isolated collection:

```bash
SB_COLLECTION=second_brain_private_eval \
  uv run sb eval data/private-eval/benchmark.json --ingest-corpus
```

The generated benchmark stores a `reference_answer` for each answered prompt. Retrieval
metrics work immediately. A future optional judge can compare model output against those
reference answers. Do not enter passwords, tokens, account numbers, or government IDs.
