# second-brain eval results

## 2026-09-12 — persisted keyword index, RRF fusion, `limit` as a cap

Benchmark: `evals/regression.json`, 26 cases against the throwaway
`second_brain_regression` collection. Retrieval metrics score the 22 answerable
cases; the 4 abstention cases are skipped for retrieval scoring.

Model/config: `embed` through the LiteLLM gateway into Qdrant, `SB_HYBRID=1`.
Retrieval-only — the chat model was configured but not called.

Measured immediately before and after the change, on the same machine, same day.
The 2026-06-07 numbers below were **not** used as the baseline: the corpus
fixtures changed on 2026-08-30 and were never re-measured, so they describe a
different benchmark.

| metric | before | after |
|---|---:|---:|
| retrieval_hit_rate | 1.0 | 1.0 |
| mean_source_recall | 1.0 | 1.0 |
| mrr | 1.0 | 1.0 |
| retrieval_passed / cases | 22 / 22 | 22 / 22 |

**The benchmark cannot show the improvement, only guard against regression.** It
was already saturated at 1.0 across the board before the change, which is a
finding about the benchmark rather than about retrieval. A harder set is queued.

It did earn its keep once: the first implementation scored **MRR 0.970**, because
moving to FTS5 dropped the stopword filtering the hand-rolled scorer had. The
match expression joins terms with `OR`, so "what", "is" and "the" made every
document in the corpus a candidate and diluted BM25 across all of them — enough
to move `admin-invoice-format`'s correct source from rank 1 to rank 3. Restoring
the filter returned MRR to 1.0. That is pinned by a test now.

### Later the same day: fusion order, and a measured cost

Verdict found that neighbours were entering the fused list at retrieved ranks, so
at the default k=5 two of five answer slots went to adjacency padding that
matched nothing, while two of three real keyword matches were dropped. Fixing it
— fuse what the retrievers matched, expand neighbours only into leftover slots —
costs one case on this benchmark:

| metric | padding in the cap | real matches first |
|---|---:|---:|
| retrieval hit-rate / recall / MRR | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| answers passed | 20 | 19 |
| rubric | 0.9615 | 0.9551 |
| abstention | 4 of 4 | 4 of 4 |

The case is `lab-code-route`, and retrieval was correct in both: the expected
source came back at rank 1. What changed is that the answer no longer contains
the phrase `workstation-01`, which lives in the *adjacent* chunk.

That is the benchmark's structure rather than a real regression. Its fixtures are
seven-to-ten-line documents, so a "neighbour" is usually the rest of the same
small note and padding looks free. On the 1,548-chunk personal corpus a neighbour
is a different part of a long document and competes with evidence that actually
matched. The change is shipped on that reasoning, with the cost recorded here
rather than tuned away, and it is a case the harder benchmark should settle.

### What the benchmark could not measure

Both measured on the live `second_brain` collection (1,550 chunks after the
2026-09-12 sweep and reference split):

| | before | after |
|---|---:|---:|
| `keyword_query` latency, median of 5 | 380 ms | 1 ms |
| bytes read from the store per query | ~7.2 MB | none |
| hits for a Cyrillic query | 0 | 3 |

The latency figure understates the change: the old path read the entire
collection on every query, so its cost grew with the corpus, while the index does
not. The Cyrillic figure is not a speedup but a capability — the previous
tokenizer was `[a-z0-9]+`, so a Russian question tokenized to the empty list and
keyword search silently returned nothing.

---

## 2026-06-07 — hybrid vector + BM25 (superseded)

Retained for history. The corpus fixtures changed on 2026-08-30 without a
re-measurement, so these numbers do not describe the current benchmark.

| metric | vector-only (`SB_HYBRID=0`) | hybrid (`SB_HYBRID=1`) |
|---|---:|---:|
| retrieval_hit_rate | 1.0 | 1.0 |
| mean_source_recall | 1.0 | 1.0 |
| mrr | 0.9697 | 1.0 |
| retrieval_passed / retrieval_cases | 22 / 22 | 22 / 22 |

Hybrid tied vector-only on hit rate and source recall for that corpus, and improved MRR by moving
the first relevant source to rank 1 for every scored retrieval case.
