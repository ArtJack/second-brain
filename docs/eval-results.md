# second-brain eval results

## 2026-09-12 — a benchmark that can fail: passage recall, precision, distractors

The regression set has scored 1.0 on every metric since it was written, which is
a finding about the benchmark rather than about retrieval. Two real defects
passed straight through it this week: a prompt change that cost two answer
cases, and a backfill bug that truncated the keyword index to the first chunks
of every long file. Neither moved a number.

Three things were wrong with how it scored.

**It scored files, not passages.** A case passed when any chunk of the right
document came back. A document of 331 chunks scores a hit on chunk 0, so the
truncation bug was invisible by construction. Cases may now name
`expected_chunk_contains`: the phrase that actually answers the question has to
appear in a chunk the reader will see.

**Nothing cost anything for being wrong.** Precision was not merely unmeasured,
it was unmeasurable, because no case declared what a wrong answer looked like.
Cases may now name `forbidden_sources`, the near-miss documents they must not
surface, and the summary reports precision and a distractor rate over the cases
that named them.

**The corpus was eight files of about 400 bytes.** Each question mapped to an
obviously distinct document and there were no near misses to get wrong.

`evals/hard.json` is the replacement: 20 documents, 20 cases, each targeting one
specific way retrieval goes wrong — an answer far from the top of a long file, a
near-miss sharing the query's vocabulary, a superseded document that still reads
as authoritative, a query whose words appear nowhere in the answer, a rare
identifier with no semantic neighbours, and non-ASCII text. The benchmark's own
correctness is tested offline: a typo in a `forbidden_sources` path would name a
document that can never be retrieved, so the case would pass every time while
measuring nothing.

Both benchmarks, same machine, same day, k=5, `SB_HYBRID=1`, Qdrant:

| metric | regression (old) | hard (new) |
|---|---:|---:|
| retrieval_passed / cases | 22 / 22 | 14 / 18 |
| retrieval_hit_rate | 100% | 77.8% |
| mean_source_recall | 100% | 100% |
| mrr | 1.000 | 0.852 |
| mean_precision | 47.3% | 28.9% |
| mean_passage_recall | not asked | 100% |
| distractor_rate | not asked | 100% |

**The headline is the distractor rate.** All four cases that named a near-miss
document retrieved it, out of a corpus of twenty. Retrieval has no notion of
supersession or recency: asked which agent owns QA, it returns the current
routing document *and* the one marked "SUPERSEDED — do not action anything in
this file", and the answer is then built from both. Asked for the offsite backup
time it returns the local snapshot policy alongside. Source recall stays at
100%, so every older metric says this is working perfectly.

That is now a measured, reproducible number to drive down rather than a
suspicion. It is not yet fixed, and no change in this session was made to chase
it.

Two secondary readings. Precision at 47.3% on the *old* benchmark says more than
half of every answer's context is documents no case asked for; the old summary
line could not show that either. And passage recall at 100% on the hard set is
genuine good news: when the right document is retrieved, the chunk holding the
answer comes with it.

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
