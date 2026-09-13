# second-brain eval results

## 2026-09-12 — `recall` retrieves through `hybrid_retrieve`

The MCP server tells agents that `recall` is the cheapest way to ground
themselves, and `recall` searched vectors only. `ask` retrieves through
`hybrid_retrieve`; `recall` called `store.query` directly and ignored
`SB_HYBRID`, so an agent grounding itself through it never got the keyword half
of retrieval. It now makes the same `hybrid_retrieve` call `ask` makes.

`sb eval` could not show this change, or the gap it closes. It scores
`hybrid_retrieve` directly, so no reading in this file ever measured the path
`recall` took. `sb eval --via-recall` scores what `recall` itself returns, and
the JSON report now names the path that ran as `retriever`.

Retrieval only, k=5, Qdrant, embeddings through the LiteLLM gateway, chat model
not called. Both benchmarks were ingested once into collections no other session
uses, `second_brain_recall_hybrid_hard` and `second_brain_recall_hybrid_regression`,
and every reading below ran against that one state: before and after the change,
through both paths, with `SB_HYBRID` on and off. The absolute numbers differ from
earlier entries, precision most, because the shared `second_brain_regression`
collection holds two copies of its eight fixtures. Compare before with after
here, not with the entries below.

### The recall path, `SB_HYBRID=1`

| metric | hard before | hard after | regression before | regression after |
|---|---:|---:|---:|---:|
| passed / cases | 13 / 18 | **14 / 18** | 21 / 22 | **22 / 22** |
| hit_rate | 72.2% | **77.8%** | 95.5% | **100%** |
| source_recall | 94.4% | **100%** | 95.5% | **100%** |
| mrr | 0.861 | **0.870** | 0.932 | **1.000** |
| mean_precision | 30.0% | 30.0% | 22.7% | **23.6%** |
| passage_recall | 94.1% | **100%** | not asked | not asked |
| distractor_rate | 100% | 100% | not asked | not asked |

One case moved on each: `adjacency-is-not-evidence` on the hard set and
`lab-paid-cap` on the regression set. Both already passed through `ask`'s
retrieval, and neither passed through `recall`.

Per-case results, compared field by field (sources, ranks, passages, precision):

| comparison | hard, 20 cases | regression, 26 cases |
|---|---:|---:|
| before, on 0754a16: recall at `SB_HYBRID=1` = `sb eval` at `SB_HYBRID=0` | 20 identical | 26 identical |
| after: recall at `SB_HYBRID=1` = `sb eval` at `SB_HYBRID=1` | 20 identical | 26 identical |
| after: recall at `SB_HYBRID=0` = `sb eval` at `SB_HYBRID=0` | 20 identical | 26 identical |
| `sb eval` on 0754a16, before = after, at either setting | 20 identical | 26 identical |

Before, `recall` searched vectors only, whatever the flag said. After, it returns
exactly what `ask` retrieves and still obeys the flag. The last row is why
`sb eval` alone could not gate this: it reads the same on both sides of the change.

Rebased onto #48 and read again, every benchmark reading above came back identical
case for case, and `sb eval` at `SB_HYBRID=0` still matches `recall` before this
change on all 46 cases. The two rows marked 0754a16 need the code from before this
change, which the rebased branch no longer has.

### What the benchmark could not measure

On the live `second_brain_v2` collection, read-only, through the `recall` tool
function and again through a stdio `sb-mcp` session started from the change,
which returned the same hits. In all four probes `recall` returned exactly what
`hybrid_retrieve` did. These readings were taken again after rebasing onto #48,
which keeps identifiers whole in keyword search, with the collection at 3,941
chunks. "Before" is the vector-only top 5, which is what `recall` returned before
this change.

The probe queries are described below rather than quoted. This file is itself
ingested, and a quoted query makes it the strongest keyword match for that query
while it says nothing about the subject.

| probe | hits from outside the vector top 5 | top-5 chunks matching the target |
|---|---:|---:|
| the plain-language question from the defect report | 0 → 3 | n/a |
| an issue id built from hyphenated one- and two-character parts | 0 → 3 | **0 → 3** holding the id |
| a two-word Cyrillic query | 0 → 2 | 5 → 5 with any Cyrillic text |
| a lookup by numbered heading | 0 → 3 | 0 → 2 holding the heading |

Before, every distance sat in the vector range, 0.33 to 0.48. After, each result
interleaves keyword hits at 0.05 to 0.15 with vector hits at 0.33 to 0.46. The
plain-language probe now returns keyword hits at 0.150 to 0.152 beside vector
hits at 0.434 and 0.435.

**With #48, the id probe finds the id.** Keyword search now matches the id as one
term, finds exactly the three chunks that hold it, and ranks them 1st to 3rd, so
all three reach `recall`. On 0754a16, before #48, the same probe returned none of
them. The id split at its hyphens into three very short terms matched separately,
and the chunks holding it ranked 5th to 7th, outside fusion's three keyword
slots. SB-F-31's length filter did not cause that miss, since its fallback kept
all three terms; keeping the id whole is what fixed it. The heading lookup is
different: on 0754a16 the filter dropped its number and one chunk holding the
heading reached the top 5, and with the number kept, two do. `recall` gets
whatever keyword search returns, so #48's gains reach it directly. The Cyrillic
probe needed no rescue, because vector search already found it.

The cost, over the four probes on the rebased tree: the retrieval step's median
rose from 24.2 ms to 30.2 ms (36 samples each), and `recall` end to end, embedding
included, from 151 ms to 161 ms (20 samples each).

### What did not change

`ask`, the fusion logic and the `k` defaults are untouched, and the tool still
returns `count` and `hits` of `source`, `distance` and `text`. Two properties
`ask` already had now apply to `recall` too: a keyword hit's `distance` is
`1/(1+bm25)`, not a cosine distance, and a search across both corpora sorts on
that mixed scale. The web API's `/recall` route calls the same function, and a
test now checks through that route that the public corpus's keyword search never
reaches the owner's collection. The running `com.secondbrain.mcp` and
`com.artjeck.sb-web` services pick this up only on their next restart.

## 2026-09-12 — keyword search keeps identifiers

A production `ask` that named an identifier answered NOT_IN_SOURCES, although three
chunks of the live collection contain it. Every part of the identifier is two
characters or shorter, so the filter that drops short words took all of them, and
keyword search ran on the question's three remaining words. Those words are in 818,
719 and 275 of the 3,939 live chunks, and none of the three chunks containing the
identifier reached the keyword top 20.

A span containing a digit is now kept whole: a hyphenated identifier, an address
with a port, a date, or a bare number. It is quoted like every other term, and
FTS5 reads a quoted run of tokens as a phrase, so an identifier matches its tokens
adjacent and in order. Short ordinary words are still dropped, and a question
without a digit searches exactly the terms it did before; a test pins six such
questions, their terms and their rankings. When the filter leaves nothing but
identifiers, the question's other short words stay, as the old fallback kept them,
so a pull request asked for by number still searches `pr` beside the number.
Stopwords, single letters and the pieces of a joined identifier do not come back.

Short acronyms without a digit are unchanged: `AI`, `Go` and `R` are still dropped,
so SB-F-31 stays open for them.

This entry and the new tests do not name the identifier the production question
asked about, do not quote that question, and keep its common words out of their
examples. The nightly ingests both, and text that concentrates a question's words
outranks the chunks that answer it (SB-F-55).

### Benchmarks: no movement, which is what this change predicts

Same machine, same day, k=5, `SB_HYBRID=1`, Qdrant, each corpus re-ingested for
each reading.

| metric | regression before | regression after | hard before | hard after |
|---|---:|---:|---:|---:|
| retrieval passed / cases | 22 / 22 | 22 / 22 | 14 / 18 | 14 / 18 |
| hit rate | 100% | 100% | 77.8% | 77.8% |
| source recall | 100% | 100% | 100% | 100% |
| MRR | 1.000 | 1.000 | 0.870 | 0.870 |
| precision | 23.6% | 23.6% | 30.0% | 30.0% |
| passage recall | not asked | not asked | 100% | 100% |
| distractor rate | not asked | not asked | 100% | 100% |

Every case's first relevant rank is unchanged as well, and the same four
distractor cases fail. 44 of the 46 benchmark queries search exactly the terms
they did before. The two that change each gain one term with a digit in it:
`abstain-calendar`, an abstention case that is not scored for retrieval, and
`rare-identifier-key-generation`, whose correct source already ranked first. So
these benchmarks guard against the stopword regression coming back, and they
cannot show this fix: neither scores a question with an identifier inside a
sentence, which is the production failure.

**Measured in collections of its own.** These readings come from
`sb_eval_kwident_regression` and `sb_eval_kwident_hard`, not the shared
`second_brain_regression` and `second_brain_hard`. Another session measured at
the same time, and chunk identity includes the source path, so ingesting the
corpus from a worktree would have added a second copy to a shared collection
rather than replacing the first. The private collections hold each fixture once.
`second_brain_regression` holds all eight twice, under the absolute path and
under the relative `evals/corpus`, and the 47.3% precision recorded below, almost
exactly double the 23.6% here, is consistent with both copies being counted.
`second_brain_hard` holds no duplicates, and why its recorded 31.1% differs from
the 30.0% here is not established. Before and after share one collection each,
so the comparison stands.

### The live collection, where the fix shows

Read-only against `second_brain_v2` through its FTS5 index, `main`'s rule against
this change's, k=5. Keyword search contributes its top 3 to fusion.

| question | chunks naming it in keyword top 3, before → after | in hybrid top 5, before → after |
|---|---|---|
| the production question: an identifier inside a sentence | 0 → 3 | 0 → 3 (ranks 1, 3, 5) |
| the same identifier on its own | 0 → 3 (they ranked 5, 6 and 7) | 0 → 3 (ranks 1, 3, 5) |
| another identifier after "what is" | 1, at rank 2 → 1, at rank 1 and alone | rank 3 → rank 1 |
| a section number beside three ordinary words | 2 → 3 | 2 → 3 |

**What it costs.** A bare number is kept too, because that is what a PR or port
number is, and bare numbers are common. A how-to question with a count in it now
also searches the count, a token in 1,466 live chunks: its keyword top 3 holds the
same three chunks, and its hybrid top 5 is identical. A question about a recent
pull request by number now also searches the number, which is in 101 live chunks,
none of them naming that pull request, so its keyword top 3 trades one set of
unrelated chunks for another and three of its hybrid top 5 change the same way.

**Not changed:** the fallback scan in `hybrid.py`, used for a collection with no
keyword index, tokenizes with its own `[a-z0-9]+` and still drops identifiers. A
process whose working directory points a relative `SB_STATE_DB` at an empty index
takes that scan without saying so (SB-F-59). The launchd services and the
stdio MCP server configured in `~/.claude.json` all start in the checkout.

## 2026-09-12 — what the 100% distractor rate actually costs: tokens, not correctness

This corrects the entry below it. When the hard benchmark first reported that all
four distractor cases retrieve the document they name as wrong, I wrote that "the
answer is then built from both". That was an inference, not a measurement, and it
was wrong.

Measured, k=5, `SB_HYBRID=1`, Qdrant, twenty documents, with answers generated:

**Answer rubric 4 of 4 passed, score 100%.** The model gives the correct answer
in every distractor case, including the two where the wrong document is ranked
*first*. Every document and question in this entry belongs to the fictional
benchmark corpus in `evals/corpus-hard/`, not to any real policy. On the
fixture's offsite-backup case, retrieval ranks the fixture's local-snapshot
document first and its offsite document second, and the answer skips source 1
and cites source 2. On the fixture's QA-routing case it cites only the fixture's
current routing document, never the one marked superseded.

So the distractor rate measures **context waste, not wrong answers.** The answer
layer is already doing the disambiguation that ranking is not.

### The ranked lists, which say why no threshold fixes this

| benchmark case | rank 1 | rank 2 | slots 3–5 |
|---|---|---|---|
| `distractor-offsite-schedule` | **wrong** 0.098 | right 0.123 | 0.281 – 0.508 |
| `distractor-local-snapshots-are-not-a-backup` | right 0.083 | **wrong** 0.119 | 0.252 – 0.459 |
| `superseded-qa-routing` | right 0.060 | **wrong** 0.130 | 0.284 – 0.447 |
| `superseded-qa-share-is-frozen` | **wrong** 0.119 | right 0.229 | 0.349 – 0.504 |

Two things fall out of that table.

**A relevance floor cannot fix the distractor problem.** In the
`distractor-offsite-schedule` case the wrong fixture document scores *closer*
than the right one, 0.098 against 0.123. Any threshold that drops one drops the
other. This is not a tuning failure: the two fixture documents are both written
as current within the benchmark, and each is the right answer to the other
case's question. No ranking signal distinguishes them, because none exists in
the documents.

**A relevance floor would fix something else entirely.** Slots 3 to 5 sit at 0.25
to 0.51 while every real match is under 0.23, a clean and consistent gap. Three of
five slots in every one of these cases go to documents that are simply not about
the question — fixture documents on key rotation and health checks returned for
the fixture's backup-timing case. That is the 28.9% precision, and it is the
thing worth fixing.

**No adjacency padding appeared in any slot.** Every one was a real match from one
retriever or the other, so the SB-F-27 fix is holding under measurement.

### What this changes

A supersession demotion rule would address two of the four cases, and those two
were already answered correctly. It is not worth building. The general fix — a
relevance floor, so `k` is a budget rather than a quota — addresses the precision
number instead, which is the real cost: three wasted slots per query in context
the model pays for and then ignores.

That change is **not** made here. It is a ranking change, it is gated on a parked
owner question about whether the current k=5 answer mix is intended, and the
measurement above is what that decision needs rather than a substitute for it.

The honest summary: the system is behaving better than the benchmark's headline
number suggested, and the benchmark was right to make the number visible anyway.

## 2026-09-12 — contextual chunk headers

Each chunk is now embedded and keyword-indexed with a header line naming its
document and the section it sits under — `Gateway runbook › Restarting` — while
the stored body, and therefore anything a citation quotes, is unchanged. The
chunk bodies are byte-identical to what shipped before, which a test pins: this
adds orientation, it does not re-cut the document.

Measured on both benchmarks, same machine, k=5, `SB_HYBRID=1`, Qdrant, corpora
re-ingested for each reading.

| metric | hard before | hard after | regression before | regression after |
|---|---:|---:|---:|---:|
| passed / cases | 14 / 18 | 14 / 18 | 22 / 22 | 22 / 22 |
| hit_rate | 77.8% | 77.8% | 100% | 100% |
| source_recall | 100% | 100% | 100% | 100% |
| mrr | 0.852 | **0.870** | 1.000 | 1.000 |
| mean_precision | 28.9% | **31.1%** | 47.3% | 47.3% |
| passage_recall | 100% | 100% | not asked | not asked |
| distractor_rate | 100% | 100% | not asked | not asked |

A small, real gain on the hard set and no movement on the saturated one, which
is what the gate asked for. The honest reading is that this is a modest change:
ranking improved slightly and nothing regressed.

**It did not touch the distractor problem, and there is a reason worth writing
down.** The benchmark's superseded fixture document declares itself superseded
in its own first heading, so every one of its chunks now carries the
word SUPERSEDED into both the embedding and the keyword index. The signal is
present in the index for the first time. Nothing reads it. A demotion rule for
documents that declare themselves superseded is the obvious next step, and it is
deliberately not in this change: it is a ranking change and it deserves its own
before-and-after rather than being folded into a chunking one.

This change alters embeddings, so it is a re-ingest, not an in-place migration.
The keyword index gains a `header` column; a table built on the older schema is
dropped and refilled by the next ingest or by `sb keyword-reindex` rather than
failing on insert.

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
supersession or recency: on the fixture's QA-routing case it returns the
fixture's current routing document *and* its superseded one, and I inferred the
answer was built from both — the entry above measured that, and it was not. On
the fixture's offsite-backup case it returns the fixture's local snapshot policy
alongside. Source recall stays at
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
