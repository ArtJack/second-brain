# second-brain eval results

Date: 2026-06-07 10:41 PDT

Benchmark: `evals/regression.json` with 26 cases against the throwaway
`second_brain_regression` collection. Retrieval metrics score 22 answerable cases; abstention
cases are skipped for retrieval scoring.

Model/config: `nomic-embed-text` embeddings through the configured Qdrant backend. This was a
retrieval-only eval, so `llama3.1:8b` was configured but not used for answer generation.

| metric | vector-only (`SB_HYBRID=0`) | hybrid (`SB_HYBRID=1`) |
|---|---:|---:|
| retrieval_hit_rate | 1.0 | 1.0 |
| mean_source_recall | 1.0 | 1.0 |
| mrr | 0.9697 | 1.0 |
| retrieval_passed / retrieval_cases | 22 / 22 | 22 / 22 |

Hybrid tied vector-only on hit rate and source recall for this corpus, and improved MRR by moving
the first relevant source to rank 1 for every scored retrieval case.
