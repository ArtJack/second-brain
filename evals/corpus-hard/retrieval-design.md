# How retrieval is put together

The pipeline is: chunk, embed, store, retrieve, fuse, answer, validate.

Fusion is **reciprocal rank fusion with k=60**, not a weighted sum of scores.
The reason is that cosine distance and a BM25-derived score share no scale, so
adding them means inventing a conversion that no measurement supports. RRF only
reads ranks, which both retrievers genuinely have.

Adjacency expansion runs after fusion, never inside it. A neighbouring chunk is
context, not evidence: it matched nothing, so it may only fill slots the
retrievers left empty. Letting neighbours compete at retrieved ranks cost three
of five answer slots at the shipped default once.

The keyword side is SQLite FTS5 with the `unicode61 remove_diacritics 2`
tokenizer. The previous hand-rolled scorer used `[a-z0-9]+`, which tokenized a
Russian question to nothing at all and silently returned no keyword rescue.

Stopwords are filtered before the match expression is built. Terms are OR-joined,
so leaving "what", "is" and "the" in makes every document a candidate and
dilutes BM25 across the corpus.
