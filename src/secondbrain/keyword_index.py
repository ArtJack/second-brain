"""A persisted keyword index, so keyword search stops reading the whole corpus.

Before this, every hybrid query pulled every chunk out of the store and scored it
in Python: on the real collection that was 7.2 MB over the network and about a
second of work before the embedding call had even started, and it grew with the
corpus. It also could not see half of what the owner writes, because the
tokenizer was `[a-z0-9]+` — a Russian question tokenized to nothing and silently
got no keyword rescue.

SQLite's FTS5 answers both. The index lives on disk beside the task database, it
is written incrementally as files are ingested, and `unicode61` tokenizes text
rather than ASCII. Each collection gets its own table, so the personal and
reference corpora cannot see each other here any more than they can in the store.

One rule governs everything below: **user text never reaches MATCH unquoted.**
FTS5's match syntax has operators (`AND`, `OR`, `NEAR`), quoting and prefix rules,
and a stray double quote in a question is a syntax error, not a search. Queries
are therefore tokenized in Python and rebuilt as quoted terms.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .config import cfg

# `\w+` under Python's default Unicode semantics matches Cyrillic, Greek and CJK
# as readily as ASCII — the whole point of replacing the old [a-z0-9]+ tokenizer.
_WORD = re.compile(r"\w+", re.UNICODE)

# A run of words joined the way identifiers are written: `XX-F-12`, `192.0.2.7:9`,
# `1999-01-02`. Not by `/`, which separates rather than joins: `XX-F-12/13` names
# two identifiers, and as one phrase it would match neither.
_SPAN = re.compile(r"\w+(?:[-.:]\w+)*", re.UNICODE)

# Terms carried over from the hand-rolled scorer this replaces. Dropping them is
# not cosmetic: the match expression joins terms with OR, so leaving "what", "is"
# and "the" in a question makes every document containing them a candidate and
# dilutes BM25 across the corpus. Measured on the regression benchmark, keeping
# them moved a correct answer from rank 1 to rank 3.
STOPWORDS = {
    "about", "all", "and", "are", "can", "did", "does", "for", "from", "how",
    "into", "is", "its", "me", "my", "of", "or", "that", "the", "their", "them",
    "this", "to", "what", "when", "where", "which", "who", "why", "with", "you",
    "your",
}


def query_terms(text: str) -> list[str]:
    """The words worth searching for, or every word if that leaves nothing.

    The fallback matters for two cases: a question made entirely of stopwords,
    and any language whose function words are not in an English stopword list —
    a Russian query keeps all of its terms, which is correct, because none of
    them are in the set above.

    Identifiers are the exception to the length filter, because their short parts
    are the most specific words a question has. A production question that named an
    identifier lost every part of it to the filter and searched only its common
    words, which matched hundreds of chunks and left the few containing the
    identifier out of the top twenty. So a span with a digit in it is kept whole,
    and `query` quotes it like any other term, which FTS5 reads as a phrase: the
    tokens adjacent and in order. As loose parts it would match almost anything,
    because one- and two-character tokens are among the commonest in the corpus.
    The digit is what separates an identifier from `re-ingest`, so a question
    without one searches exactly the terms it did before. A span the filter
    already kept, such as `8765`, is not added again, which would double its weight.

    When the filter leaves nothing but identifiers, the question's other short
    words stay, as the fallback kept them before: a pull request asked for by
    number still searches `pr` beside the number. Stopwords stay out, and so do
    single letters, which are fragments such as the `s` of "what's", and the pieces
    of a joined identifier, which the identifier itself stands in for.
    """
    lowered = text.lower()
    words = _WORD.findall(lowered)
    meaningful = [word for word in words if word not in STOPWORDS and len(word) > 2]
    spans = [span for span in dict.fromkeys(_SPAN.findall(lowered)) if re.search(r"\d", span)]
    identifiers = [span for span in spans if span not in meaningful]
    if not identifiers:
        return meaningful or words
    if not meaningful:
        pieces = {word for span in spans for word in _WORD.findall(span)}
        return [word for word in words if len(word) > 1 and word not in STOPWORDS and word not in pieces] + identifiers
    return meaningful + identifiers


def _require_fts5() -> None:
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x, tokenize='unicode61 remove_diacritics 2')")
    except sqlite3.OperationalError as exc:  # pragma: no cover - depends on the interpreter build
        raise RuntimeError(
            "this Python's sqlite3 was built without FTS5, which the keyword index needs. "
            "Install a Python with FTS5 (the python.org and Homebrew builds have it), "
            "or set SB_HYBRID=0 to run vector-only."
        ) from exc


# The shape a table must have to be usable. A table that does not match is from
# an older version and is migrated in place, rows and all, before use.
_COLUMNS = ("source", "chunk", "name", "header", "document")


def _table(collection: str) -> str:
    """A collection name as a safe table identifier."""
    return "kw_" + re.sub(r"\W+", "_", collection)


def default_index_path() -> Path:
    return cfg.state_db.parent / "keyword-index.sqlite3"


class KeywordIndex:
    def __init__(self, path: Path | None = None) -> None:
        _require_fts5()
        self.path = Path(path) if path is not None else default_index_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._known: set[str] = set()

    def _connect(self) -> sqlite3.Connection:
        # WAL, because the MCP tools became genuinely concurrent: ingest writes
        # this file from one worker thread while ask reads it from another. In
        # the default rollback-journal mode a write holds an EXCLUSIVE lock for
        # the whole transaction and a reader waits only busy_timeout before
        # raising `database is locked` — which reached the client as a hard error
        # on the user-facing path. WAL lets readers proceed during a write; the
        # longer timeout covers the checkpoint.
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure(self, conn: sqlite3.Connection, collection: str) -> str:
        table = _table(collection)
        if table not in self._known:
            # Brought up to the current columns with its rows, never dropped:
            # `_migrate` records what dropping cost.
            self._migrate(conn, table)
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
                "source UNINDEXED, chunk UNINDEXED, name UNINDEXED, header, document, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            self._known.add(table)
        return table

    def _migrate(self, conn: sqlite3.Connection, table: str) -> None:
        """Bring a table built by an older version up to the current columns, keeping its rows.

        The first version dropped any table whose columns did not match and said
        "the next ingest or reindex" would refill it. Only a reindex refills a
        whole collection; an ingest refills the files it touched. So the nightly
        after a schema change would have replaced a 3,820-row index with the few
        dozen rows of that night's changed files, and keyword search would have
        silently covered a sliver of the corpus. Reads were no better: a query
        naming the new column against an old table raised, retrieval swallowed the
        error, and every answer went vector-only without a line in any log.

        FTS5 cannot add a column, so the rows are copied into a table of the
        current shape and the old one is dropped. A column the old table lacks is
        filled with empty strings; a reindex fills it properly.
        """
        existing = self._columns(conn, table)
        if existing is None or existing == _COLUMNS:
            return
        scratch = f"{table}__migrating"
        conn.execute(f"DROP TABLE IF EXISTS {scratch}")
        conn.execute(
            f"CREATE VIRTUAL TABLE {scratch} USING fts5("
            "source UNINDEXED, chunk UNINDEXED, name UNINDEXED, header, document, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        carried = ", ".join(column if column in existing else "''" for column in _COLUMNS)
        conn.execute(f"INSERT INTO {scratch} ({', '.join(_COLUMNS)}) SELECT {carried} FROM {table}")
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {scratch} RENAME TO {table}")

    def _columns(self, conn: sqlite3.Connection, table: str) -> tuple[str, ...] | None:
        """The existing table's columns, or None if it does not exist."""
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if not row:
            return None
        return tuple(info[1] for info in conn.execute(f"PRAGMA table_info({table})"))

    def _exists(self, conn: sqlite3.Connection, collection: str) -> str | None:
        table = _table(collection)
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if not row:
            return None
        # A read must never meet an old-shaped table: querying a column it
        # lacks raises, and retrieval turns that into a silent vector-only
        # answer. Migrating here costs one copy, once.
        self._migrate(conn, table)
        return table

    def upsert_chunks(self, collection: str, rows: list[dict], replace: bool = True) -> None:
        """Write these rows, by default replacing everything already held for their sources.

        `replace` is the contract, and it is a per-source wholesale replace, not
        a per-chunk one: a caller hands over *all* of a file's chunks in one
        call, and what was there before is gone. Ingest works that way. A caller
        that must split one source across several calls has to pass
        `replace=False` and clear the ground itself, or each call will delete
        what the previous one just wrote.
        """
        if not rows:
            return
        with self._connect() as conn:
            table = self._ensure(conn, collection)
            # One DELETE per source, not per chunk. `source` and `chunk` are
            # UNINDEXED FTS5 columns with no b-tree behind them, so each DELETE
            # scans the whole table: deleting per chunk made an ingest quadratic
            # in corpus size, and the transaction that held the write lock grew
            # with it. Re-ingesting a file replaces all of its rows anyway.
            if replace:
                for source in dict.fromkeys(row["source"] for row in rows):
                    conn.execute(f"DELETE FROM {table} WHERE source = ?", (source,))
            conn.executemany(
                f"INSERT INTO {table} (source, chunk, name, header, document) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        row["source"],
                        row["chunk"],
                        row.get("name", ""),
                        row.get("header", ""),
                        row.get("document", ""),
                    )
                    for row in rows
                ],
            )

    def delete_source(self, collection: str, source: str) -> None:
        with self._connect() as conn:
            table = self._exists(conn, collection)
            if table:
                conn.execute(f"DELETE FROM {table} WHERE source = ?", (source,))

    def reset(self, collection: str) -> None:
        with self._connect() as conn:
            table = self._exists(conn, collection)
            if table:
                conn.execute(f"DROP TABLE {table}")
                self._known.discard(table)

    def count(self, collection: str) -> int:
        with self._connect() as conn:
            table = self._exists(conn, collection)
            if not table:
                return 0
            return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def query(self, collection: str, query: str, limit: int = 3) -> list[dict]:
        """Top matches as the hit dicts the retrieval layer already speaks."""
        if limit <= 0:
            return []
        terms = query_terms(query)
        if not terms:
            return []
        # Every term quoted, joined by OR: no operator in the user's words can
        # change the shape of the query, and an unbalanced quote cannot reach FTS5.
        match = " OR ".join(f'"{term}"' for term in terms)
        with self._connect() as conn:
            table = self._exists(conn, collection)
            if not table:
                return []
            rows = conn.execute(
                f"SELECT source, chunk, name, header, document, bm25({table}) AS score "
                f"FROM {table} WHERE {table} MATCH ? ORDER BY score LIMIT ?",
                (match, limit),
            ).fetchall()
        hits: list[dict] = []
        for row in rows:
            # bm25() is negative and more negative is better; flip it so the
            # distance below reads the same way cosine distance does.
            relevance = -float(row["score"])
            hits.append(
                {
                    # The body only. The header is a retrieval aid: it is matched
                    # against, and it is not part of what a citation quotes.
                    "document": row["document"],
                    "metadata": {
                        "source": row["source"],
                        "chunk": row["chunk"],
                        "name": row["name"],
                        "header": row["header"],
                    },
                    "distance": 1 / (1 + relevance) if relevance > 0 else 1.0,
                    "retrieval": "keyword",
                }
            )
        return hits


def rebuild_from_store(store, index: KeywordIndex | None = None, batch_size: int = 500) -> int:
    """Populate the index for a collection that was ingested before it existed.

    Every collection in the live deployment predates this module, so without a
    backfill the index stays empty, `keyword_query` keeps taking the fallback
    scan, and nothing improves until every file happens to change. This pays the
    full-collection read exactly once instead of on every query.
    """
    index = index or KeywordIndex()
    collection = getattr(store, "collection_name", None)
    if not collection:
        raise ValueError("store does not expose a collection name")
    # Dropping the table first is what makes the `replace=False` below safe *and*
    # necessary: there is nothing left to replace, and a file whose chunks span
    # more than one batch would otherwise have each batch delete the last.
    index.reset(collection)
    rows: list[dict] = []
    written = 0
    for hit in store.documents():
        meta = hit.get("metadata") or {}
        source = meta.get("source")
        if not source:
            continue
        rows.append(
            {
                "source": source,
                "chunk": meta.get("chunk", 0),
                "name": meta.get("name", str(source).rsplit("/", 1)[-1]),
                "header": meta.get("header", ""),
                "document": hit.get("document", ""),
            }
        )
        if len(rows) >= batch_size:
            index.upsert_chunks(collection, rows, replace=False)
            written += len(rows)
            rows = []
    if rows:
        index.upsert_chunks(collection, rows, replace=False)
        written += len(rows)
    return written
