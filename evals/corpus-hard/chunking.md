# Chunking

Default chunk size is 1200 characters with 150 characters of overlap. The
overlap exists so a sentence spanning a boundary is retrievable from either
side.

Chunks are keyed by source path and ordinal, so re-ingesting a file replaces its
chunks rather than duplicating them. A file edited to produce fewer chunks would
otherwise leave the old higher-numbered chunks behind as orphans.
