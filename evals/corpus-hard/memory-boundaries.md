# Memory write boundaries

A fact learned against the default collection is written to the memory
directory. A fact learned against any other collection is written to a
subdirectory named for that collection, so a sandbox visitor's text never lands
beside the owner's own notes.

The nightly scan ingests the memory directory into the default collection, and
it skips those subdirectories deliberately. Walking them would put a visitor's
text into the owner's real brain by the back door, which is the exact hole the
subdirectories exist to close.

Provenance is recorded in the file: `source: user` for a fact the owner typed,
`source: mcp` for one written by a tool call. They must be distinguishable.

Ingest accepts only paths under the configured roots, resolved first, so a
symlink pointing outside is refused rather than followed.
