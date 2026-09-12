# These documents are fiction

Every file in this directory was written to be *found* by `evals/hard.json`, not
because anyone measured what it says. The gateway's budget alert does not fire
at 80 percent; restic does not run at 03:30; `GW-KEY-GEN-7` is not a key
generation. The numbers exist so a benchmark case has something specific to
retrieve.

They are deliberately shaped like the owner's real notes — same voice, same
structure, same kind of detail — because a fixture that reads like a fixture
tests nothing. That resemblance is exactly why this directory must never be
ingested into a real collection: a reader cannot tell these apart from notes
the owner wrote, and the brain's whole promise is that a citation points at
something true.

The nightly scan excludes `corpus` and `corpus-hard` by directory name for this
reason, and `sb gc --enforce-rules` removes them from a collection that already
holds them.

The same applies to `evals/corpus/`, which is the older and smaller fixture set.
