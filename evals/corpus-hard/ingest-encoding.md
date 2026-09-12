# How files are decoded

The ladder is UTF-8, then cp1252, then latin-1. UTF-16 is attempted **only**
behind a byte-order mark.

That exception is the whole point. UTF-16 decodes almost any even-length byte
sequence without raising, so trying it speculatively turns roughly half of all
non-UTF-8 notes into dense CJK mojibake, which is then embedded and cited as if
it were the owner's own words. A decode producing a NUL byte is rejected for the
same reason.

A file that no encoding decodes raises, naming the file and every encoding
tried, rather than being silently skipped.
