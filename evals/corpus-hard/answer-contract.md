# What an answer is allowed to say

Grounded means: no source, no claim. The model is given retrieved chunks and
told to answer only from them.

When the sources do not contain the answer, the model must emit the refusal
marker rather than answer from its own knowledge. The marker is
`NOT_IN_SOURCES:` and it must appear at the start of the reply.

Citations are numeric and one-based, pointing at the sources list. A citation
number outside the range of that list is an invalid citation and is counted as a
defect, not rounded down.

An answer that retrieves sources but cites none is flagged as unsupported. That
is a different state from a refusal and the two must not be collapsed: a refusal
is correct behaviour, an unsupported answer is a failure.
