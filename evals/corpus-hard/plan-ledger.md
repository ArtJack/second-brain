# The planning ledger

Ids are minted once from the maximum existing id including closed rows, never
from the count of open ones. Reusing an id silently rewrites history.

Confidence is frozen at filing: proven, probable, or hypothesis. It is not
revised later to match the outcome, because a confidence that moves to match
results measures nothing.

Outcomes are written by the session that executed the work, not by the agent
that proposed it. A percentage is reported only past ten settled rows; below
that it is counts.
