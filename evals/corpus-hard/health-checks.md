# What the health command checks

Deployed service labels are read from the launchd plists and the installer
scripts rather than from a hand-maintained list, so a service that exists but
was never added to the list cannot hide.

The report leads with the change since the last run: newly failing, recovered,
unchanged. A wall of green with one new red buried in it is a report nobody
reads.

Health is not a CI runner. It reports state; it does not fix it and does not
restart anything.
