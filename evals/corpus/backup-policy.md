# Synthetic fixture: backup policy

This file is regression-test data, not a learned user memory. The path below is an
illustrative mount point, not anyone's real backup location.

Back up learned Markdown memories and the SQLite state database nightly. Export a Qdrant
collection snapshot weekly. Store backups on the external volume under
`/mnt/backups/second-brain` and perform a restore drill once per month.
