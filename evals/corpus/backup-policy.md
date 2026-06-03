# Synthetic fixture: backup policy

This file is regression-test data, not a learned user memory.

Back up learned Markdown memories and the SQLite state database nightly. Export a Qdrant
collection snapshot weekly. Store backups on Disk E under `/Volumes/DISK/AI/artjeck/backups`
and perform a restore drill once per month.
