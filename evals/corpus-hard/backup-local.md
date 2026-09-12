# Local snapshot policy (not the offsite one)

Local Qdrant snapshots are taken **before any destructive operation** and
before each weekly maintenance window. They live on the same disk as the
collection they snapshot, which is precisely why they are not a backup: a disk
failure takes both.

Retention is 3 snapshots. Older ones are pruned automatically.

This document is about local snapshots only. The offsite schedule is a separate
policy and a different time of day.
