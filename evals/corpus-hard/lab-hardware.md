# The machines

`mini` is an M4 Mac mini. It runs the gateway, the brain's MCP server, Qdrant,
and one Ollama instance. It is the machine the owner works on.

`ai-lab` is the former Alienware, now on Ubuntu. It carries the GPU and the
larger chat models. It was on Windows past end-of-life until the migration.

Two Oracle VMs carry production. Disk on `mini` runs near full, so the rule is
move, never copy.
