# The MCP tool surface

Nine tools: recall, ask, learn, forget, ingest, status, add_task, list_tasks,
complete_task.

Every tool is async and offloads its blocking work to a thread, because a
synchronous tool runs inline on the event loop and blocks every other request
for its duration.

Write tools are annotated as writes. `forget` is a write and is deliberately not
a general-purpose delete: it refuses any path outside the memory directory.
