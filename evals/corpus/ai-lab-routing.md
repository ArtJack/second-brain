# Synthetic fixture: AI lab routing

This file is regression-test data, not a learned user memory. The lab below is invented —
two made-up hosts and an invented budget — so the retrieval cases have stable, specific
facts to find without documenting anyone's real machines.

The LiteLLM gateway is `http://127.0.0.1:4000`. The `code` alias routes to `workstation-01`
running qwen-coder. The `chat` alias routes to `gpu-01` running llama3.1. The `embed` alias
also runs on `gpu-01` using nomic-embed-text. The paid Claude fallback has a $25 cap per
30-day period. Keeping one role warm on each machine avoids model reload delays.
