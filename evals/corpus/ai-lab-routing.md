# Synthetic fixture: AI lab routing

This file is regression-test data, not a learned user memory.

The LiteLLM gateway is `http://127.0.0.1:4000`. The `code` alias routes to the M4 Mac
mini running qwen-coder. The `chat` alias routes to the Alienware GTX 1070 running
llama3.1. The `embed` alias also runs on the Alienware using nomic-embed-text. The paid
Claude fallback has a $50 cap per 30-day period. Keeping one role warm on each machine
avoids model reload delays.
