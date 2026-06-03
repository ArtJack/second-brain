# Home AI Lab — decisions log (demo note)

## Model routing
We route the **`code`** model to the **M4 Mac mini** (qwen-coder), and **`chat`** plus
**`embed`** to the **Alienware GTX 1070** (llama3.1 and nomic-embed). Each box keeps one
role warm so an agent never waits on a model reload. The reason is the M4 only has 16 GB,
so it swap-deaths if it tries to hold two large models at once.

## Gateway
Everything goes through one LiteLLM gateway at `http://127.0.0.1:4000`. Postgres was
locked down to host-local only. The paid Claude route has a $50 / 30-day budget cap.
