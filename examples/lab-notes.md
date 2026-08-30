# Home AI Lab — decisions log (demo note)

Example input document, invented for demonstration: it shows the shape of a note worth
ingesting — a decision, and the reason behind it — using the made-up hosts from the eval
fixtures rather than any real machine.

## Model routing
We route the **`code`** model to **`workstation-01`** (qwen-coder), and **`chat`** plus
**`embed`** to **`gpu-01`** (llama3.1 and nomic-embed). Each box keeps one role warm so an
agent never waits on a model reload. The reason is that the workstation is memory-bound and
swap-deaths if it tries to hold two large models at once.

## Gateway
Everything goes through one LiteLLM gateway at `http://127.0.0.1:4000`. Postgres was locked
down to host-local only. The paid Claude route has a $25 / 30-day budget cap.
