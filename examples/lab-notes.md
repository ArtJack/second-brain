# Home AI Lab — decisions log (demo note)

## Model routing
We route the **`code`** model to the **M4 Mac mini** (qwen-coder), and **`chat`** plus
**`embed`** to the **Alienware GTX 1070** (llama3.1 and nomic-embed). Each box keeps one
role warm so an agent never waits on a model reload. The reason is the M4 only has 16 GB,
so it swap-deaths if it tries to hold two large models at once.

## Gateway
Everything goes through one LiteLLM gateway at `http://192.168.1.159:4000`. Postgres was
locked down to host-local only. The paid Claude route has a $50 / 30-day budget cap.

## Career goal
Target role is **AI Agent Engineer**, comp band $160–250k. The biggest portfolio gap is a
deployed product with a public URL. The chosen vertical is trucking/logistics (IFTA filing,
Bill-of-Lading extraction), because that's a niche with real paying customers.
