# second-brain web

Standalone Next.js App Router client for the cited-Q&A demo.

## Local run

```bash
npm install
npm run dev
```

Copy `.env.example` to `.env.local` and set:

- `BRAIN_API_URL`: the FastAPI `sb-web-api` origin.
- `BRAIN_API_TOKEN`: server-side bearer token forwarded by Next API routes.
- `NEXT_PUBLIC_TURNSTILE_SITE_KEY` / `TURNSTILE_SECRET_KEY`: Cloudflare Turnstile keys.

If the API or env is unavailable, the app renders committed fallback answers for the suggested prompts.

## Deploy

Create a standalone Vercel project rooted at `web/`. Set the same environment variables in Vercel. The browser only calls `/api/brain/*`; direct access to the lab API stays server-side.
