# Workers AI backend

Runs Clef on Cloudflare Workers AI through the REST API
(`POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash`).

## Setup
1. Note your Cloudflare account id.
2. Create an API token with the "Workers AI" template (a custom token needs Workers AI - Read and Edit), per Cloudflare's REST API page.
3. Export `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (or other names via `account_id_env` / `token_env` in the backend config). Credentials are never read from the config file.

Models: `clef-flash` (default, `@cf/cloudflare/clef-flash`) and `clef` (`@cf/cloudflare/clef`). Images must be data: URLs (max 4, PNG/JPEG/WebP); remote URLs are rejected by Cloudflare.

## Free allocation and cost
Read 2026-10-12 from https://developers.cloudflare.com/workers-ai/platform/pricing/:
- Free: 10,000 Neurons per day; beyond that $0.011 per 1,000 Neurons on the Workers Paid plan.
- clef-flash: $0.038 per M input tokens (3,455 Neurons per M). clef: $0.240 per M input tokens (21,818 Neurons per M). Output tokens are not charged.

Arithmetic from those numbers: a 1,000-token call costs 3.455 Neurons on clef-flash (about 2,894 such calls inside the daily free 10,000 Neurons) and 21.818 Neurons on clef (about 458 calls). An image adds up to 1,024 tokens (width/32 x height/32 + 3).
