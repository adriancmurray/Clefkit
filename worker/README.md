# Clefkit worker (Workers AI)

`worker/` is a local SystemOne endpoint. `wrangler dev` serves `POST /v1/systemone` and `GET /health` on `127.0.0.1:8788`. The AI binding answers each request with Cloudflare's Clef models (`clef`, `clef-flash`). The worker has no auth, so it binds to loopback only and is not meant to be deployed as is.

## Run by hand

    cd worker
    wrangler dev --port 8788 --ip 127.0.0.1

Check it with `curl http://127.0.0.1:8788/health`, which returns `{"ok":true,"backend":"workers-ai"}`.

## Start at login (LaunchAgent)

    worker/launchd/install.sh

This writes `~/Library/LaunchAgents/com.clefkit.workers-ai.plist` and loads it. The agent runs the worker through a login shell, so wrangler's PATH and login config are found. It starts at login and restarts if it exits, at most once every 30 s. The script waits up to 60 s for `/health` and prints `ok`, or the last 20 log lines on failure.

- Stop any hand-started `wrangler dev` on port 8788 first, or the agent cannot bind.
- Run `install.sh` again after moving the repo or changing where wrangler is installed. It rewrites the plist.
- Uninstall with `worker/launchd/uninstall.sh`. It stops the agent and removes the plist.

## Log

`~/Library/Logs/clefkit-workers-ai.log` holds both stdout and stderr.

## Auth and quota

The worker stores no token. It uses your wrangler login (`wrangler login`), and the AI binding calls Cloudflare under that login. If the login expires, requests fail with a 502 and the log shows why. Each decision uses a little of the Workers AI daily allocation.
