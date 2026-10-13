---
name: decide
description: Get a typed decision (yes/no, one choice from a fixed set, or a score) about a situation from a Clef decision model through the clefkit decide tool, instead of a free-text judgment. Use it when a routing, risk, or triage call should be consistent and checkable.
---

# Typed decisions with clefkit

Use the `decide` tool when you need a structured answer about a state: a yes/no, one option from a fixed set, or a level on an ordered scale. Prefer it to a free-text judgment when the answer drives a branch, such as which model to use, whether to confirm a command first, or how urgent a ticket is.

## Writing questions

- Give each question a short id: 1 to 64 characters of `A-Z a-z 0-9 _ . -`.
- Use `noul` for yes/no. Say in `criteria` what true and false mean.
- Use `choice` for one option from a set. Options must be mutually exclusive, and each needs a criterion that says when to pick it.
- Use `score` for an ordered scale. `criteria[i]` describes level i, counting from 0, so list the levels from least to most.
- Put only what the decision needs into `state`. It can be a string or any JSON value.

## Reading answers

Every answer carries a `confidence`. Treat a confidence below 0.6 as unsure. Do not act on that answer alone: ask the user, or fall back to your own judgment, and say which you did. If `ok` is false, the result lists each backend tried and why it failed; fall back the same way.

## Presets

Pass `preset` instead of `questions`. The `presets` tool lists them:

- `model-routing`: `model` (haiku, sonnet, opus), `effort`, `needs_review`, `delegate`.
- `command-risk`: `risk` (read_only, writes_local, destructive, network, privileged), `needs_confirmation`.
- `triage`: `urgency`, `kind` (bug, feature, question, chore, incident), `actionable`.

If you pass both `questions` and `preset`, the explicit `questions` are used.

## Privacy

`state` is sent to whichever backend is configured. The local server keeps it on this machine. A remote backend, such as Cloudflare Workers AI, sends it off the machine. Do not put secrets in `state`. The `backends_status` tool shows which backends are reachable.
