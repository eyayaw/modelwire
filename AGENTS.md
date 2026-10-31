# modelwire agent instructions

## Purpose

modelwire is a small async client for OpenAI-, Anthropic-, and System One-compatible APIs,
with one class per API and one request method, `ask()`.

## Map

```
_Client      checks the URL and the arguments, owns the httpx2 client, _post retries
  OpenAI     ask(): request body --> _post("responses") --> _responses_reply
  Anthropic  ask(): request body --> _post("messages") --> _messages_reply
  SystemOne  ask(): request body --> _post("systemone") --> _complete per question
```

All code is in `src/modelwire/__init__.py`.
The tests in `tests/test_modelwire.py` run each class against `httpx2.MockTransport`.

## Rules

- Code that builds a request or reads a response cites the official reference in a comment.
  When a provider changes its API, start from those links, and send or read only what a source documents.
- `_Client` offers services and calls no hooks, so each class reads top to bottom with its own constructor and `ask()`.
- Decoders trust the types the API documents.
  Only error bodies and System One answers, which vary by server and by model, get type checks.
- Nothing is retried after the request is sent, because a second request can pay for the same reply again.

## Non-goals

- Conversations, tools, images, streaming, and a sync API.
- A mode argument that picks the API, since the class is the API.
- Reply caching, concurrency limits, keys from the environment, and extra request headers.
- Checking a reply against its schema.
- Runtime dependencies other than httpx2.

## Verify

    prek run --all-files

Tests call no real model API, because each call costs money.
