# modelwire

modelwire is a small async client for the OpenAI Responses, Anthropic Messages, and System One APIs,
and servers compatible with them.
The OpenAI and Anthropic APIs serve generative models,
and the System One API serves decision models, such as TypeSafe's Jev.
Each client has one request method, `ask()`.
modelwire requires Python 3.15+ and depends only on [httpx2](https://pypi.org/project/httpx2/).

```sh
uv add "modelwire @ git+https://github.com/eyayaw/modelwire@v0.1.0"
```

## Generative models

`OpenAI` and `Anthropic` share the method `ask(model, prompt, *, system="", schema=None, max_tokens=4096)`.
Pass a JSON Schema to `schema` for structured outputs.

```python
import asyncio
import json
import os

from modelwire import Anthropic

SCHEMA = {
    "type": "object",
    "properties": {"remote": {"type": "boolean"}},
    "required": ["remote"],
    "additionalProperties": False,
}
PROMPT = "Is this job remote? The posting says: work from anywhere."


async def main():
    async with Anthropic(os.environ["ANTHROPIC_API_KEY"]) as client:
        reply = await client.ask(
            "claude-haiku-5-5", PROMPT, system="Answer in JSON.", schema=SCHEMA
        )
    print(json.loads(reply.text), reply.input_tokens)


asyncio.run(main())
```

Similarly, for OpenAI models, change only the class and the model name:

```python
from modelwire import OpenAI

async with OpenAI(os.environ["OPENAI_API_KEY"]) as client:
    reply = await client.ask("gpt-6.1-terra", PROMPT, system="Answer in JSON.", schema=SCHEMA)
```

The returned `Reply` contains `text`, `input_tokens`, `output_tokens`,
`cache_read_tokens`, and `cache_write_tokens`.
`input_tokens` includes cached tokens on both APIs, and a count omitted by the server is zero.
Put the fixed part of your prompts in `system`, which `Anthropic` marks for caching.
See the provider guides for structured outputs and prompt caching:
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching),
[Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
and [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## Decision models

`SystemOne.ask(model, state, questions)` returns a dictionary with exactly one answer per question,
each checked for its type's numeric fields.
For the question types and answer fields,
see the [System One reference](https://docs.typesafe.ai/api/system-one.md).

```python
import asyncio
import os

from modelwire import SystemOne

QUESTIONS = {
    "remote": {"type": "noul", "instructions": "Is the job remote?"},
    "level": {"type": "score", "instructions": "How senior?", "criteria": ["Junior.", "Senior."]},
}
POSTING = "Senior data scientist. The posting says: work from anywhere."


async def main():
    async with SystemOne(os.environ["TYPESAFE_API_KEY"]) as client:
        answers = await client.ask("jev-latest", POSTING, QUESTIONS)
    print(answers["remote"]["noul"], answers["level"]["score"])


asyncio.run(main())
```

## Servers and retries

Pass `base_url` to use another compatible server,
and leave out the key for a local server if not required:

```python
client = OpenAI(base_url="http://localhost:8000/v1")
```

Use each instance with `async with`, or release its connections with `await client.close()`.
The default is two retries for connection and write failures and HTTP 408, 409, 429, and 5xx;
set `retries` to change it.
A server's `x-should-retry` header overrides the status,
and a requested retry delay over 60 seconds raises `APIError` immediately.
Read failures are not retried, because the server can still be processing the request.

## Errors

Catch `APIError` for an HTTP error or a response without a reply;
its `status_code`, `retry_after`, and `request_id` attributes describe the response.
Catch `ReplyError` for an unusable reply,
including a refusal, truncated text, or an incomplete answer.
Transport failures raise `httpx2.TransportError`, and invalid client arguments raise `ValueError`.

## Not included

modelwire does not provide conversations, tools, images, streaming, a synchronous API,
reply caching, concurrency limits, extra request headers, or validation of replies against a schema.
Anthropic keys that are not scoped to a workspace are not supported.
