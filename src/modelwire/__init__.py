"""Async clients for OpenAI Responses, Anthropic Messages, Gemini Interactions, and System One."""

import asyncio
import email.utils
import ipaddress
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

import httpx2

__all__ = ["APIError", "Anthropic", "Gemini", "OpenAI", "Reply", "ReplyError", "SystemOne"]

type _JSON = dict[str, Any]

# Only a failure before the request reaches the server is retried.
# After that, the server may be writing a reply that a second request would pay for again.
_RETRY_ERRORS = (
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.PoolTimeout,
    httpx2.WriteError,
    httpx2.WriteTimeout,
)
# The statuses OpenAI's and Anthropic's SDKs retry; 5xx covers Anthropic's 529.
# https://github.com/openai/openai-python#retries
# https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#retries
_RETRY_STATUSES = frozenset({408, 409, 429, *range(500, 600)})
_CUT_OFF = "the reply reached max_tokens; raise it"
_NO_TEXT = "the reply has no text"
# OpenAI's SDK waits 5 s for a connection and 600 s for the rest, since a reasoning model can
# think for minutes: https://github.com/openai/openai-python/blob/main/src/openai/_constants.py
_CONNECT_TIMEOUT = 5.0
# The header OpenAI and Anthropic put the request ID in:
# https://developers.openai.com/api/reference/overview#debugging-requests
# https://platform.claude.com/docs/en/api/errors#request-id
# TypeSafe documents none, but its responses carry x-typesafe-request-id, seen on 2026-10-31.
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-typesafe-request-id")


@dataclass(frozen=True, slots=True)
class Reply:
    """
    Reply text and token usage.

    input_tokens counts the whole prompt, cache_read_tokens and cache_write_tokens included.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


class APIError(Exception):
    """
    The server answered with an HTTP error, or with a body that has no reply.

    A failed generation can arrive as an error object in an HTTP 200 body, so status_code can
    be 200. code is the body's error code when it is a string, such as "server_error".
    retry_after is the wait the server asked for, in seconds, and request_id is the provider's
    ID for the request. Each is None when the server omits it.
    """

    def __init__(self, response: httpx2.Response) -> None:
        body = _parse(response.content)
        self.status_code = response.status_code
        self.code: str | None = None
        # The error object's code, which OpenAI's SDK exposes too, checked on 2026-11-10:
        # https://github.com/openai/openai-python/blob/main/src/openai/_exceptions.py
        match body:
            case {"error": {"code": str(code)}}:
                self.code = code
        self.retry_after = _retry_after(response)
        self.request_id = next(filter(None, map(response.headers.get, _REQUEST_ID_HEADERS)), None)
        super().__init__(f"HTTP {self.status_code} from {response.url}: {_reason(response, body)}")


class ReplyError(Exception):
    """The server replied, but the reply is unusable: refused, truncated, empty, or incomplete."""


class _Client:
    _base_url: str
    _key_header: str
    _key_prefix = ""
    _headers: frozendict[str, str] = frozendict()

    def __init__(
        self,
        key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 600.0,
        retries: int = 2,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """base_url defaults to the provider's own server. A key of None sends no credentials."""
        credentials = _credentials(self._key_header, key, self._key_prefix)
        base_url = self._base_url if base_url is None else base_url
        try:
            url = httpx2.URL(base_url)
        except httpx2.InvalidURL:
            url = httpx2.URL()
        if url.scheme not in {"https", "http"} or not url.host:
            raise ValueError(f"base_url must be an absolute https or http URL, not {base_url!r}")
        # httpx2 would send a username and password as Basic auth, past the clear-text check.
        if url.userinfo:
            raise ValueError("base_url must not contain a username or password; pass the key")
        if url.scheme == "http" and credentials and not _loopback(url.host):
            raise ValueError(f"refusing to send the key in clear text to {url.host}; use https")
        if not timeout > 0:
            raise ValueError(f"timeout must be positive, not {timeout!r}")
        if retries < 0:
            raise ValueError(f"retries must be 0 or more, not {retries!r}")
        self._url = url
        self._retries = retries
        # A followed redirect would carry the key to whichever host the server names.
        self._http = httpx2.AsyncClient(
            base_url=url,
            headers=self._headers | credentials,
            timeout=httpx2.Timeout(timeout, connect=min(timeout, _CONNECT_TIMEOUT)),
            follow_redirects=False,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={str(self._url)!r})"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Release the pooled connections. The client cannot send afterward."""
        await self._http.aclose()

    async def _post(self, path: str, body: _JSON, *, marker: str) -> _JSON:
        """POST body to path and return the JSON reply, which must contain marker and no error."""
        for attempt in range(self._retries):
            try:
                response = await self._http.post(path, json=body)
            except _RETRY_ERRORS:
                wait = None
            else:
                if not _retryable(response):
                    break
                wait = _retry_after(response)
                if wait is not None and wait > 60:
                    break  # The server asks for longer than a retry should wait.
            # Exponential backoff from 0.5 s to 8 s, jittered as OpenAI's SDK computes it:
            # https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py
            await asyncio.sleep(wait or 0.5 * 2 ** min(attempt, 4) * random.uniform(0.75, 1.0))
        else:
            response = await self._http.post(path, json=body)
        data = _parse(response.content) if response.is_success else None
        # A server reports a failure after HTTP 200 as an error object, checked on 2026-11-10:
        # https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_error.py
        # https://openrouter.ai/docs/api/reference/errors-and-debugging
        if not isinstance(data, dict) or marker not in data or data.get("error"):
            raise APIError(response)
        return data


class OpenAI(_Client):
    """Client for OpenAI's Responses API, or any server compatible with it."""

    _base_url = "https://api.openai.com/v1"
    _key_header = "Authorization"
    _key_prefix = "Bearer "

    async def ask(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Reply:
        """
        Return the model's reply to prompt.

        With a schema, the server returns structured outputs, so reply.text is JSON. Raises
        ReplyError for a refused, truncated, or empty reply, APIError for an HTTP error or a
        failed generation, and httpx2.TransportError when the connection fails.
        """
        # The request and its response:
        # https://developers.openai.com/api/reference/resources/responses/methods/create
        body: _JSON = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            # modelwire never continues a response, so the server need not store it.
            "store": False,
        }
        if system.strip():
            body["instructions"] = system
        if schema is not None:
            # The API requires a name for the format.
            json_schema = {"type": "json_schema", "name": "reply", "strict": True, "schema": schema}
            body["text"] = {"format": json_schema}
        return _responses_reply(await self._post("responses", body, marker="status"))


class Anthropic(_Client):
    """Client for Anthropic's Messages API, or any server compatible with it."""

    _base_url = "https://api.anthropic.com/v1"
    # x-api-key is the legacy fallback for this header, checked on 2026-11-10:
    # https://platform.claude.com/docs/en/api/overview
    _key_header = "Authorization"
    _key_prefix = "Bearer "
    # The version the official SDK sends, checked on 2026-10-31:
    # https://platform.claude.com/docs/en/api/versioning
    _headers = frozendict({"anthropic-version": "2026-10-21"})

    async def ask(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Reply:
        """
        Return the model's reply to prompt.

        With a schema, the server returns structured outputs, so reply.text is JSON. Raises
        ReplyError for a refused, truncated, or empty reply, APIError for an HTTP error, and
        httpx2.TransportError when the connection fails.
        """
        # The request and its response:
        # https://platform.claude.com/docs/en/api/messages
        body: _JSON = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if system.strip():
            # The system prompt is the fixed part of a prompt, so it is marked for caching.
            cache = {"type": "ephemeral"}
            body["system"] = [{"type": "text", "text": system, "cache_control": cache}]
        if schema is not None:
            # output_config is generally available and needs no beta header:
            # https://platform.claude.com/docs/en/build-with-claude/structured-outputs
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return _messages_reply(await self._post("messages", body, marker="stop_reason"))


class Gemini(_Client):
    """Client for Google's Interactions API, or any server compatible with it."""

    _base_url = "https://generativelanguage.googleapis.com/v1"
    _key_header = "x-goog-api-key"

    async def ask(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Reply:
        """
        Return the model's reply to prompt.

        With a schema, the server returns structured outputs, so reply.text is JSON. Raises
        ReplyError for a truncated or empty reply, APIError for an HTTP error, such as blocked
        content, and httpx2.TransportError when the connection fails.
        """
        # The request and its response, checked on 2026-11-10:
        # https://aistudio.google.com/api-reference/interactions-api/create-interaction.md?api-version=v1
        body: _JSON = {
            "model": model,
            "input": prompt,
            "generation_config": {"max_output_tokens": max_tokens},
            # modelwire never continues an interaction, so the server need not store it.
            "store": False,
        }
        if system.strip():
            body["system_instruction"] = system
        if schema is not None:
            text_format = {"type": "text", "mime_type": "application/json", "schema": schema}
            body["response_format"] = text_format
        return _interaction_reply(await self._post("interactions", body, marker="status"))


class SystemOne(_Client):
    """Client for TypeSafe's System One API, or any server compatible with it."""

    _base_url = "https://api.typesafe.ai/v1"
    _key_header = "Authorization"
    _key_prefix = "Bearer "

    async def ask(
        self,
        model: str,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Return exactly one answer per question, each checked for its type's numeric fields.

        Raises ReplyError for a missing or malformed answer, APIError for an HTTP error, and
        httpx2.TransportError when the connection fails.
        """
        # The request and its response: https://docs.typesafe.ai/api/system-one.md
        # and the OpenAPI schema at https://api.typesafe.ai/openapi.json
        body = {"model": model, "state": state, "questions": questions}
        answers = (await self._post("systemone", body, marker="answers"))["answers"] or {}
        for name, question in questions.items():
            if not _complete(answer := answers.get(name), question):
                kind = question.get("type")
                raise ReplyError(f"no complete {kind} answer to {name!r}: {_quote(repr(answer))}")
        return {name: answers[name] for name in questions}


def _responses_reply(body: _JSON) -> Reply:
    if (status := body["status"]) != "completed":
        reason = (body.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            raise ReplyError(_CUT_OFF)
        raise ReplyError(f"the reply ended with status {_quote(repr(status))}: {_detail(reason)}")
    parts = [*(item.get("content") or []) for item in body.get("output") or []]
    if refusal := next((part for part in parts if part.get("type") == "refusal"), None):
        raise ReplyError(f"the model refused: {_detail(refusal.get('refusal'))}")
    text = "".join(part.get("text") or "" for part in parts if part.get("type") == "output_text")
    if not text:
        raise ReplyError(_NO_TEXT)
    usage = body.get("usage") or {}
    # input_tokens already counts the tokens read from and written to the cache:
    # https://developers.openai.com/api/docs/guides/prompt-caching
    details = usage.get("input_tokens_details") or {}
    return Reply(
        text=text,
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=details.get("cached_tokens") or 0,
        cache_write_tokens=details.get("cache_write_tokens") or 0,
    )


def _messages_reply(body: _JSON) -> Reply:
    # Every stop_reason but end_turn ends the reply short, since modelwire sends no tools or
    # stop sequences: https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
    # A refusal's category and explanation are in stop_details:
    # https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#refusal-response
    stop = body["stop_reason"]
    if stop == "refusal":
        details = body.get("stop_details") or {}
        reason = _detail(details.get("explanation"), details.get("category"))
        raise ReplyError(f"the model refused: {reason}")
    if stop == "max_tokens":
        raise ReplyError(_CUT_OFF)
    if stop != "end_turn":
        raise ReplyError(f"the reply ended with stop_reason {_quote(repr(stop))}")
    blocks = body.get("content") or []
    text = "".join(block.get("text") or "" for block in blocks if block.get("type") == "text")
    if not text:
        raise ReplyError(_NO_TEXT)
    usage = body.get("usage") or {}
    read = usage.get("cache_read_input_tokens") or 0
    written = usage.get("cache_creation_input_tokens") or 0
    return Reply(
        text=text,
        # input_tokens leaves out the tokens read from or written to the cache:
        # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        input_tokens=(usage.get("input_tokens") or 0) + read + written,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=read,
        cache_write_tokens=written,
    )


def _interaction_reply(body: _JSON) -> Reply:
    # max_output_tokens is the one documented cause of an incomplete interaction:
    # https://aistudio.google.com/docs/thinking.md#token-limits
    if (status := body["status"]) == "incomplete":
        raise ReplyError(_CUT_OFF)
    if status != "completed":
        raise ReplyError(f"the reply ended with status {_quote(repr(status))}")
    steps = [step for step in body.get("steps") or [] if step.get("type") == "model_output"]
    parts = [*(step.get("content") or []) for step in steps]
    text = "".join(part.get("text") or "" for part in parts if part.get("type") == "text")
    if not text:
        raise ReplyError(_NO_TEXT)
    usage = body.get("usage") or {}
    thought = usage.get("total_thought_tokens") or 0
    return Reply(
        text=text,
        input_tokens=usage.get("total_input_tokens") or 0,
        # Thought tokens are billed as output and counted apart from it:
        # https://aistudio.google.com/docs/thinking.md#pricing
        output_tokens=(usage.get("total_output_tokens") or 0) + thought,
        cache_read_tokens=usage.get("total_cached_tokens") or 0,
        # The API caches implicitly and reports no tokens written:
        # https://aistudio.google.com/docs/caching.md
        cache_write_tokens=0,
    )


def _complete(answer: object, question: dict[str, Any]) -> bool:
    kind = question.get("type")
    if not isinstance(answer, dict) or answer.get("type") != kind:
        return False
    criteria = question.get("criteria") or ()
    match kind:
        case "noul":
            return _number(answer.get("noul"))
        case "choice":
            options = list(criteria)
            picked = answer.get("choice") in options
        case "score":
            # Its probabilities are keyed by level index, as its legend is. The legend echoes
            # the caller's criteria, so it goes unchecked.
            options = [str(level) for level in range(len(criteria))]
            picked = _number(answer.get("score"))
        case _:
            # A box answer has coordinates, but a box question needs an image, and
            # modelwire sends none.
            return False
    probabilities = answer.get("probabilities")
    return (
        picked
        and _number(answer.get("confidence"))
        and isinstance(probabilities, dict)
        and all(_number(probabilities.get(option)) for option in options)
    )


def _number(value: object) -> bool:
    # bool is an int, and json.loads accepts NaN and Infinity; none of them is an answer's number.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _credentials(header: str, key: str | None, prefix: str) -> dict[str, str]:
    if key is None:
        return {}
    # httpx2 would quote a malformed key in its error, and so leak it.
    if not (key and key.isascii() and key.isprintable() and key == key.strip()):
        raise ValueError("key must be printable ASCII without surrounding whitespace")
    return {header: prefix + key}


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _retryable(response: httpx2.Response) -> bool:
    # OpenAI's SDK lets the server's x-should-retry overrule the status:
    # https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py
    match response.headers.get("x-should-retry"):
        case "true":
            return not response.is_success
        case "false":
            return False
    return response.status_code in _RETRY_STATUSES


def _retry_after(response: httpx2.Response) -> float | None:
    # Retry-After is either delay seconds or an HTTP-date:
    # https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after
    value = response.headers.get("retry-after", "")
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (email.utils.parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except TypeError, ValueError:  # Not a date, or one without a time zone.
            return None
    return seconds if seconds > 0 else None


def _reason(response: httpx2.Response, body: object) -> str:
    try:
        message = _innermost(body)
    except RecursionError:  # Only a hostile server nests this deep.
        message = None
    if message:
        return _quote(message)
    if response.has_redirect_location:
        return f"redirected to {_quote(response.headers['location'])}; set base_url to it"
    if body is None and (response.is_success or response.status_code in {404, 405}):
        kind = _quote(response.headers.get("content-type", "no content type").split(";")[0])
        return f"got {kind}, not JSON; check that base_url ends in its version, such as /v1"
    return "the response has no reply" if response.is_success else _quote(response.reason_phrase)


def _innermost(value: object) -> str | None:
    # A gateway nests the provider's error, often as JSON text, so the search descends into
    # it. The keys run from least to most specific, and the last message found wins.
    #   error, message  Anthropic's and OpenAI's bodies, as in
    #                   https://platform.claude.com/docs/en/api/errors
    #                   LM Studio sends error as plain text, undocumented but reported in
    #                   https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/618
    #   detail, msg     https://fastapi.tiangolo.com/tutorial/handling-errors/, which
    #                   TypeSafe follows, adding an object detail on HTTP 400.
    #   metadata, raw   https://openrouter.ai/docs/api/reference/errors-and-debugging
    match value:
        case str():
            inner = _parse(value)
            return (isinstance(inner, dict | list) and _innermost(inner)) or value
        case list():
            return "; ".join(filter(None, map(_innermost, value))) or None
        case dict():
            found = None
            for key in ("error", "detail", "msg", "message", "metadata", "raw"):
                found = _innermost(value.get(key)) or found
            if found and "loc" in value:  # FastAPI names the field at fault.
                found = f"{found} at {value['loc']}"
            return found
    return None


def _quote(text: str) -> str:
    # Server text lands in exception messages and logs, so it becomes one printable,
    # bounded line: no newline can forge a log record, and no escape can drive a terminal.
    line = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    return line if len(line) <= 300 else f"{line[:299]}…"


def _parse(text: str | bytes) -> object:
    try:
        return json.loads(text)
    except ValueError, RecursionError:
        return None


def _detail(*values: object) -> str:
    texts = (value for value in values if isinstance(value, str) and value.strip())
    return next(map(_quote, texts), "no details")
