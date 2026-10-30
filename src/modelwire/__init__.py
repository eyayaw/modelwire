"""Async client for OpenAI-compatible APIs."""

import asyncio
import email.utils
import ipaddress
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

import httpx2

__all__ = ["APIError", "OpenAI", "Reply", "ReplyError"]

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
# The statuses OpenAI's SDK retries: https://github.com/openai/openai-python#retries
_RETRY_STATUSES = frozenset({408, 409, 429, *range(500, 600)})
_CUT_OFF = "the reply reached max_tokens; raise it"
_NO_TEXT = "the reply holds no text"
# OpenAI's SDK waits 5 s for a connection and 600 s for the rest, since a reasoning model can
# think for minutes: https://github.com/openai/openai-python/blob/main/src/openai/_constants.py
_CONNECT_TIMEOUT = 5.0
# The header the API puts its request ID in:
# https://developers.openai.com/api/reference/overview#debugging-requests
_REQUEST_ID_HEADERS = ("x-request-id",)


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
    The server answered with an HTTP error, or with a body that holds no reply.

    retry_after is the wait the server asked for, in seconds, and request_id the provider's
    ID for the request. Either is None when the server omits it.
    """

    def __init__(self, response: httpx2.Response) -> None:
        self.status_code = response.status_code
        self.retry_after = _retry_after(response)
        self.request_id = next(filter(None, map(response.headers.get, _REQUEST_ID_HEADERS)), None)
        super().__init__(f"HTTP {response.status_code} from {response.url}: {_reason(response)}")


class ReplyError(Exception):
    """The server replied, but the reply is unusable: refused, truncated, empty, or incomplete."""


class _Client:
    def __init__(
        self,
        base_url: str,
        *,
        credentials: dict[str, str],
        headers: frozendict[str, str] = frozendict(),
        timeout: float,
        retries: int,
        transport: httpx2.AsyncBaseTransport | None,
    ) -> None:
        try:
            url = httpx2.URL(base_url)
        except httpx2.InvalidURL:
            url = httpx2.URL()
        if url.scheme not in {"https", "http"} or not url.host:
            raise ValueError(f"base_url must be an absolute https or http URL, not {base_url!r}")
        # httpx2 would send a username and password as Basic auth, past the clear-text check.
        if url.userinfo:
            raise ValueError("base_url must not hold a username or password; pass the key")
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
            headers=headers | credentials,
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
        """POST body to path and return the JSON reply, which must hold marker."""
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
        if not isinstance(data, dict) or marker not in data:
            raise APIError(response)
        return data


class OpenAI(_Client):
    """Client for OpenAI's Responses API, or any server compatible with it."""

    def __init__(
        self,
        key: str | None = None,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 600.0,
        retries: int = 2,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url,
            credentials=_credentials("Authorization", key, "Bearer "),
            timeout=timeout,
            retries=retries,
            transport=transport,
        )

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


def _responses_reply(body: _JSON) -> Reply:
    if (status := body["status"]) != "completed":
        reason = (body.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            raise ReplyError(_CUT_OFF)
        error = body.get("error") or {}
        detail = _detail(reason, error.get("message"), error.get("code"))
        raise ReplyError(f"the reply ended with status {_quote(repr(status))}: {detail}")
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


def _credentials(header: str, key: str | None, prefix: str = "") -> dict[str, str]:
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
    # Retry-After holds either delay seconds or an HTTP-date:
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


def _reason(response: httpx2.Response) -> str:
    body = _parse(response.content)
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
    return "the response holds no reply" if response.is_success else _quote(response.reason_phrase)


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
