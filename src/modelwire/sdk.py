"""An httpx2 transport that sends modelwire's requests through a provider's own SDK client."""

import json
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

import httpx2

if TYPE_CHECKING:
    # Only a caller who passes one of these clients has its SDK installed.
    import anthropic
    import openai
    import typesafe_sdk
    from google import genai

__all__ = ["transport"]

type _JSON = dict[str, Any]
type _Send = Callable[[Any, _JSON, httpx2.Timeout], Awaitable[httpx2.Response]]


def transport(client: object) -> httpx2.AsyncBaseTransport:
    """
    Return a transport that sends each request through client, a provider's own SDK client.

    Pass it as transport to the modelwire class of the same provider. The SDK client's key and
    base URL apply, and the modelwire class's do not. Raises TypeError unless client is an
    openai.AsyncOpenAI, anthropic.AsyncAnthropic, google.genai.Client, or
    typesafe_sdk.AsyncTypeSafeClient, and ValueError for a google.genai.Client that retries or
    follows redirects. The transport raises TypeError, before sending, for a request from
    another provider's class.
    """
    # A sync client, or an unknown one, would fail only after it sent the request. A client's
    # class is loaded once the client exists, so its SDK is looked up and never imported.
    for expected, path, send in _SDKS:
        package, _, name = expected.rpartition(".")
        cls = getattr(sys.modules.get(package), name, None)
        if isinstance(cls, type) and isinstance(client, cls):
            if send is _gemini:
                _check_gemini(cast("genai.Client", client))
            return _Transport(client, expected, path, send)
    known = ", ".join(expected for expected, _, _ in _SDKS)
    given = f"{type(client).__module__}.{type(client).__qualname__}"
    raise TypeError(f"client must be an instance of one of {known}, not {given}")


class _Transport(httpx2.AsyncBaseTransport):
    def __init__(self, client: object, expected: str, path: str, send: _Send) -> None:
        self._client = client
        self._expected = expected
        self._path = path
        self._send = send

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        # modelwire ends its URL with the provider's path, whatever base_url holds.
        if not str(request.url).endswith(self._path):
            raise TypeError(
                f"{self._expected} cannot send a request for {request.url.path}; "
                "pass its transport to the modelwire class of the same provider"
            )
        timeouts = httpx2.Timeout(**request.extensions["timeout"])
        try:
            response = await self._send(self._client, json.loads(request.content), timeouts)
        except Exception as error:
            # Each SDK wraps an httpx2 failure in its own connection error, and the OpenAI,
            # Anthropic, and Gemini SDKs hold the response on their status error. Measured on
            # 2026-11-21 with openai 3.51.0, anthropic 2.8.0, google-genai 3.3.0, and
            # typesafe-sdk 0.7.5.
            if isinstance(error.__cause__, httpx2.RequestError):
                raise error.__cause__ from None
            # An error from another endpoint, such as a token exchange, is the SDK's to report.
            response = getattr(error, "response", None)
            if isinstance(response, httpx2.Response) and self._answers(response):
                return response
            raise
        # An SDK client given another HTTP library's client answers with that library's response.
        if not isinstance(response, httpx2.Response):
            kind = f"{type(response).__module__}.{type(response).__qualname__}"
            raise TypeError(f"the SDK client must send with an httpx2 client; it returned {kind}")
        return response

    def _answers(self, response: httpx2.Response) -> bool:
        try:
            return response.request.url.path.endswith(self._path)
        except RuntimeError:  # Without its request, nothing says which endpoint answered.
            return False


async def _openai(
    client: openai.AsyncOpenAI, body: _JSON, timeouts: httpx2.Timeout
) -> httpx2.Response:
    # post() with cast_to returns the raw response, checked on 2026-11-21:
    # https://github.com/openai/openai-python#undocumented-endpoints
    # Its options turn the SDK's retries and redirects off for this request, and keep an admin
    # key out of it, as responses.create does. Measured on 2026-11-21 with openai 3.51.0.
    options: openai.RequestOptions = {
        "max_retries": 0,
        "timeout": timeouts,
        "follow_redirects": False,
        "security": {"bearer_auth": True},
    }
    return await client.post("/responses", body=body, cast_to=httpx2.Response, options=options)


async def _anthropic(
    client: anthropic.AsyncAnthropic, body: _JSON, timeouts: httpx2.Timeout
) -> httpx2.Response:
    # post() with cast_to returns the raw response, checked on 2026-11-21:
    # https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#undocumented-endpoints
    # Its options turn the SDK's retries and redirects off for this request, measured on
    # 2026-11-21 with anthropic 2.8.0.
    return await client.post(
        "/v1/messages",
        body=body,
        cast_to=httpx2.Response,
        options={"max_retries": 0, "timeout": timeouts, "follow_redirects": False},
    )


def _check_gemini(client: genai.Client) -> None:
    # The SDK takes no retry count or redirect setting for one request, so both must be off on
    # the client. genai.Client always sets retry_config to a RetryConfig whose max_retries is
    # attempts - 1, four by default; getattr satisfies its optional type. Measured on 2026-11-21
    # with google-genai 3.3.0.
    config = client.aio.interactions.sdk_configuration
    retries = getattr(config.retry_config, "max_retries", 1)
    if retries or getattr(config.async_client, "follow_redirects", True):
        raise ValueError(
            "client must neither retry nor follow redirects; build it with "
            "http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=1), "
            "async_client_args={'follow_redirects': False}), "
            "or pass an httpx_async_client that does not follow redirects"
        )


async def _gemini(client: genai.Client, body: _JSON, timeouts: httpx2.Timeout) -> httpx2.Response:
    _check_gemini(client)  # The caller can change its HTTP client after transport() checked it.
    interactions = client.aio.interactions
    return (await interactions.with_raw_response.create(**body, timeout=timeouts)).http_response


async def _typesafe(
    client: typesafe_sdk.AsyncTypeSafeClient, body: _JSON, timeouts: httpx2.Timeout
) -> httpx2.Response:
    import typesafe_sdk  # noqa: PLC0415  A caller who passes this client has it installed.

    # RetryPolicy(max_retries=0) turns the SDK's retries off, checked on 2026-11-21:
    # https://docs.typesafe.ai/sdk/python/api/clients/async.md
    # The SDK returns a reply only when the response matches its own schema. It raises any other
    # response as an error holding the status, headers, and parsed body, and this function
    # rebuilds the response from those, with the standard reason phrase for its status.
    # Measured on 2026-11-21 with typesafe-sdk 0.7.5.
    try:
        reply = await client.system_one(
            body["state"],
            body["questions"],
            model=body["model"],
            retry=typesafe_sdk.RetryPolicy(max_retries=0),
            timeout=timeouts,
        )
    except typesafe_sdk.TypeSafeAPIError as error:
        # A JSON string body and a plain-text body both arrive as str, so the content type
        # tells them apart.
        text = isinstance(error.body, str) and "json" not in error.headers.get("content-type", "")
        content = error.body if text else "" if error.body is None else json.dumps(error.body)
        # The SDK has already decompressed the body, so the rebuilt response drops
        # content-encoding.
        headers = httpx2.Headers(error.headers)
        headers.pop("content-encoding", None)
        return httpx2.Response(error.status, headers=headers, content=content)
    return reply.raw_http_response


# Each SDK client class, the path modelwire posts to for its provider, and the function that
# sends through it.
_SDKS: tuple[tuple[str, str, _Send], ...] = (
    ("openai.AsyncOpenAI", "/responses", _openai),
    ("anthropic.AsyncAnthropic", "/messages", _anthropic),
    ("google.genai.Client", "/interactions", _gemini),
    ("typesafe_sdk.AsyncTypeSafeClient", "/systemone", _typesafe),
)
