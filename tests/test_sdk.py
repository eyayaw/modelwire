import asyncio
import gzip
import json
import os
import subprocess
import sys

import anthropic
import httpx2
import openai
import pytest
import typesafe_sdk
from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions
from test_modelwire import ANSWERS, REPLIES, SCHEMA, Server, answer, reply, sleeps

from modelwire import Anthropic, APIError, Gemini, OpenAI, ReplyError, SystemOne
from modelwire.sdk import transport

__all__ = ["sleeps"]  # A fixture, which pytest finds by its name in this module.

ANSWERED = {
    "model": "jev-latest",
    "usage": {"input_tokens": 120, "output_tokens": 7},
    "answers": ANSWERS,
}
CLASSES = [OpenAI, Anthropic, Gemini, SystemOne]
# Each provider's own SDK client, sending through the test's httpx2 client. Google's SDK takes
# its retry count only when built, so this one is built with a single attempt.
SDK_CLIENTS = {
    OpenAI: lambda http: openai.AsyncOpenAI(api_key="sk-test", http_client=http),
    Anthropic: lambda http: anthropic.AsyncAnthropic(auth_token="sk-test", http_client=http),
    Gemini: lambda http: genai.Client(
        api_key="sk-test",
        http_options=HttpOptions(
            httpx_async_client=http, retry_options=HttpRetryOptions(attempts=1)
        ),
    ),
    SystemOne: lambda http: typesafe_sdk.AsyncTypeSafeClient(api_key="sk-test", http_client=http),
}
URLS = {
    OpenAI: "https://api.openai.com/v1/responses",
    Anthropic: "https://api.anthropic.com/v1/messages",
    Gemini: "https://generativelanguage.googleapis.com/v1/interactions",
    SystemOne: "https://api.typesafe.ai/v1/systemone",
}


def ok(cls):
    return httpx2.Response(200, json=ANSWERED if cls is SystemOne else REPLIES[cls])


def ask(client):
    return answer(client) if isinstance(client, SystemOne) else reply(client, schema=SCHEMA)


def run(server, cls, call=ask, **settings):
    async def go():
        async with httpx2.AsyncClient(transport=server.transport) as http:
            sdk_client = SDK_CLIENTS[cls](http)
            async with cls(transport=transport(sdk_client), **settings) as client:
                return await call(client)

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def no_provider_settings(monkeypatch):
    # An SDK reads its key and base URL from the environment when it is given none.
    for name in os.environ:
        if name.startswith(("OPENAI_", "ANTHROPIC_", "GOOGLE_", "GEMINI_", "TYPESAFE_")):
            monkeypatch.delenv(name)


@pytest.mark.parametrize("cls", CLASSES)
def test_the_sdk_sends_the_same_body_to_its_own_url_and_the_reply_is_read(cls):
    direct, through = Server(ok(cls)), Server(ok(cls))

    async def go():
        async with cls("sk-test", transport=direct.transport) as client:
            return await ask(client)

    assert run(through, cls) == asyncio.run(go())
    assert str(through.requests[0].url) == URLS[cls]
    assert json.loads(through.requests[0].content) == json.loads(direct.requests[0].content)


def test_the_request_goes_to_the_sdk_clients_base_url():
    server = Server(ok(OpenAI))

    async def go():
        async with httpx2.AsyncClient(transport=server.transport) as http:
            sdk_client = openai.AsyncOpenAI(
                api_key="sk-test", base_url="https://gateway.example/v1", http_client=http
            )
            await reply(OpenAI(transport=transport(sdk_client)))

    asyncio.run(go())
    assert str(server.requests[0].url) == "https://gateway.example/v1/responses"


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    ("settings", "timeout"),
    [
        ({}, {"connect": 5.0, "read": 600.0, "write": 600.0, "pool": 600.0}),
        ({"timeout": 2.0}, {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0}),
    ],
)
def test_the_sdk_request_takes_modelwires_timeout(cls, settings, timeout):
    if cls is Gemini:  # Its SDK keeps one limit, the longest, for every phase.
        timeout = dict.fromkeys(timeout, max(timeout.values()))
    server = Server(ok(cls))
    run(server, cls, **settings)
    assert server.requests[0].extensions["timeout"] == timeout


@pytest.mark.parametrize("cls", CLASSES)
def test_modelwire_retries_and_the_sdk_does_not(sleeps, cls):
    server = Server(
        httpx2.ConnectError("refused"),
        httpx2.Response(429, headers={"retry-after": "7"}),
        ok(cls),
    )
    run(server, cls)
    assert server.responses == []
    assert sleeps == [0.5, 7.0]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "failure",
    [httpx2.ReadTimeout("slow"), httpx2.ReadError("reset"), httpx2.RemoteProtocolError("gone")],
)
def test_a_failure_after_the_request_was_sent_raises_the_httpx2_error(sleeps, cls, failure):
    server = Server(failure, ok(cls))
    with pytest.raises(type(failure)):
        run(server, cls)
    assert len(server.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("cls", CLASSES)
def test_an_undecodable_body_raises_the_httpx2_error(cls):
    # A stream, since httpx2 would decode content when the response is built.
    body = httpx2.ByteStream(b"not gzip")
    response = httpx2.Response(200, stream=body, headers={"content-encoding": "gzip"})
    with pytest.raises(httpx2.DecodingError):
        run(Server(response), cls)


@pytest.mark.parametrize("cls", CLASSES)
def test_an_http_error_raises_api_error_with_what_the_server_sent(sleeps, cls):
    response = httpx2.Response(
        429,
        json={"error": {"code": "rate_limited", "message": "Slow down."}},
        headers={"retry-after": "90", "x-request-id": "req_1"},
    )
    with pytest.raises(APIError, match=r"^HTTP 429 from .*: Slow down\.$") as error:
        run(Server(response), cls)
    caught = error.value
    assert (caught.status_code, caught.code, caught.request_id) == (429, "rate_limited", "req_1")
    assert caught.retry_after == 90


def test_a_failed_generation_raises_api_error_with_its_code():
    body = {"status": "failed", "error": {"code": "server_error", "message": "Upstream died."}}
    with pytest.raises(APIError, match=r"^HTTP 200 .*: Upstream died\.$") as error:
        run(Server(httpx2.Response(200, json=body)), OpenAI)
    assert error.value.code == "server_error"


@pytest.mark.parametrize(
    ("cls", "body", "message"),
    [
        (
            OpenAI,
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "^the reply reached max_tokens; raise it$",
        ),
        (
            Anthropic,
            {"stop_reason": "refusal", "stop_details": {"explanation": "Unsafe."}},
            r"^the model refused: Unsafe\.$",
        ),
        (Gemini, {"status": "incomplete"}, "^the reply reached max_tokens; raise it$"),
        # TypeSafe's SDK refuses this body, and the transport rebuilds the response from its error.
        (SystemOne, {"answers": None}, r"^no complete choice answer to 'kind': None$"),
    ],
)
def test_an_unusable_reply_raises_reply_error(cls, body, message):
    with pytest.raises(ReplyError, match=message):
        run(Server(httpx2.Response(200, json=body)), cls)


def test_a_max_tokens_the_sdk_requires_streaming_for_is_sent():
    server = Server(ok(Anthropic))
    run(server, Anthropic, lambda client: reply(client, max_tokens=64000))
    assert json.loads(server.requests[0].content)["max_tokens"] == 64000


def test_a_system_one_json_string_error_keeps_its_message():
    with pytest.raises(APIError, match=r": No such model\.$"):
        run(Server(httpx2.Response(404, json="No such model.")), SystemOne)


def test_a_system_one_text_error_is_not_read_as_json():
    response = httpx2.Response(404, text="<html></html>", headers={"content-type": "text/html"})
    with pytest.raises(APIError, match=r": got text/html, not JSON; "):
        run(Server(response), SystemOne)


def test_a_system_one_error_with_a_latin_1_header_is_read():
    response = httpx2.Response(
        404, json={"detail": "Unknown model."}, headers=[(b"x-note", b"\xe9")]
    )
    with pytest.raises(APIError, match=r": Unknown model\.$"):
        run(Server(response), SystemOne)


def test_a_compressed_system_one_error_body_is_read():
    body = gzip.compress(json.dumps({"detail": "Unknown model."}).encode())
    response = httpx2.Response(404, content=body, headers={"content-encoding": "gzip"})
    with pytest.raises(APIError, match=r": Unknown model\.$"):
        run(Server(response), SystemOne)


@pytest.mark.parametrize("cls", [OpenAI, Anthropic])
def test_a_redirect_is_refused_through_an_sdk_client_that_follows_them(cls):
    redirect = httpx2.Response(307, headers={"location": "https://elsewhere.example/v1"})
    server = Server(redirect, ok(cls))

    async def go():
        async with httpx2.AsyncClient(transport=server.transport, follow_redirects=True) as http:
            await reply(cls(transport=transport(SDK_CLIENTS[cls](http))))

    with pytest.raises(APIError, match=r": redirected to https://elsewhere\.example/v1; "):
        asyncio.run(go())
    assert len(server.requests) == 1


def test_an_openai_admin_key_is_not_sent_with_a_prompt():
    server = Server()

    async def go():
        async with httpx2.AsyncClient(transport=server.transport) as http:
            sdk_client = openai.AsyncOpenAI(admin_api_key="sk-admin", http_client=http)
            await reply(OpenAI(transport=transport(sdk_client)))

    with pytest.raises(TypeError, match="authentication"):
        asyncio.run(go())
    assert server.requests == []


@pytest.mark.parametrize(
    "settings", [{}, {"base_url": "https://gateway.example/v1/responses?tenant=a"}]
)
def test_another_providers_class_raises_before_any_request(settings):
    server = Server()

    async def go():
        async with httpx2.AsyncClient(transport=server.transport) as http:
            sdk_client = openai.AsyncOpenAI(api_key="sk-test", http_client=http)
            await reply(Anthropic(transport=transport(sdk_client), **settings))

    with pytest.raises(TypeError, match=r"^openai\.AsyncOpenAI cannot send a request for "):
        asyncio.run(go())
    assert server.requests == []


class AsyncOpenAI:
    """Another package's class with the SDK client's name."""

    def __init__(self, api_key):
        pass


@pytest.mark.parametrize(
    "other",
    [openai.OpenAI, AsyncOpenAI, typesafe_sdk.TypeSafeClient],
    ids=["sync", "a namesake", "sync TypeSafe"],
)
def test_a_client_of_another_class_raises_type_error(other):
    with pytest.raises(TypeError, match=r"^client must be an instance of one of "):
        transport(other(api_key="sk-test"))


def test_a_subclass_of_the_sdk_client_is_accepted():
    class Gateway(openai.AsyncOpenAI):
        pass

    assert isinstance(transport(Gateway(api_key="sk-test")), httpx2.AsyncBaseTransport)


@pytest.mark.parametrize(("attempts", "follows"), [(None, False), (1, True)])
def test_a_gemini_client_that_retries_or_follows_redirects_raises_value_error(attempts, follows):
    options = HttpOptions(
        retry_options=HttpRetryOptions(attempts=attempts),
        async_client_args={"follow_redirects": follows},
    )
    with pytest.raises(ValueError, match=r"^client must neither retry nor follow redirects"):
        transport(genai.Client(api_key="sk-test", http_options=options))


def test_a_gemini_client_set_to_follow_redirects_later_raises_before_any_request():
    server = Server()

    async def go():
        async with httpx2.AsyncClient(transport=server.transport) as http:
            sending = transport(SDK_CLIENTS[Gemini](http))
            http.follow_redirects = True
            await reply(Gemini(transport=sending))

    with pytest.raises(ValueError, match=r"^client must neither retry nor follow redirects"):
        asyncio.run(go())
    assert server.requests == []


def test_an_sdk_error_from_another_endpoint_is_raised_unchanged():
    class TokenError(Exception):
        request = httpx2.Request("POST", "https://auth.example/token")
        response = httpx2.Response(401, request=request)

    class Exchanging(openai.AsyncOpenAI):
        async def post(self, *args, **kwargs):
            raise TokenError

    with pytest.raises(TokenError):
        asyncio.run(reply(OpenAI(transport=transport(Exchanging(api_key="sk-test")))))


def test_an_sdk_client_on_another_http_library_raises_type_error():
    class Other(openai.AsyncOpenAI):
        async def post(self, *args, **kwargs):
            return object()

    with pytest.raises(TypeError, match=r"^the SDK client must send with an httpx2 client"):
        asyncio.run(reply(OpenAI(transport=transport(Other(api_key="sk-test")))))


def test_closing_leaves_the_sdk_client_open():
    async def go():
        sdk_client = openai.AsyncOpenAI(api_key="sk-test")
        async with OpenAI(transport=transport(sdk_client)):
            pass
        try:
            return sdk_client.is_closed()
        finally:
            await sdk_client.close()

    assert asyncio.run(go()) is False


@pytest.mark.parametrize(
    ("module", "unloaded"),
    [
        ("modelwire", ["modelwire.sdk"]),
        ("modelwire.sdk", ["openai", "anthropic", "google.genai", "typesafe_sdk"]),
    ],
)
def test_an_import_loads_neither_the_sdk_module_nor_an_sdk(module, unloaded):
    code = f"import sys, {module}; print(sorted(set(sys.argv[1:]) & set(sys.modules)))"
    done = subprocess.run(
        [sys.executable, "-c", code, *unloaded], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"
