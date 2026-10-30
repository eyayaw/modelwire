import asyncio
import email.utils
import json
import re
from datetime import UTC, datetime, timedelta

import httpx2
import pytest

import modelwire
from modelwire import Anthropic, APIError, OpenAI, Reply, ReplyError

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
TEXT = '{"ok": true}'
HEADERS = {"authorization", "x-api-key", "anthropic-version"}
# Each class's reply holding TEXT, from a prompt of 120 tokens: 100 read from the cache and 5
# written to it.
REPLIES = {
    OpenAI: {
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Hmm."}]},
            {"type": "message", "content": [{"type": "output_text", "text": TEXT}]},
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 5},
        },
    },
    Anthropic: {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": TEXT}],
        "usage": {
            "input_tokens": 15,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 5,
            "output_tokens": 7,
        },
    },
}
SYSTEM = {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}


class Server:
    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []
        self.transport = httpx2.MockTransport(self.handle)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def ok(cls):
    return httpx2.Response(200, json=REPLIES[cls])


def run(server, call, cls=OpenAI, key="sk-test", **settings):
    async def go():
        async with cls(key, transport=server.transport, **settings) as client:
            return await call(client)

    return asyncio.run(go())


def reply(client, **options):
    return client.ask("a-model", "Is it ok?", **{"system": "Be brief.", **options})


@pytest.fixture
def sleeps(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(modelwire.asyncio, "sleep", sleep)
    monkeypatch.setattr(modelwire.random, "uniform", lambda low, high: high)
    return delays


@pytest.mark.parametrize(
    ("cls", "key", "schema", "url", "headers", "body"),
    [
        (
            OpenAI,
            "sk-test",
            SCHEMA,
            "https://api.openai.com/v1/responses",
            {"authorization": "Bearer sk-test"},
            {
                "model": "a-model",
                "instructions": "Be brief.",
                "input": "Is it ok?",
                "max_output_tokens": 4096,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "reply",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
            },
        ),
        (
            OpenAI,
            None,
            None,
            "https://api.openai.com/v1/responses",
            {},
            {
                "model": "a-model",
                "instructions": "Be brief.",
                "input": "Is it ok?",
                "max_output_tokens": 4096,
                "store": False,
            },
        ),
        (
            Anthropic,
            "sk-test",
            SCHEMA,
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": "sk-test", "anthropic-version": "2026-10-21"},
            {
                "model": "a-model",
                "system": [SYSTEM],
                "messages": [{"role": "user", "content": "Is it ok?"}],
                "max_tokens": 4096,
                "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            },
        ),
        (
            Anthropic,
            None,
            None,
            "https://api.anthropic.com/v1/messages",
            {"anthropic-version": "2026-10-21"},
            {
                "model": "a-model",
                "system": [SYSTEM],
                "messages": [{"role": "user", "content": "Is it ok?"}],
                "max_tokens": 4096,
            },
        ),
    ],
)
def test_request_follows_the_wire_format(cls, key, schema, url, headers, body):
    server = Server(ok(cls))
    run(server, lambda client: reply(client, schema=schema), cls, key)
    [request] = server.requests
    assert str(request.url) == url
    assert {name: value for name, value in request.headers.items() if name in HEADERS} == headers
    assert json.loads(request.content) == body


@pytest.mark.parametrize(
    ("settings", "timeout"),
    [
        ({}, {"connect": 5.0, "read": 600.0, "write": 600.0, "pool": 600.0}),
        ({"timeout": 2.0}, {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0}),
    ],
)
def test_a_connection_waits_at_most_5_s_and_the_reply_takes_timeout(settings, timeout):
    server = Server(ok(OpenAI))
    run(server, reply, **settings)
    assert server.requests[0].extensions["timeout"] == timeout


@pytest.mark.parametrize("cls", REPLIES)
def test_reply_holds_the_text_and_the_whole_prompt_count(cls):
    assert run(Server(ok(cls)), reply, cls) == Reply(TEXT, 120, 7, 100, 5)


def test_transient_failures_are_retried_after_backoff_or_retry_after(sleeps):
    server = Server(
        httpx2.ConnectError("refused"),
        httpx2.Response(429, headers={"retry-after": "7"}),
        ok(OpenAI),
    )
    assert run(server, reply).text == TEXT
    assert sleeps == [0.5, 7.0]


@pytest.mark.parametrize("status", [408, 409, 503, 599])
def test_the_last_failure_raises_once_retries_run_out(sleeps, status):
    server = Server(httpx2.Response(status), httpx2.Response(status))
    with pytest.raises(APIError, match=rf"^HTTP {status} "):
        run(server, reply, retries=1)
    assert sleeps == [0.5]


def test_backoff_doubles_up_to_8_s_less_up_to_a_quarter_at_random(sleeps, monkeypatch):
    monkeypatch.setattr(modelwire.random, "uniform", lambda low, high: low)
    with pytest.raises(APIError):
        run(Server(*[httpx2.Response(503)] * 7), reply, retries=6)
    assert sleeps == [0.375, 0.75, 1.5, 3.0, 6.0, 6.0]


@pytest.mark.parametrize(("status", "hint", "requests"), [(503, "false", 1), (400, "true", 2)])
def test_the_server_overrules_the_status_with_x_should_retry(sleeps, status, hint, requests):
    server = Server(*[httpx2.Response(status, headers={"x-should-retry": hint})] * 2)
    with pytest.raises(APIError):
        run(server, reply, retries=1)
    assert len(server.requests) == requests


IN_2_MINUTES = email.utils.format_datetime(datetime.now(UTC) + timedelta(minutes=2), usegmt=True)


@pytest.mark.parametrize("delay", ["120", IN_2_MINUTES])
def test_a_retry_after_beyond_a_minute_raises_at_once_and_says_how_long(sleeps, delay):
    server = Server(httpx2.Response(429, headers={"retry-after": delay}), ok(OpenAI))
    with pytest.raises(APIError, match=r"^HTTP 429 ") as error:
        run(server, reply)
    assert 110 < (error.value.retry_after or 0) <= 120
    assert sleeps == []


@pytest.mark.parametrize("header", ["x-request-id", "request-id"])
def test_an_api_error_holds_the_providers_request_id(header):
    with pytest.raises(APIError) as error:
        run(Server(httpx2.Response(400, headers={header: "req_1"})), reply)
    assert error.value.request_id == "req_1"


@pytest.mark.parametrize(
    "failure",
    [httpx2.ReadTimeout("slow"), httpx2.ReadError("reset"), httpx2.RemoteProtocolError("gone")],
)
def test_a_failure_after_the_request_was_sent_is_not_retried(sleeps, failure):
    with pytest.raises(type(failure)):
        run(Server(failure, ok(OpenAI)), reply)
    assert sleeps == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # OpenRouter nests the provider's error as JSON text under metadata.raw.
        (
            {
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"raw": json.dumps({"error": {"message": "Bad schema."}})},
                }
            },
            "Bad schema.",
        ),
        (
            {"error": {"message": "Provider returned error", "metadata": {"raw": "Timed out."}}},
            "Timed out.",
        ),
        ({"error": "Model not loaded."}, "Model not loaded."),
        ({"error": {"message": '{"minimum": 0}'}}, '{"minimum": 0}'),
        ({"message": "model must be a string", "error": "Bad Request"}, "model must be a string"),
        ({"detail": "Unknown model."}, "Unknown model."),
        (
            {"detail": [{"loc": ["body", "model"], "msg": "Field required"}]},
            "Field required at ['body', 'model']",
        ),
    ],
)
def test_a_client_error_is_not_retried_and_quotes_the_innermost_message(sleeps, body, message):
    with pytest.raises(APIError, match=f"^HTTP 422 .*: {re.escape(message)}$") as error:
        run(Server(httpx2.Response(422, json=body)), reply)
    assert error.value.status_code == 422
    assert sleeps == []


def test_an_error_nested_past_the_recursion_limit_raises_api_error():
    deep = '{"error":' * 5000 + '"x"' + "}" * 5000
    with pytest.raises(APIError, match=r"^HTTP 422 "):
        run(Server(httpx2.Response(422, content=deep)), reply)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx2.Response(404, text="<html></html>", headers={"content-type": "text/html"}),
            "got text/html, not JSON; check that base_url ends in its version, such as /v1",
        ),
        (httpx2.Response(401), "Unauthorized"),
        (httpx2.Response(200, json={}), "the response holds no reply"),
        # A followed redirect would carry the key to another host.
        (
            httpx2.Response(307, headers={"location": "https://elsewhere.example/v1"}),
            "redirected to https://elsewhere.example/v1; set base_url to it",
        ),
    ],
)
def test_an_error_without_a_server_message_says_what_to_do(response, message):
    server = Server(response, ok(OpenAI))
    with pytest.raises(APIError, match=f": {re.escape(message)}$"):
        run(server, reply)
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("cls", "body", "message"),
    [
        (
            OpenAI,
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}],
            },
            r"^the model refused: No\.$",
        ),
        (
            OpenAI,
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "^the reply reached max_tokens; raise it$",
        ),
        (
            OpenAI,
            {"status": "failed", "error": {"code": "server_error"}},
            "^the reply ended with status 'failed': server_error$",
        ),
        (OpenAI, {"status": "completed", "output": []}, "^the reply holds no text$"),
        (
            OpenAI,
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": ""}]}],
            },
            "^the reply holds no text$",
        ),
        (
            Anthropic,
            {"stop_reason": "refusal", "stop_details": {"explanation": "Unsafe."}},
            r"^the model refused: Unsafe\.$",
        ),
        (
            Anthropic,
            {"stop_reason": "max_tokens", "content": []},
            "^the reply reached max_tokens; raise it$",
        ),
        (Anthropic, {"stop_reason": "end_turn", "content": []}, "^the reply holds no text$"),
        (
            Anthropic,
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": ""}]},
            "^the reply holds no text$",
        ),
        (
            Anthropic,
            {"stop_reason": "pause_turn", "content": [{"type": "text", "text": "x"}]},
            "^the reply ended with stop_reason 'pause_turn'$",
        ),
    ],
)
def test_an_unusable_reply_raises_reply_error(cls, body, message):
    with pytest.raises(ReplyError, match=message):
        run(Server(httpx2.Response(200, json=body)), reply, cls)


@pytest.mark.parametrize(
    ("cls", "settings", "call", "argument"),
    [
        (OpenAI, {"base_url": "api.openai.com/v1"}, reply, "base_url"),
        (OpenAI, {"base_url": "https://user:pw@api.openai.com/v1"}, reply, "base_url"),
        (OpenAI, {"base_url": "http://10.0.0.5:8000/v1"}, reply, "refusing"),
        (OpenAI, {"key": "sk-test\n"}, reply, "key"),
        (OpenAI, {"key": " sk-test"}, reply, "key"),
        (Anthropic, {"key": ""}, reply, "key"),
        (OpenAI, {"timeout": 0}, reply, "timeout"),
        (OpenAI, {"timeout": float("nan")}, reply, "timeout"),
        (OpenAI, {"retries": -1}, reply, "retries"),
    ],
)
def test_a_bad_argument_raises_before_any_request(cls, settings, call, argument):
    server = Server()
    with pytest.raises(ValueError, match=f"^{argument} "):
        run(server, call, cls, **settings)
    assert server.requests == []


@pytest.mark.parametrize(
    ("base_url", "key"),
    [
        ("http://localhost:8000/v1", "sk-test"),
        ("http://[::1]:8000/v1", "sk-test"),
        ("http://10.0.0.5/v1", None),
    ],
)
def test_plain_http_is_allowed_to_a_loopback_host_or_without_a_key(base_url, key):
    assert run(Server(ok(OpenAI)), reply, key=key, base_url=base_url).text == TEXT


FORGED = "Bad request.\nINFO forged log line \x1b[2J" + "x" * 1000


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(400, json={"error": {"message": FORGED}}),
        httpx2.Response(400, extensions={"reason_phrase": FORGED.encode()}),
    ],
)
def test_server_text_is_quoted_as_one_bounded_printable_line(response):
    with pytest.raises(APIError) as error:
        run(Server(response), reply)
    message = str(error.value)
    assert message.isprintable()
    assert "Bad request. INFO forged log line [2Jxxx" in message
    assert message.endswith("…")
    assert len(message.split(": ", 1)[1]) == 300


def test_repr_shows_the_base_url_and_never_the_key():
    assert repr(OpenAI("sk-secret")) == "OpenAI(base_url='https://api.openai.com/v1')"


@pytest.mark.parametrize(("cls", "field"), [(OpenAI, "instructions"), (Anthropic, "system")])
def test_a_blank_system_prompt_is_not_sent(cls, field):
    server = Server(ok(cls))
    run(server, lambda client: reply(client, system=" \n"), cls)
    assert field not in json.loads(server.requests[0].content)
