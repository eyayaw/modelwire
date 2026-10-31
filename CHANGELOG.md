# Changelog

## [0.1.0](https://github.com/eyayaw/modelwire/releases/tag/v0.1.0) - 2026-10-31

### Added

- `OpenAI`, `Anthropic`, and `SystemOne` clients for the OpenAI Responses, Anthropic Messages, and System One APIs, each with one `ask()` method.
- Reply text and token counts, cached tokens included, in `Reply`.
- System One answers checked against each question's type.
- `APIError` with the status, `retry_after`, and `request_id`, and `ReplyError` for a refused, truncated, or incomplete reply.
- Retries of failed connections and of HTTP 408, 409, 429, and 5xx, set by `retries`.
