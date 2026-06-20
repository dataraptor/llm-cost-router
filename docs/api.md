# HTTP API

`api/` is a **thin FastAPI adapter** over the `frugalroute` engine: every endpoint
validates its input, calls the engine **in-process**, and serializes the result. No
routing, cost, or metric logic lives here — those numbers come from `core` only (a passing
grep test enforces it). See [`architecture.md`](architecture.md#in-process-wiring) for how
this fits the whole stack.

All routes are under the configurable `FRUGALROUTE_API_PREFIX` (default `/api`). `GET /`
redirects to `/docs` (the auto-generated OpenAPI UI).

---

## Endpoints

| Method | Path | Key? | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | no | `status`, `version`, `has_api_key` (drives the UI's "View Proof"). |
| `GET` | `/api/config` | no | Prompt version, tiers, strategies, **pinned pricing**, defaults — sourced from `core`, no duplicated numbers. |
| `GET` | `/api/metrics` | no | Process-lifetime counters (below). |
| `GET` | `/api/examples` | no | Bundled demo queries (`id/benchmark/label/query` only — **no answers leaked**). |
| `POST` | `/api/route` | live | Route one query; returns the full `RouteResult` + a "saved vs always-Opus" delta. |
| `GET` | `/api/route/stream` | live | Same routing, **streamed as SSE** (one event per boundary). |
| `GET` | `/api/eval/sample` | no | The committed precomputed eval bundle (the Frontier proof, no key/network). |
| `POST` | `/api/eval` | live | A **bounded** quick eval (`quick=true` only; full sweeps are CLI-only). |

`/api/route` and `/api/route/stream` accept **exactly one of** `query` or `example_id`
(plus optional `strategy`, `benchmark`, `tau`, `theta`); the stream endpoint validates and
resolves *before* streaming, so bad input is a clean pre-stream `422/404`, never a `200`
empty stream. The streamed `done` body is **byte-identical** to `POST /api/route` for the
same inputs (both are derived from the one serialized `RouteResult`).

---

## SSE event order

```
event: phase     data: {"phase":"gen","tier":"claude-haiku-4-5"}
event: candidate data: {"answer":"…","tier":"…","cost_usd":…}
event: gate      data: {"sufficient":true,"confidence":0.91,"reason":"…","cost_usd":…}
event: cost      data: {"cost_usd_cumulative":…}
event: done      data: {…full RouteResponse…}
```

On escalation a `phase {escalate, strong}` precedes the final `cost`/`done`; a refusal
emits a `refusal` event (never a fabricated answer); an Anthropic-side 429 surfaces as a
`retry` event. Frame format is standard SSE: `event: <type>\ndata: <json>\n\n`.

---

## Error model

Every error is a typed, structured JSON body (not a bare stack trace), and the engine
validates its `FRUGALROUTE_*` config **at startup** so an invalid value fails loudly.

| Status | `type` | When |
|---|---|---|
| `400` | `bad-request` | malformed request (e.g. a non-`quick` eval over HTTP). |
| `404` | `not-found` | unknown `example_id`, or no committed sample bundle. |
| `422` | validation | scalar/shape validation failed (FastAPI + the "one of query/example_id" rule). |
| `429` | `rate-limited` | over the per-IP token bucket — plus `Retry-After`. |
| `503` | `missing-key` | live routing requested with no backend key configured. |
| `503` | `busy` | over the concurrency cap — load is shed, never queued unbounded; plus `Retry-After`. |
| `504` | `timeout` | request exceeded `FRUGALROUTE_REQUEST_TIMEOUT_S` (no hung connection). |

The `429 rate-limited` (this server's limiter) is distinct from the Anthropic-side 429 the
SDK retries and the stream surfaces as a `retry` event. Client error messages never leak a
server path or the API key.

---

## Observability

Every LLM call and every route logs **one JSON line** (model, tokens, cost, latency,
escalation, refusal — never the key or a full query body), and every HTTP request carries
an `X-Request-ID` (accepted or generated) into the access log.

`GET /api/metrics` exposes process-lifetime counters (reset on restart), summed from the
engine's own cost accounting — not recomputed here:

```
requests_total · cost_usd_total · escalation_rate · refused_total
latency_p50_s · latency_p95_s · since
```

---

## Configuration

All settings are env-driven with safe defaults (full table in the root
[`README.md`](../README.md#configuration)). The load-bearing ones:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Native backend key. Read only from env; **never logged or serialized**. |
| `FRUGALROUTE_BACKEND` | *(native)* | `azure` injects the gpt-5.5 adapter; empty uses the native Anthropic client. |
| `FRUGALROUTE_MAX_CONCURRENCY` | `6` | Max simultaneous backend calls process-wide; sizes the `503 busy` back-pressure. |
| `FRUGALROUTE_REQUEST_TIMEOUT_S` | `60` | Per-request bound → typed `504` on exceed. |
| `FRUGALROUTE_CORS_ORIGINS` | `*` | Comma-separated allow-origins. **Lock down in production.** |
| `FRUGALROUTE_RATE_LIMIT_ENABLED` | `false` | Enable the per-IP token-bucket rate limit. |

The key is **runtime-only** by design: it is never baked into a Docker layer (proven by
`docker history … | grep -i SENTINEL` returning `0`). Live routing/streaming turns on only
when a key is supplied at runtime; with no key, `/api/health` reports `has_api_key:false`
and a live query returns the honest *missing-key* `503` rather than a fake answer.
</content>
