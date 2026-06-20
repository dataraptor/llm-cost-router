# FrugalRoute API

> Part of **FrugalRoute**. See the [root README](../README.md) for the headline,
> the proof, and the quickstart.

A **thin FastAPI adapter** over the [`frugalroute`](../core) engine. It validates
HTTP requests, calls the engine in-process (no service-to-service HTTP for the
engine), and serializes the engine's contracts to JSON. It holds **no**
routing/metrics/cost logic: delete `api/` and the engine is unchanged.

## Install & run

```bash
pip install -e "core[dev]"
pip install -e "api[dev]"
uvicorn frugalroute_api.app:app --port 8000
# open http://localhost:8000/docs
```

By default the live routing endpoints use the native Anthropic client and return a
typed `503 missing-key` error when `ANTHROPIC_API_KEY` is unset, pointing at the
precomputed proof. To drive the live demo with the bundled Azure OpenAI gpt-5.5
backend instead, set `FRUGALROUTE_BACKEND=azure` (plus the `AZURE_OPENAI_*` env).

## Endpoints (prefix `/api`)

| Method & path | Purpose |
|---|---|
| `GET /api/health` | liveness + whether a backend key is configured |
| `GET /api/config` | pricing/tiers/prompt-version (sourced from core), UI defaults |
| `GET /api/examples` | the demo example picker (id/benchmark/label/query only) |
| `POST /api/route` | route one query (cascade or predictive) -> `RouteResult` + UI extras |
| `GET /api/eval/sample` | the precomputed frontier **bundle** (no key/network) |
| `POST /api/eval` | a bounded live **quick** eval -> same bundle shape |
| `GET /api/route/stream` | stream one cascade route over SSE (token deltas, gate verdict, escalation events) |

Every non-2xx response is the structured shape
`{"error": {"type", "message", "detail"}}`, never an unstructured 500.

## Environment

| Var | Default | Meaning |
|---|---|---|
| `FRUGALROUTE_CORS_ORIGINS` | `*` | comma-separated allowed origins (lock down in prod) |
| `FRUGALROUTE_API_PREFIX` | `/api` | route prefix |
| `FRUGALROUTE_SAMPLE_RUN_PATH` | bundled fixture | path to the committed eval bundle |
| `FRUGALROUTE_BACKEND` | _(native Anthropic)_ | `azure` to use the gpt-5.5 adapter |
| `FRUGALROUTE_ROUTER_PATH` | _(none)_ | trained predictive router (joblib) for live predictive |

The bundled `data/sample_run.json` is the committed frozen sample run that powers the
Frontier proof with no key or network.
