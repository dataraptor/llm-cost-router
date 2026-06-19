# core — the FrugalRoute engine

The standalone engine: a framework-free, installable Python package
(`frugalroute`). It knows nothing about HTTP, UI, or deployment — `eval`, `api`,
and `app` build on top of it. Importing it never requires an API key (the
Anthropic client is built lazily in `get_client()`).

**Depends on:** nothing else in this repo. This is the bottom of the stack.

## What's here (split 01)

- `models.py` — `GateVerdict` (the only API-sent type) + `RouteResult` /
  `FrontierPoint` / `EvalReport` contracts (build-spec §7).
- `prompts.py` — the gate + per-benchmark generation prompts behind `PROMPT_VERSION` (§6).
- `llm.py` — pinned `PRICING`, the config-driven `DEFAULT_TIERS`, the cache-aware
  `cost_usd(...)` engine, and a refusal-safe `call(...)` wrapper.

## Install

From the repo root:

```bash
pip install -e "core[dev]"
python -c "import frugalroute; print(frugalroute.PROMPT_VERSION)"
```

## Test / lint / type (no API key required)

```bash
ruff format --check core/src core/tests
ruff check core/src core/tests
mypy core/src/frugalroute            # run from repo root, or `cd core && mypy src/frugalroute`
pytest core/tests -m "not api" -q --cov=frugalroute --cov-report=term-missing
```

The no-key suite (`-m "not api"`) must pass with **no** `ANTHROPIC_API_KEY` set
and no network. Tests marked `@pytest.mark.api` are auto-skipped unless the key
is present (see `core/.env.example`).
