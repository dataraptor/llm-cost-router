# Contributing to FrugalRoute

Thanks for helping! This project gates every change on a small, fast, **no-key**
quality bar. CI runs the *exact* commands below — if they pass locally, CI passes.

## The two-tier test model

- **No-key gates (every push / PR).** Lint, types, the no-key pytest suites with a
  coverage floor, the frontend unit + e2e tests, and a package/boot smoke. These
  need **no secrets**, so PRs from forks run the full gate. This is
  [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
- **Live `@api` suite (manual / nightly only).** The tests marked `@pytest.mark.api`
  make real Anthropic API calls. They run only when an `ANTHROPIC_API_KEY` **secret**
  is configured, skip cleanly otherwise, and **never block** a normal PR. This is
  [`.github/workflows/api-tests.yml`](.github/workflows/api-tests.yml) (cost-bounded,
  est. < $0.05/run).

Markers: `@pytest.mark.api` (live Anthropic) and `@pytest.mark.azure` (live Azure
gpt-5.5 adapter). The no-key gate excludes **both** (`-m "not api and not azure"`),
so a local `.env` with Azure creds doesn't accidentally make the gate spend money.

## Run the gates locally

```bash
# Python — lint + types (mypy is run from inside each package for its strict config)
ruff format --check core/src core/tests api/src api/tests eval
ruff check        core/src core/tests api/src api/tests eval
( cd core && mypy src/frugalroute )
( cd api  && mypy src/frugalroute_api )

# Python — no-key tests + coverage floor (core & api are separate rootdirs that
# share test basenames, so they run as separate invocations — not one command).
pytest core/tests        -m "not api and not azure" -q --cov=frugalroute
pytest api/tests         -m "not api and not azure" -q --cov=frugalroute_api
pytest tests/integration -m "not api and not azure" -q

# Frontend — unit (the gate) + e2e (Playwright vs a mocked API)
cd app && npm ci && npm test
npx playwright install chromium && npm run test:e2e
```

Or, on a POSIX shell with `make`: `make test` (no-key core + api + integration + app
unit). On Windows without `make`, use `scripts/dev.ps1` for the stack and the
explicit commands above for the gates.

## Coverage floor

The threshold is the **single source of truth** in each package's
`pyproject.toml` under `[tool.coverage.report] fail_under` — currently **95%** for
`core` and **96%** for `api`. `pytest --cov=...` enforces it: a drop below the floor
fails the run with the missing-lines report. The rigor modules (llm cost paths,
`metrics`/oracle/baselines/frontier, graders) sit at ~100% individually. Don't lower
the floor to make a change pass — add the missing tests.

## Branch protection (maintainers)

On `main`, require the **CI** workflow's jobs to pass before merge
(*Settings → Branches → Branch protection rules*): `lint-type`, `test-python`
(both `3.11` and `3.12`), `test-frontend`, `build-smoke`. Do **not** require
`api-tests.yml` (it is secret-gated and would block forks). Keep "Require branches to
be up to date" on so the gate runs against the merge result.

## Conventions

- Keep `core` framework-free; `api` is a thin adapter with **no engine logic**.
- Pure functions stay I/O- and log-free; edge modules (`llm.call`, `route`, the API
  middleware) do the logging.
- Never weaken a gate to go green — fix the code or add the test.
