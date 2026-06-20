# app: FrugalRoute web frontend

> Part of **FrugalRoute**. See the [root README](../README.md) for the headline,
> the proof, and the quickstart.

The user-facing frontend: the high-fidelity **dc-runtime Design Component**
(`FrugalRoute.dc.html` + `support.js`), wired to the live FrugalRoute API. It
renders what the backend returns and holds no business logic of its own: every
answer, gate verdict, cost, and price comes from `api/` over HTTP at runtime.

## Layout

```
app/
  FrugalRoute.dc.html   # the Design Component (template + embedded logic)
  support.js            # the dc-runtime (vendored; do not edit)
  index.html            # entry: redirects "/" -> FrugalRoute.dc.html
  src/                  # framework-free, unit-tested ES modules
    config.js           #   API base-URL resolution
    api.js              #   typed-ish client + normalized ApiError
    format.js           #   pure mapping/format helpers + the view model
    bridge.js           #   publishes the above on window.FR for the dc logic
  tests/
    unit/               #   node --test (no browser, no network)
    e2e/                #   Playwright vs a mocked /api
```

The embedded `<script data-dc-script>` is evaluated by the dc-runtime (it cannot
`import`), so `bridge.js` (loaded as `<script type="module">`) publishes the
client and formatters on `window.FR`.

## Run it locally

The dc-runtime `fetch`es `location.href` to refresh its template, so the app
**must be served over HTTP**; `file://` will not boot.

```bash
# serve the app (any static server works)
cd app && node tests/e2e/static-server.mjs 5500      # -> http://localhost:5500/
#   or:  python -m http.server 5500
```

Then open `http://localhost:5500/`. Point the page at a running API one of three ways:

- `window.FRUGALROUTE_API = "http://localhost:8000/api"` (e.g. via the console
  before first route), or
- a `<meta name="frugalroute-api" content="…">`, or
- the default same-origin `/api` (reverse-proxied in the production container).

Start the API from `api/`: `uvicorn frugalroute_api.app:app --reload`
(set `FRUGALROUTE_BACKEND=azure` to use the gpt-5.5 adapter on this box). With no
key, `/api/route` returns a 503 `missing-key` and the UI shows the blocking card
with a **"View the Proof ->"** escape hatch.

## Test

```bash
cd app
npm install
npm test            # unit: format/api/config/state (node --test), the gate
npm run test:e2e    # Playwright vs a mocked /api (needs: npx playwright install chromium)
```

The unit suite is the no-browser gate. The e2e drives the eight Single-Query
states against mocked `/api` responses; its browser binary is environment-gated
(`npx playwright install chromium`).
