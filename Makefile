# FrugalRoute — one-command stack + common tasks (split-10 §run-scripts).
# Cross-platform note: these targets use a POSIX shell (bash / WSL / Git Bash).
# On Windows PowerShell without `make`, use the equivalent scripts/*.ps1.

PY ?= python
API_PORT ?= 8000
APP_PORT ?= 5500
SAMPLE := api/src/frugalroute_api/data/sample_run.json

.PHONY: install dev eval sample screenshot test test-py test-app help

help:
	@echo "make install   - editable install of core[dev] + api[dev]"
	@echo "make dev       - serve api (:$(API_PORT)) + app (:$(APP_PORT), same-origin /api proxy)"
	@echo "make sample    - regenerate the committed sample bundle (no key)"
	@echo "make eval      - live eval (needs ANTHROPIC_API_KEY), then re-bundle"
	@echo "make screenshot- recapture docs/frontier.png from the committed sample"
	@echo "make test      - no-key gates: core + api + integration + app"

install:
	$(PY) -m pip install -e "core[dev]"
	$(PY) -m pip install -e "api[dev]"

# Serve the whole stack same-origin: uvicorn in the background, the app static
# server (proxying /api -> the api) in the foreground. Ctrl-C stops both.
dev:
	@echo "FrugalRoute dev stack:  app http://localhost:$(APP_PORT)/   (api on :$(API_PORT))"
	@$(PY) -m uvicorn frugalroute_api.app:app --port $(API_PORT) & echo $$! > .uvicorn.pid; \
	trap 'kill `cat .uvicorn.pid` 2>/dev/null; rm -f .uvicorn.pid' EXIT; \
	FRUGALROUTE_API_PROXY=http://localhost:$(API_PORT) node app/tests/e2e/static-server.mjs $(APP_PORT)

# No-key reproducible source run -> committed bundle.
sample:
	$(PY) scripts/gen_sample_run.py
	$(PY) scripts/bundle_sample.py eval/runs/sample.jsonl $(SAMPLE)

# Live eval (needs a key) -> committed bundle.
eval:
	$(PY) -m frugalroute.cli eval --strategy both --benchmark gsm8k --out eval/runs/sample.jsonl
	$(PY) scripts/bundle_sample.py eval/runs/sample.jsonl $(SAMPLE)

screenshot:
	cd app && npm run screenshot

test: test-py test-app

# core and api are separate packages (each its own rootdir) and share some test
# basenames, so they run as separate pytest invocations.
test-py:
	pytest core/tests -m "not api and not azure" -q
	pytest api/tests -m "not api and not azure" -q
	pytest tests/integration -m "not api and not azure" -q

test-app:
	cd app && npm test
