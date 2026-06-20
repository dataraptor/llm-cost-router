# FrugalRoute API — multi-stage image (split 13).
#
# Build context is the REPO ROOT (not ./api): the api package depends on the
# sibling `core` package (frugalroute), installed as a local path dep, so the
# build needs both trees. Compose sets `context: .` for this reason. Manual:
#
#   docker build -t frugalroute-api -f api/Dockerfile .
#
# Stage 1 installs core + api (with the azure live-backend extra) into a venv;
# stage 2 is a slim runtime that only copies the venv + the api source. No build
# toolchain, no source for `core` (it lives in site-packages) reaches the final
# image, and the API key is never baked in — it enters only via runtime env.

# Pinned base (digest of python:3.12-slim-bookworm, linux/amd64).
FROM python:3.12-slim-bookworm@sha256:76d4b7b6305788c6b4c6a19d6a22a3921bf802e9af4d5e1e5bd771208dba74bf AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Isolated venv so we can copy exactly the runtime deps (no pip/setuptools cruft
# carried implicitly from the system) into the final stage.
RUN python -m venv /opt/venv

WORKDIR /build
# Copy only what the install needs (the .dockerignore keeps tmp/runs/.git/tests
# out of the context). core first — api's install resolves `frugalroute` from it.
COPY core/ ./core/
COPY api/ ./api/
# Install the engine (+ azure live backend) then the thin api adapter. The
# committed sample bundle ships as api package data, so /api/eval/sample works
# in-container with no key.
RUN pip install ./core[azure] ./api


# ---- runtime ----
FROM python:3.12-slim-bookworm@sha256:76d4b7b6305788c6b4c6a19d6a22a3921bf802e9af4d5e1e5bd771208dba74bf AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# curl for the HEALTHCHECK; non-root user for the runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

USER appuser
WORKDIR /home/appuser

EXPOSE 8000

# has_api_key reflects the runtime env (the key is not present at build time);
# /api/health is the readiness signal the app container waits on.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "frugalroute_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
