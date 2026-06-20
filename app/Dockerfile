# FrugalRoute app — static UI + same-origin /api reverse proxy (split 13).
#
# Build context is ./app (self-contained; no backend source needed):
#
#   docker build -t frugalroute-app -f app/Dockerfile ./app
#
# nginx-unprivileged already runs as a non-root user (uid 101) and listens on
# 8080 — no CDN is contacted at runtime: React is vendored (app/vendor) and loaded
# before support.js, so its CDN loader early-returns; Babel never fires (the page's
# src/*.js are plain ES modules, not x-imported JSX).

# Pinned base (digest of nginxinc/nginx-unprivileged:1.27-alpine).
FROM nginxinc/nginx-unprivileged:1.27-alpine@sha256:65e3e85dbaed8ba248841d9d58a899b6197106c23cb0ff1a132b7bfe0547e4c0

# Server block: static at /, reverse-proxy /api/* → api:8000, cache headers, gzip.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Only the files the served app needs (the .dockerignore also keeps node_modules,
# tests, and config out of the build context).
WORKDIR /usr/share/nginx/html
COPY index.html FrugalRoute.dc.html support.js ./
COPY src/ ./src/
COPY vendor/ ./vendor/

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD wget -q -O /dev/null http://localhost:8080/healthz || exit 1
