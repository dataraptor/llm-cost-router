// api.js — the typed-ish HTTP client for the FrugalRoute API (split-07 §2).
//
// Every call returns a Promise and throws a normalized {@link ApiError} on a
// non-2xx response, surfacing the split-06 error `type` so the UI can pick the
// right state (`missing-key` → blocking "View the Proof" card, `api-error` →
// inline card, …). No DOM, no app state — unit-testable with a mock `fetch`.

import { apiBaseUrl } from "./config.js";

/** A normalized API failure carrying the split-06 error `type` + HTTP `status`. */
export class ApiError extends Error {
  constructor(type, message, status) {
    super(message || type || "request failed");
    this.name = "ApiError";
    this.type = type || "error";
    this.status = typeof status === "number" ? status : 0;
  }
}

/** Parse a body as JSON, tolerating empty/non-JSON bodies (→ null). */
function parseJson(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Issue one request and normalize failures.
 *
 * @param {string} path        e.g. "/route" (joined onto the resolved base)
 * @param {object} [init]      fetch init
 * @param {object} [opts]
 * @param {string}   [opts.base]    override the base URL (tests)
 * @param {Function} [opts.fetchFn] override fetch (tests)
 */
async function request(path, init, opts = {}) {
  const base = opts.base !== undefined ? opts.base : apiBaseUrl();
  const fetchFn = opts.fetchFn || (typeof fetch !== "undefined" ? fetch : null);
  if (!fetchFn) throw new ApiError("network-error", "fetch is not available", 0);

  let res;
  try {
    res = await fetchFn(base + path, init);
  } catch (err) {
    throw new ApiError("network-error", (err && err.message) || "network request failed", 0);
  }

  const text = typeof res.text === "function" ? await res.text() : "";
  const body = parseJson(text);

  if (!res.ok) {
    const errObj = body && typeof body === "object" ? body.error : null;
    const type = errObj && errObj.type ? errObj.type : "api-error";
    const message =
      (errObj && errObj.message) ||
      (body && body.detail) ||
      "HTTP " + res.status;
    throw new ApiError(type, message, res.status);
  }
  return body;
}

/** GET /config → {prompt_version, model_tiers, pricing, always_strong_cost_ref_usd, defaults, …}. */
export function getConfig(opts) {
  return request("/config", undefined, opts);
}

/** GET /examples → [{id, benchmark, label, query}]. */
export function getExamples(opts) {
  return request("/examples", undefined, opts);
}

// Query-string params the stream endpoint accepts (EventSource is GET-only).
const STREAM_PARAMS = ["strategy", "query", "example_id", "benchmark", "tau", "theta"];

/**
 * Build the `GET /route/stream` URL for an EventSource from a route body.
 * Pure (no DOM) so it is unit-testable; the dc-runtime passes the result to
 * `new EventSource(url)`. Only present, non-null params are serialized.
 *
 * @param {object} body  same shape as {@link postRoute}'s body
 * @param {object} [opts] {base} to override the resolved API base (tests)
 */
export function routeStreamUrl(body, opts = {}) {
  const base = opts.base !== undefined ? opts.base : apiBaseUrl();
  const params = new URLSearchParams();
  const b = body || {};
  for (const key of STREAM_PARAMS) {
    if (b[key] !== undefined && b[key] !== null) params.set(key, String(b[key]));
  }
  return base + "/route/stream?" + params.toString();
}

/** POST /route with {strategy, query|example_id, benchmark?, tau?, theta?} → RouteResponse. */
export function postRoute(body, opts) {
  return request(
    "/route",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {}),
    },
    opts,
  );
}

/**
 * GET /eval/sample → the precomputed `EvalReport` bundle that powers the Frontier
 * view: `{reports:[cascade,predictive], benchmark, frozen_split, generated_at}`.
 * A missing bundle is a 404 → `ApiError(type:"not-found")`, which the UI renders
 * as its honest N/A empty state (split-08 §4).
 */
export function getEvalSample(opts) {
  return request("/eval/sample", undefined, opts);
}

/**
 * POST /eval with {strategy, benchmark, quick:true} → a freshly-computed bundle
 * (same shape as {@link getEvalSample}). Used by the optional live "Run eval"
 * action; the view re-renders through the same mapping. Bounded/synchronous per
 * the split-06 contract.
 */
export function postEval(body, opts) {
  return request(
    "/eval",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {}),
    },
    opts,
  );
}
