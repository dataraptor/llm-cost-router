// config.js — resolve the API base URL for the frontend (split-07 §1).
//
// Resolution order (first hit wins):
//   1. `window.FRUGALROUTE_API`  — set by an embedding page / e2e harness
//   2. `<meta name="frugalroute-api" content="...">`
//   3. default `"/api"`          — same-origin (the split-13 reverse-proxy setup)
//
// Resolved lazily (at request time, not module-load) so a harness can set
// `window.FRUGALROUTE_API` after the module is imported.

const DEFAULT_API_BASE = "/api";

/** Strip a single trailing slash so `base + "/route"` never doubles up. */
function trimTrailingSlash(value) {
  return typeof value === "string" ? value.replace(/\/+$/, "") : value;
}

/**
 * The API base URL the client should call.
 *
 * @param {object} [opts]
 * @param {Window}   [opts.win] window-like object (defaults to global `window`)
 * @param {Document} [opts.doc] document-like object (defaults to global `document`)
 * @returns {string}
 */
export function apiBaseUrl(opts = {}) {
  const win = opts.win !== undefined ? opts.win : globalThis.window;
  const doc =
    opts.doc !== undefined
      ? opts.doc
      : typeof document !== "undefined"
        ? document
        : undefined;

  if (win && win.FRUGALROUTE_API) {
    return trimTrailingSlash(String(win.FRUGALROUTE_API)) || DEFAULT_API_BASE;
  }
  if (doc && typeof doc.querySelector === "function") {
    const meta = doc.querySelector('meta[name="frugalroute-api"]');
    const content = meta && meta.getAttribute("content");
    if (content) return trimTrailingSlash(content) || DEFAULT_API_BASE;
  }
  return DEFAULT_API_BASE;
}

export { DEFAULT_API_BASE };
