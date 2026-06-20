// Unit tests for API base-URL resolution (split-07 §1).

import { test } from "node:test";
import assert from "node:assert/strict";

import { apiBaseUrl, DEFAULT_API_BASE } from "../../src/config.js";

test("window.FRUGALROUTE_API wins, trailing slash trimmed", () => {
  const win = { FRUGALROUTE_API: "https://x.test/api/" };
  assert.equal(apiBaseUrl({ win, doc: undefined }), "https://x.test/api");
});

test("falls back to <meta name='frugalroute-api'>", () => {
  const doc = {
    querySelector: (sel) =>
      sel === 'meta[name="frugalroute-api"]' ? { getAttribute: () => "http://m.test/api/" } : null,
  };
  assert.equal(apiBaseUrl({ win: {}, doc }), "http://m.test/api");
});

test("defaults to '/api' when nothing is set", () => {
  assert.equal(apiBaseUrl({ win: {}, doc: { querySelector: () => null } }), DEFAULT_API_BASE);
  assert.equal(apiBaseUrl({ win: undefined, doc: undefined }), "/api");
});
