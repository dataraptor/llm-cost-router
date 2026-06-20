// Unit tests for the API client + error mapping (split-07 tests 5–8).
// `fetch` is injected via opts.fetchFn so these run with no network.

import { test } from "node:test";
import assert from "node:assert/strict";

import { ApiError, getConfig, getExamples, postRoute, getEvalSample, postEval } from "../../src/api.js";

/** A minimal fake Response. */
function fakeRes(status, body) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return { ok: status >= 200 && status < 300, status, text: async () => text };
}

/** Capture the calls and return a scripted response. */
function fakeFetch(res) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    if (res instanceof Error) throw res;
    return typeof res === "function" ? res(url, init) : res;
  };
  fn.calls = calls;
  return fn;
}

const BASE = "http://api.test/api";

// --- test 5: postRoute posts the right body and parses a RouteResult --------

test("postRoute posts JSON body to /route and parses the result (test 5)", async () => {
  const fetchFn = fakeFetch(
    fakeRes(200, {
      query: "q",
      strategy: "cascade",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "42",
      cost_usd: 0.0018,
      cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
    }),
  );
  const body = { strategy: "cascade", example_id: "gsm8k-1142", tau: 0.8 };
  const result = await postRoute(body, { base: BASE, fetchFn });

  assert.equal(fetchFn.calls.length, 1);
  const call = fetchFn.calls[0];
  assert.equal(call.url, BASE + "/route");
  assert.equal(call.init.method, "POST");
  assert.equal(call.init.headers["content-type"], "application/json");
  assert.deepEqual(JSON.parse(call.init.body), body);
  assert.equal(result.answer, "42");
  assert.equal(result.tier_used, "claude-haiku-4-5");
});

// --- test 6: 503 missing-key → ApiError{type:'missing-key'} ----------------

test("503 {error:{type:'missing-key'}} → ApiError missing-key (test 6)", async () => {
  const fetchFn = fakeFetch(
    fakeRes(503, { error: { type: "missing-key", message: "ANTHROPIC_API_KEY is not set.", detail: null } }),
  );
  await assert.rejects(
    postRoute({ strategy: "cascade", query: "q" }, { base: BASE, fetchFn }),
    (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.type, "missing-key");
      assert.equal(err.status, 503);
      assert.match(err.message, /ANTHROPIC_API_KEY/);
      return true;
    },
  );
});

// --- test 7: 502 api-error → ApiError{type:'api-error'} --------------------

test("502 {error:{type:'api-error'}} → ApiError api-error (test 7)", async () => {
  const fetchFn = fakeFetch(
    fakeRes(502, { error: { type: "api-error", message: "The model backend returned an error." } }),
  );
  await assert.rejects(
    postRoute({ strategy: "cascade", query: "q" }, { base: BASE, fetchFn }),
    (err) => err instanceof ApiError && err.type === "api-error" && err.status === 502,
  );
});

test("400 bad-request → ApiError bad-request", async () => {
  const fetchFn = fakeFetch(fakeRes(400, { error: { type: "bad-request", message: "Provide exactly one." } }));
  await assert.rejects(
    postRoute({ strategy: "cascade" }, { base: BASE, fetchFn }),
    (err) => err instanceof ApiError && err.type === "bad-request" && err.status === 400,
  );
});

// --- test 8: getConfig / getExamples parse & shape correctly ---------------

test("getConfig parses the config payload (test 8)", async () => {
  const cfg = {
    prompt_version: "v3",
    model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
    pricing: { "claude-haiku-4-5": { input_per_mtok: 1, output_per_mtok: 5 } },
    always_strong_cost_ref_usd: 0.0065,
    defaults: { tau: 0.8, theta: 0.6 },
    pricing_pinned_date: "2026-06-19",
  };
  const fetchFn = fakeFetch(fakeRes(200, cfg));
  const out = await getConfig({ base: BASE, fetchFn });
  assert.equal(fetchFn.calls[0].url, BASE + "/config");
  assert.deepEqual(out, cfg);
});

test("getExamples parses the example list (test 8)", async () => {
  const examples = [{ id: "gsm8k-1142", benchmark: "gsm8k", label: "g", query: "q" }];
  const fetchFn = fakeFetch(fakeRes(200, examples));
  const out = await getExamples({ base: BASE, fetchFn });
  assert.equal(fetchFn.calls[0].url, BASE + "/examples");
  assert.deepEqual(out, examples);
});

// --- robustness: network + non-JSON bodies ---------------------------------

test("network failure → ApiError{type:'network-error', status:0}", async () => {
  const fetchFn = fakeFetch(new TypeError("Failed to fetch"));
  await assert.rejects(
    getConfig({ base: BASE, fetchFn }),
    (err) => err instanceof ApiError && err.type === "network-error" && err.status === 0,
  );
});

test("non-JSON error body (e.g. 500 HTML) → structured ApiError, no crash (R10)", async () => {
  const fetchFn = fakeFetch(fakeRes(500, "<html>Internal Server Error</html>"));
  await assert.rejects(
    postRoute({ strategy: "cascade", query: "q" }, { base: BASE, fetchFn }),
    (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 500);
      assert.equal(err.type, "api-error"); // default when no {error:{type}}
      return true;
    },
  );
});

test("empty 2xx body → null (no parse crash)", async () => {
  const fetchFn = fakeFetch(fakeRes(204, ""));
  const out = await postRoute({ strategy: "cascade", query: "q" }, { base: BASE, fetchFn });
  assert.equal(out, null);
});

// --- split-08: the eval bundle client --------------------------------------

test("getEvalSample GETs /eval/sample and returns the bundle", async () => {
  const bundle = { reports: [{ strategy: "cascade", points: [] }], benchmark: "gsm8k" };
  const fetchFn = fakeFetch(fakeRes(200, bundle));
  const out = await getEvalSample({ base: BASE, fetchFn });
  assert.equal(fetchFn.calls[0].url, BASE + "/eval/sample");
  assert.equal(fetchFn.calls[0].init, undefined); // a plain GET
  assert.deepEqual(out, bundle);
});

test("getEvalSample 404 → ApiError{type:'not-found'} (drives the N/A state)", async () => {
  const fetchFn = fakeFetch(fakeRes(404, { error: { type: "not-found", message: "No precomputed eval sample." } }));
  await assert.rejects(
    getEvalSample({ base: BASE, fetchFn }),
    (err) => err instanceof ApiError && err.type === "not-found" && err.status === 404,
  );
});

test("postEval POSTs {quick:true} to /eval and returns the fresh bundle", async () => {
  const bundle = { reports: [], benchmark: "gsm8k" };
  const fetchFn = fakeFetch(fakeRes(200, bundle));
  const body = { strategy: "both", benchmark: "gsm8k", quick: true };
  const out = await postEval(body, { base: BASE, fetchFn });
  const call = fetchFn.calls[0];
  assert.equal(call.url, BASE + "/eval");
  assert.equal(call.init.method, "POST");
  assert.deepEqual(JSON.parse(call.init.body), body);
  assert.deepEqual(out, bundle);
});
