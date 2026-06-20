// e2e: the Single-Query view driven by a mocked /api (split-07 tests 9–15 + R10).
// /api/* is intercepted with page.route(); the app is served same-origin so
// config.js resolves the default "/api". React loads from the CDN (vendored in
// split 13).

import { test, expect } from "@playwright/test";
import * as F from "./fixtures.js";
import { framesForResult, mockStream, failStream } from "./sse.js";

const PAGE = "/FrugalRoute.dc.html";

/** Register config+examples mocks (always) plus a per-test /route handler.
 *
 * split 09: the single-query view is driven by the SSE stream, so the `route`
 * fixture is delivered as an event stream (its terminal `done`/`error` carries the
 * fixture). The POST `/api/route` is also mocked as the fallback contract. */
async function mockApi(page, { route, routeStatus = 200 } = {}) {
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
  if (route !== undefined) {
    const streamFixture = routeStatus >= 400 ? { error: route.error } : route;
    await mockStream(page, framesForResult(streamFixture));
    await page.route("**/api/route", (r) =>
      r.fulfill({ status: routeStatus, contentType: "application/json", body: JSON.stringify(route) }),
    );
  }
}

/** Collect *real* console errors + dc-runtime "never resolved" binding warnings.
 *
 * The hidden raw <x-dc> template carries {{ }} placeholders inside SVG attributes;
 * the browser parses that template once on load and logs benign "<rect> attribute
 * y: Expected length, '{{ p.ry }}'" errors BEFORE the dc-runtime swaps in the
 * rendered tree. Those are an artifact of serving any dc page (the original mockup
 * does it too), not app errors — filtered out here. A genuinely missing binding
 * still surfaces via the dc-runtime's "never resolved" warning. */
function collectProblems(page) {
  const problems = [];
  page.on("console", (msg) => {
    const t = msg.text();
    if (t.includes("{{") && /Expected /.test(t)) return; // raw-template SVG parse noise
    if (msg.type() === "error") problems.push("console.error: " + t);
    if (/never resolved/.test(t)) problems.push("unresolved binding: " + t);
  });
  page.on("pageerror", (err) => problems.push("pageerror: " + err.message));
  return problems;
}

async function clickRoute(page) {
  await page.getByRole("button", { name: "Route", exact: true }).click();
}

// --- test 9: boot, fetch config+examples, prefill the first example query ---
test("boots, fetches config+examples, prefills the first example query", async ({ page }) => {
  await mockApi(page);
  const problems = collectProblems(page);
  await page.goto(PAGE);
  await expect(page.locator("textarea")).toHaveValue(/Natalia sold clips/);
  // provenance from /config (not hardcoded)
  await expect(page.getByText("prompt_version: v3")).toBeVisible();
  await expect(page.getByText("tiers: haiku-4-5 → opus-4-8")).toBeVisible();
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 10: accepted-cheap ------------------------------------------------
test("accepted-cheap: stepper ends Haiku, gate sufficient, cost '= Haiku + gate', savings up", async ({ page }) => {
  await mockApi(page, { route: F.ACCEPTED });
  const problems = collectProblems(page);
  await page.goto(PAGE);
  await clickRoute(page);
  const sq = page.locator('[data-screen-label="single-query"]');
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible();
  await expect(page.getByText("sufficient · conf")).toBeVisible();
  // cost number + breakdown render as split text nodes ($ literal + {{ costStr }});
  // assert on the section's combined text instead of a single-node match.
  await expect(sq).toContainText("= Haiku + gate");
  await expect(sq).toContainText("$0.0018");
  // savings band ticked up green (▲, positive)
  await expect(page.locator("text=/▲/").first()).toBeVisible();
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 11: escalated (honest loss) --------------------------------------
test("escalated: full stepper, gate insufficient, ⚠ more than Opus-only, savings can go negative", async ({ page }) => {
  await mockApi(page, { route: F.ESCALATED });
  const problems = collectProblems(page);
  await page.goto(PAGE);
  // pick the hard example then route
  await page.locator("select").selectOption("gsm8k-882");
  await clickRoute(page);
  const sq = page.locator('[data-screen-label="single-query"]');
  await expect(page.getByText("Answer · escalated")).toBeVisible();
  await expect(page.getByText("insufficient · conf")).toBeVisible();
  await expect(sq).toContainText("= Haiku + gate + Opus");
  // the §3a binding shows the real reference, not a hardcoded literal
  await expect(sq).toContainText("more than Opus-only ($0.0070)");
  await expect(sq).toContainText("$0.0088");
  // savings ticked down (clay ▼)
  await expect(page.locator("text=/▼/").first()).toBeVisible();
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 12: strong refusal surfaced --------------------------------------
test("strong refusal: alert-edged answer card, 'surfaced, not hidden', no crash", async ({ page }) => {
  await mockApi(page, { route: F.STRONG_REFUSAL });
  const problems = collectProblems(page);
  await page.goto(PAGE);
  await page.locator("select").selectOption("refuse-1");
  await clickRoute(page);
  // the honest "surfaced, not hidden" refusal note + the empty-answer placeholder
  await expect(page.getByText(/Surfaced, not hidden/)).toBeVisible();
  await expect(page.getByText("(no answer — the model returned a refusal)")).toBeVisible();
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 13: missing key (blocking card + escape hatch) -------------------
test("missing-key 503: blocking card names ANTHROPIC_API_KEY, 'View the Proof' opens the Frontier", async ({ page }) => {
  await mockApi(page, { route: F.MISSING_KEY, routeStatus: 503 });
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText(/ANTHROPIC_API_KEY/)).toBeVisible();
  const proof = page.getByRole("button", { name: /View the Proof/ });
  await expect(proof).toBeVisible();
  await proof.click();
  // Frontier view is now showing (its leaderboard + headline render from mockup data)
  await expect(page.getByText("Leaderboard")).toBeVisible();
});

// --- test 14: predictive (no gate node, margin, no checkmark) --------------
test("predictive: no gate card, decision margin shown, correctness never shown", async ({ page }) => {
  await mockApi(page, { route: F.PREDICTIVE });
  const problems = collectProblems(page);
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Predictive", exact: true }).click();
  await clickRoute(page);
  await expect(page.getByText("Decision margin")).toBeVisible();
  await expect(page.getByText("P(needs strong)")).toBeVisible();
  await expect(page.getByText(/predicted upfront — no cheap call, no gate/)).toBeVisible();
  // correctness is never shown in live mode (no checkmark) — the honest note instead
  await expect(page.getByText(/correctness not shown/)).toBeVisible();
  // no gate verdict card
  await expect(page.getByText(/sufficient · conf/)).toHaveCount(0);
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 15: reduced motion (final state instant) -------------------------
test("prefers-reduced-motion: final accepted state renders without animation", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockApi(page, { route: F.ACCEPTED });
  await page.goto(PAGE);
  await clickRoute(page);
  // Under reduced motion the cost/savings are SET (not tweened) — assert the
  // final numbers are present essentially immediately.
  await expect(page.getByText("$0.0018")).toBeVisible({ timeout: 2000 });
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible({ timeout: 2000 });
});

// --- R10 adversarial: empty answer + malformed gate + cost 0 ---------------
test("adversarial: empty answer + malformed gate + cost 0 → honest, no crash/NaN", async ({ page }) => {
  await mockApi(page, { route: F.ADVERSARIAL });
  const problems = collectProblems(page);
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText("(no answer returned)")).toBeVisible();
  await expect(page.getByText("$0.0000")).toBeVisible();
  // no NaN / undefined anywhere in the rendered single-query region
  const body = await page.locator('[data-screen-label="single-query"]').innerText();
  expect(body).not.toMatch(/NaN|undefined/);
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- non-JSON 500 (R10 continued) ------------------------------------------
test("non-JSON 500 body → inline error card, dc-runtime stays alive", async ({ page }) => {
  await mockApi(page);
  // Force the postRoute fallback (stream transport fails) and let it hit a
  // non-JSON 500 — the client must still surface a structured error card.
  await failStream(page);
  await page.route("**/api/route", (r) =>
    r.fulfill({ status: 500, contentType: "text/html", body: "<html>Internal Server Error</html>" }),
  );
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText("Routing error")).toBeVisible();
  // app still interactive: switch strategy without a crash
  await page.getByRole("button", { name: "Predictive", exact: true }).click();
  await expect(page.getByText("Routing error")).toHaveCount(0);
});
