// e2e: the Single-Query view driven by a mocked SSE endpoint (split-09 tests 8–14
// + the R10 adversarial check). The app opens a real EventSource to
// /api/route/stream; we fulfill that stream with hand-crafted frame sequences and
// assert the live stepper / candidate / gate / cost choreography, plus that a
// broken / out-of-order stream recovers (fallback or honest error), never hangs.

import { test, expect } from "@playwright/test";
import * as F from "./fixtures.js";
import { mockStream, sseBody } from "./sse.js";

const PAGE = "/FrugalRoute.dc.html";
const CHEAP = "claude-haiku-4-5";
const STRONG = "claude-opus-4-8";

async function mockMeta(page) {
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
}

/** Real console errors + dc-runtime "never resolved" binding warnings (raw-template
 *  SVG `{{ }}` parse noise filtered, as in single-query.spec.js). */
function collectProblems(page) {
  const problems = [];
  page.on("console", (msg) => {
    const t = msg.text();
    if (t.includes("{{") && /Expected /.test(t)) return;
    if (msg.type() === "error") problems.push("console.error: " + t);
    if (/never resolved/.test(t)) problems.push("unresolved binding: " + t);
  });
  page.on("pageerror", (err) => problems.push("pageerror: " + err.message));
  return problems;
}

async function clickRoute(page) {
  await page.getByRole("button", { name: "Route", exact: true }).click();
}

const sq = (page) => page.locator('[data-screen-label="single-query"]');

// --- test 8: accepted stream choreography ----------------------------------
test("accepted stream: candidate appears, gate fills, cost assembles, Opus stays hollow", async ({ page }) => {
  const problems = collectProblems(page);
  await mockMeta(page);
  await mockStream(page, [
    ["phase", { phase: "gen", tier: CHEAP }],
    ["candidate", { answer: "April: 48. May: 24. Total = 72.", tier: CHEAP, cost_usd: 0.0014 }],
    ["cost", { cost_usd_cumulative: 0.0014 }],
    ["phase", { phase: "gate", tier: CHEAP }],
    ["gate", { sufficient: true, confidence: 0.91, reason: "Commits to a single number.", cost_usd: 0.0004 }],
    ["cost", { cost_usd_cumulative: 0.0018 }],
    ["done", F.ACCEPTED],
  ]);
  await page.goto(PAGE);
  await clickRoute(page);

  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible();
  await expect(page.getByText("sufficient · conf")).toBeVisible();
  await expect(sq(page)).toContainText("= Haiku + gate");
  await expect(sq(page)).toContainText("$0.0018"); // cost assembled to the cheap+gate total
  await expect(page.locator("text=/▲/").first()).toBeVisible(); // savings ticked up
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 9: escalated stream ----------------------------------------------
test("escalated stream: third link draws, Opus solid, cost → additive total, honest loss", async ({ page }) => {
  const problems = collectProblems(page);
  await mockMeta(page);
  await page.goto(PAGE);
  await page.locator("select").selectOption("gsm8k-882");
  await mockStream(page, [
    ["phase", { phase: "gen", tier: CHEAP }],
    ["candidate", { answer: "Robe 1: 3…", tier: CHEAP, cost_usd: 0.0014 }],
    ["cost", { cost_usd_cumulative: 0.0014 }],
    ["phase", { phase: "gate", tier: CHEAP }],
    ["gate", { sufficient: false, confidence: 0.58, reason: "Hedged.", cost_usd: 0.0004 }],
    ["phase", { phase: "escalate", tier: STRONG }],
    ["cost", { cost_usd_cumulative: 0.0088 }],
    ["done", F.ESCALATED],
  ]);
  await clickRoute(page);

  await expect(page.getByText("Answer · escalated")).toBeVisible();
  await expect(page.getByText("insufficient · conf")).toBeVisible();
  await expect(sq(page)).toContainText("= Haiku + gate + Opus");
  await expect(sq(page)).toContainText("more than Opus-only ($0.0070)");
  await expect(sq(page)).toContainText("$0.0088");
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 10: 429 retry sublabel -------------------------------------------
test("retry event: node shows 'retrying (rate-limited)…' sublabel, never an error toast", async ({ page }) => {
  const problems = collectProblems(page);
  await mockMeta(page);
  // The stream stalls at a retry (no further frames / no done) and the POST
  // fallback is slow, so the UI sits visibly on the retry sublabel — proving a
  // 429 backoff is surfaced as an informational wait, not an error.
  await mockStream(page, [
    ["phase", { phase: "gen", tier: CHEAP }],
    ["retry", { stage: "gen", reason: "rate-limited" }],
  ]);
  await page.route("**/api/route", async (r) => {
    await new Promise((res) => setTimeout(res, 3000)); // slow retry/fallback
    r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(F.ACCEPTED) });
  });
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText("retrying (rate-limited)…")).toBeVisible();
  await expect(page.getByText("Routing error")).toHaveCount(0); // retry is NOT an error
  // and it still resolves to the real answer once the (slow) call returns
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible();
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 11: refusal(strong) mid-stream -----------------------------------
test("refusal(strong): alert-edged 'surfaced, not hidden' card, no crash", async ({ page }) => {
  await mockMeta(page);
  await mockStream(page, [
    ["phase", { phase: "gen", tier: CHEAP }],
    ["candidate", { answer: "…", tier: CHEAP, cost_usd: 0.0014 }],
    ["cost", { cost_usd_cumulative: 0.0014 }],
    ["phase", { phase: "gate", tier: CHEAP }],
    ["gate", { sufficient: false, confidence: 0.0, reason: "doubt", cost_usd: 0.0004 }],
    ["phase", { phase: "escalate", tier: STRONG }],
    ["refusal", { stage: "strong", message: "refusal" }],
    ["cost", { cost_usd_cumulative: 0.0086 }],
    ["done", F.STRONG_REFUSAL],
  ]);
  await page.goto(PAGE);
  await page.locator("select").selectOption("refuse-1");
  await clickRoute(page);
  await expect(page.getByText(/Surfaced, not hidden/)).toBeVisible();
  await expect(page.getByText("(no answer — the model returned a refusal)")).toBeVisible();
});

// --- test 12: error{missing-key} event -------------------------------------
test("error{missing-key} event: blocking card + 'View the Proof' escape hatch", async ({ page }) => {
  await mockMeta(page);
  await mockStream(page, [
    ["phase", { phase: "gen", tier: CHEAP }],
    ["error", { type: "missing-key", message: F.MISSING_KEY.error.message }],
  ]);
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText(/ANTHROPIC_API_KEY/)).toBeVisible();
  const proof = page.getByRole("button", { name: /View the Proof/ });
  await expect(proof).toBeVisible();
  await proof.click();
  await expect(page.getByText("Leaderboard")).toBeVisible();
});

// --- test 13: predictive stream --------------------------------------------
test("predictive stream: embed→classify→done, decision margin, no gate node", async ({ page }) => {
  const problems = collectProblems(page);
  await mockMeta(page);
  await mockStream(page, [
    ["phase", { phase: "embed", tier: null }],
    ["phase", { phase: "classify", tier: CHEAP }],
    ["cost", { cost_usd_cumulative: 0.0014 }],
    ["done", F.PREDICTIVE],
  ]);
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Predictive", exact: true }).click();
  await clickRoute(page);
  await expect(page.getByText("Decision margin")).toBeVisible();
  await expect(page.getByText(/predicted upfront — no cheap call, no gate/)).toBeVisible();
  await expect(page.getByText(/sufficient · conf/)).toHaveCount(0);
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 14a (reduced motion) ---------------------------------------------
test("reduced motion: stream sets the final state instantly, correct", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockMeta(page);
  await mockStream(page, [
    ["phase", { phase: "gen", tier: CHEAP }],
    ["candidate", { answer: "72", tier: CHEAP, cost_usd: 0.0014 }],
    ["cost", { cost_usd_cumulative: 0.0014 }],
    ["phase", { phase: "gate", tier: CHEAP }],
    ["gate", { sufficient: true, confidence: 0.91, reason: "ok", cost_usd: 0.0004 }],
    ["cost", { cost_usd_cumulative: 0.0018 }],
    ["done", F.ACCEPTED],
  ]);
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText("$0.0018")).toBeVisible({ timeout: 2000 });
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible({ timeout: 2000 });
});

// --- test 14b: fallback when EventSource transport fails --------------------
test("fallback: stream transport fails → postRoute renders the correct final state", async ({ page }) => {
  await mockMeta(page);
  await page.route("**/api/route/stream**", (r) => r.abort()); // force EventSource error
  await page.route("**/api/route", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(F.ACCEPTED) }),
  );
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible();
  await expect(sq(page)).toContainText("$0.0018");
});

// --- R10 (a): server closes after phase:gate, before done → no hang --------
test("R10: stream closes after phase:gate (no done) → fallback, never a half-filled stepper", async ({ page }) => {
  await mockMeta(page);
  // a truncated stream: events up to the gate, then EOF (no done event)
  await page.route("**/api/route/stream**", (r) =>
    r.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody([
        ["phase", { phase: "gen", tier: CHEAP }],
        ["candidate", { answer: "partial…", tier: CHEAP, cost_usd: 0.0014 }],
        ["cost", { cost_usd_cumulative: 0.0014 }],
        ["phase", { phase: "gate", tier: CHEAP }],
        ["gate", { sufficient: true, confidence: 0.91, reason: "ok", cost_usd: 0.0004 }],
      ]),
    }),
  );
  // the fallback POST returns the real, complete result
  await page.route("**/api/route", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(F.ACCEPTED) }),
  );
  await page.goto(PAGE);
  await clickRoute(page);
  // recovers to the real final answer — never stuck showing the partial candidate
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible();
  await expect(sq(page)).not.toContainText("partial…");
});

// --- R10 (b): out-of-order events → no crash, partial never shown as final --
test("R10: out-of-order events → no crash; partial answer never shown as final", async ({ page }) => {
  const problems = collectProblems(page);
  await mockMeta(page);
  // scrambled order: gate before its phase, cost early — but done arrives last
  await page.route("**/api/route/stream**", (r) =>
    r.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody([
        ["gate", { sufficient: true, confidence: 0.91, reason: "ok", cost_usd: 0.0004 }],
        ["cost", { cost_usd_cumulative: 0.0018 }],
        ["phase", { phase: "gate", tier: CHEAP }],
        ["candidate", { answer: "scrambled candidate", tier: CHEAP, cost_usd: 0.0014 }],
        ["phase", { phase: "gen", tier: CHEAP }],
        ["done", F.ACCEPTED],
      ]),
    }),
  );
  await page.goto(PAGE);
  await clickRoute(page);
  await expect(page.getByText("Accepted at Haiku 4.5")).toBeVisible();
  const body = await sq(page).innerText();
  expect(body).not.toMatch(/NaN|undefined/);
  expect(body).not.toMatch(/scrambled candidate/); // the real answer replaced it
  expect(problems, problems.join("\n")).toEqual([]);
});
