// Full-stack browser integration (split-10 tests 1-4, R9-browser): the REAL api
// (uvicorn, no key) + the app, no /api mocking. Proves the committed proof renders
// in a real browser and the no-key single-query path is honest end to end.
//
//   cd app && npx playwright test --config playwright.integration.config.js

import { test, expect } from "@playwright/test";

const PAGE = "/FrugalRoute.dc.html";

/** Point the app at the real api origin (cross-port; api CORS is "*"). */
test.beforeEach(async ({ page }, testInfo) => {
  const apiBase = testInfo.config.metadata.apiBase;
  await page.addInitScript((base) => {
    window.FRUGALROUTE_API = base;
  }, apiBase);
});

/** Real console errors + unresolved-binding warnings (filter the benign raw
 *  <x-dc> template {{ }} SVG parse noise served with any dc page). */
function collectProblems(page) {
  const problems = [];
  page.on("console", (msg) => {
    const t = msg.text();
    if (t.includes("{{")) return;
    if (/Failed to load resource/.test(t)) return; // the no-key 503 on /route is expected
    if (msg.type() === "error") problems.push("console.error: " + t);
    if (/never resolved/.test(t)) problems.push("unresolved binding: " + t);
  });
  page.on("pageerror", (err) => problems.push("pageerror: " + err.message));
  return problems;
}

const frontier = (page) => page.locator('[data-screen-label="frontier"]');

// --- Tests 1+2 (R3/R1): the stack boots, Frontier renders the committed proof -
test("Frontier renders the committed sample over the real api (headline + 6 rows, no console errors)", async ({ page }) => {
  const problems = collectProblems(page);
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Proof", exact: true }).click();
  await expect(frontier(page)).toBeVisible();

  // The real headline from the committed run (retention 100%).
  const headline = page.getByText(/Retains .* of Opus accuracy at .* of the cost/);
  await expect(headline).toBeVisible();
  await expect(headline).toContainText("100%");

  // The 6-row leaderboard, in order, sourced from the live bundle.
  for (const name of [
    "always-cheap",
    "always-strong",
    "random @ cost",
    "FrugalRoute · cascade",
    "FrugalRoute · predictive",
    "oracle (ceiling)",
  ]) {
    await expect(frontier(page).getByText(name, { exact: true })).toBeVisible();
  }
  // No fabricated numbers leaked through.
  const body = await frontier(page).innerText();
  expect(body).not.toMatch(/NaN|undefined/);
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- Test 3 (R4): no-key single query → honest missing-key card + Proof escape -
test("no-key single query → missing-key card, then 'View the Proof' lands on the populated Frontier", async ({ page }) => {
  await page.goto(PAGE);
  await expect(page.locator("textarea")).toHaveValue(/Natalia sold clips/); // booted off /config+/examples
  await page.getByRole("button", { name: "Route", exact: true }).click();

  // The live api has no key → the engine raises missing-key → the blocking card.
  await expect(page.getByText("Live routing unavailable")).toBeVisible();
  const proof = page.getByRole("button", { name: /View the Proof/ });
  await expect(proof).toBeVisible();

  // The escape hatch lands on the real, populated Frontier (no key needed).
  await proof.click();
  await expect(frontier(page)).toBeVisible();
  await expect(page.getByText(/Retains .* of Opus accuracy/)).toBeVisible();
});
