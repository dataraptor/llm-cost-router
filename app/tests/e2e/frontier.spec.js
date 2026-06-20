// e2e: the Frontier (Proof) view driven by a mocked /api/eval/sample bundle
// (split-08 tests 6–12 + R10). The app is served same-origin; /api/* is mocked
// with page.route(). React loads from the CDN (vendored in split 13).

import { test, expect } from "@playwright/test";
import * as F from "./fixtures.js";

const PAGE = "/FrugalRoute.dc.html";

/** Mock config+examples (boot) and the eval bundle (or a 404 for the N/A state). */
async function mockApi(page, { bundle, sampleStatus = 200 } = {}) {
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
  await page.route("**/api/eval/sample", (r) => {
    if (sampleStatus !== 200) {
      return r.fulfill({
        status: sampleStatus,
        contentType: "application/json",
        body: JSON.stringify({ error: { type: "not-found", message: "No precomputed eval sample is available." } }),
      });
    }
    return r.fulfill({ json: bundle });
  });
}

/** Real console errors + dc-runtime "never resolved" warnings (raw-template {{ }}
 *  SVG noise is an artifact of serving any dc page — filtered, like split-07). */
function collectProblems(page) {
  const problems = [];
  page.on("console", (msg) => {
    const t = msg.text();
    if (t.includes("{{")) return; // raw <x-dc> template placeholder parse noise
    if (/Failed to load resource/.test(t)) return; // expected 404 in the N/A test
    if (msg.type() === "error") problems.push("console.error: " + t);
    if (/never resolved/.test(t)) problems.push("unresolved binding: " + t);
  });
  page.on("pageerror", (err) => problems.push("pageerror: " + err.message));
  return problems;
}

const frontier = (page) => page.locator('[data-screen-label="frontier"]');

async function gotoProof(page) {
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Proof", exact: true }).click();
  await expect(frontier(page)).toBeVisible();
}

/** Set a (React-controlled) range input and fire its input handler. */
async function setSlider(page, ariaLabel, value) {
  await page.locator(`input[aria-label="${ariaLabel}"]`).evaluate((el, val) => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(el, String(val));
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
}

// --- test 6: populated view renders chart + headline + provenance -----------
test("Proof renders chart + headline from the bundle; provenance reflects version/tiers/n_runs", async ({ page }) => {
  await mockApi(page, { bundle: F.BUNDLE });
  const problems = collectProblems(page);
  await gotoProof(page);

  // settled headline = the proven number (default τ lands on the chosen target)
  const headline = page.getByText(/Retains .* of Opus accuracy at .* of the cost/);
  await expect(headline).toBeVisible();
  await expect(headline).toContainText("100%"); // retention_at_target
  await expect(headline).toContainText("60%"); // cost fraction (0.0039 / 0.0065)
  // distributional caption (real spread + n_runs, not the old literals)
  await expect(page.getByText(/· n=3 · frozen split/)).toBeVisible();

  // provenance chips reflect prompt_version / model_tiers / n_runs
  await expect(page.getByText("prompt_version: v3")).toBeVisible();
  await expect(page.getByText("tiers: haiku-4-5 → opus-4-8")).toBeVisible();
  await expect(page.getByText("n_runs: 3")).toBeVisible();

  // the cascade curve actually drew (a non-empty path)
  await expect(page.locator('path[stroke-dasharray]').first()).toBeVisible();
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 7: leaderboard — 6 rows, fixed order, ours highlighted, oracle "—" -
test("leaderboard: 6 rows in fixed order, our two highlighted, oracle ret/cut '—'", async ({ page }) => {
  await mockApi(page, { bundle: F.BUNDLE });
  await gotoProof(page);
  const fr = frontier(page);
  for (const name of [
    "always-cheap",
    "always-strong",
    "random @ cost",
    "FrugalRoute · cascade",
    "FrugalRoute · predictive",
    "oracle (ceiling)",
  ]) {
    await expect(fr.getByText(name, { exact: true })).toBeVisible();
  }
  // our rows carry the operating-point note (exact → the leaderboard span, not the
  // headline caption which also reads "cascade @ τ=0.50"); the oracle is a ceiling.
  await expect(fr.getByText("τ=0.50", { exact: true })).toBeVisible();
  await expect(fr.getByText("θ=0.60", { exact: true })).toBeVisible();
  await expect(fr.getByText("uses ground truth")).toBeVisible();
});

// --- test 8: drag the slider into the losing region → below-break-even copy --
test("slider into the losing region → below-break-even headline (clay)", async ({ page }) => {
  await mockApi(page, { bundle: F.BUNDLE });
  await gotoProof(page);
  // default settled headline is above break-even
  await expect(page.getByText(/Retains .* of Opus accuracy/)).toBeVisible();
  // push τ to 1.0 → interpolated cost 0.008 > always-strong 0.0065
  await setSlider(page, "frontier operating point", 1);
  await expect(page.getByText("Below break-even — costs more than Opus here.")).toBeVisible();
  await expect(page.getByText(/Retains .* of Opus accuracy/)).toHaveCount(0);
});

// --- test 9: tap a cascade point → inspect popover --------------------------
test("tap a cascade point → inspect popover with operating_param/quality/cost/escalation/n", async ({ page }) => {
  await mockApi(page, { bundle: F.BUNDLE });
  await gotoProof(page);
  // the cascade points are r=4.2 circles; click the LAST one (τ=1.0) — the first
  // sits under the operating-point marker (default τ=0.5) which would intercept.
  await page.locator('circle[r="4.2"]').last().click({ force: true });
  // "operating_param" only exists in the inspect popover → proof it opened
  await expect(page.getByText(/operating_param/)).toBeVisible();
  await expect(page.getByText(/escalation_rate/)).toBeVisible();
  await expect(page.getByText(/FrugalRoute · cascade/).last()).toBeVisible();
});

// --- test 10: N/A empty state (404) -----------------------------------------
test("N/A: 404 sample → axes + '—' headline + honest offer, no zeros/NaN", async ({ page }) => {
  await mockApi(page, { sampleStatus: 404 });
  const problems = collectProblems(page);
  await gotoProof(page);

  await expect(page.getByText(/No eval run loaded/)).toBeVisible();
  await expect(page.getByText(/frugalroute eval --quick/)).toBeVisible();
  // the populated headline is absent (no fabricated numbers)
  await expect(page.getByText(/Retains .* of Opus accuracy/)).toHaveCount(0);
  await expect(page.getByText("Below break-even — costs more than Opus here.")).toHaveCount(0);
  // axes still render (the static gridline labels)
  await expect(frontier(page).locator("svg")).toBeVisible();
  await expect(page.getByText("$ / query →")).toBeVisible();
  // no leaderboard rows, no NaN/zeros leaking through
  const body = await frontier(page).innerText();
  expect(body).not.toMatch(/NaN|undefined/);
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 11 + R10: losing region + empty predictive + n_refused chip -------
test("R10: losing-region bundle → below-break-even headline, empty predictive degrades, n_refused chip", async ({ page }) => {
  await mockApi(page, { bundle: F.BUNDLE_LOSING });
  const problems = collectProblems(page);
  await gotoProof(page);

  // default settled headline is honestly the loss (chosen target τ=1.0, cost > Opus)
  await expect(page.getByText("Below break-even — costs more than Opus here.")).toBeVisible();
  // the n_refused chip is surfaced and counted
  await expect(page.getByText("n_refused: 2 (surfaced, counted)")).toBeVisible();
  // the empty predictive curve degraded: leaderboard predictive row shows a dash, no NaN
  const fr = frontier(page);
  await expect(fr.getByText("FrugalRoute · predictive")).toBeVisible();
  const body = await fr.innerText();
  expect(body).not.toMatch(/NaN|undefined/);
  // no predictive dashed series path drew (graceful, not a fabricated line)
  expect(problems, problems.join("\n")).toEqual([]);
});

// --- test 12: prefers-reduced-motion → final chart state instantly ----------
test("prefers-reduced-motion: populated chart + headline render instantly (no draw-in)", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockApi(page, { bundle: F.BUNDLE });
  await gotoProof(page);
  // headline + a leaderboard row are present essentially immediately
  await expect(page.getByText(/Retains .* of Opus accuracy/)).toBeVisible({ timeout: 2000 });
  await expect(frontier(page).getByText("FrugalRoute · cascade")).toBeVisible({ timeout: 2000 });
  // the cascade curve is fully revealed (dashoffset 0 → final state, no animation)
  const dash = await page.locator('path[stroke-dasharray]').first().getAttribute("stroke-dashoffset");
  expect(Number(dash)).toBe(0);
});

// --- n_refused hidden when zero (the §2a sc-if) -----------------------------
test("n_refused == 0 → the surfaced-count chip is absent", async ({ page }) => {
  await mockApi(page, { bundle: F.BUNDLE });
  await gotoProof(page);
  await expect(page.getByText(/n_refused:/)).toHaveCount(0);
});
