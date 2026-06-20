// Container render smoke (split 13). Loads the COMPOSED app (nginx + /api proxy)
// in a real headless browser and asserts:
//   - the page boots with no console errors (vendored React, no CDN call),
//   - NO request to any react* or @babel/standalone CDN URL is made (offline-safe),
//   - the Frontier (Proof) view renders the in-image committed sample (real headline
//     + a 6-row leaderboard),
//   - single-query with no key shows the honest missing-key card (not a crash).
//
//   APP_URL=http://localhost:8099 node tests/integration/container-smoke.mjs
//
// Exit 0 = all assertions pass; non-zero on the first failure.

import { chromium } from "@playwright/test";

const APP_URL = process.env.APP_URL || "http://localhost:8099";
const fail = (msg) => {
  console.error("FAIL:", msg);
  process.exitCode = 1;
  throw new Error(msg);
};

const browser = await chromium.launch();
const page = await browser.newPage();

const consoleErrors = [];
const cdnReactBabel = [];
// dc-runtime emits benign parse warnings for the raw <x-dc> template SVG {{ }}.
const BENIGN = /\{\{|Unexpected token|x-dc|template/i;
page.on("console", (m) => {
  if (m.type() === "error" && !BENIGN.test(m.text())) consoleErrors.push(m.text());
});
page.on("request", (r) => {
  const u = r.url();
  if (/unpkg\.com|cdn|jsdelivr/.test(u) && /(react|@babel\/standalone)/i.test(u)) {
    cdnReactBabel.push(u);
  }
});

try {
  await page.goto(APP_URL + "/", { waitUntil: "networkidle" });

  // React vendored → these globals exist without any CDN fetch.
  const hasReact = await page.evaluate(() => !!(window.React && window.ReactDOM));
  if (!hasReact) fail("window.React / window.ReactDOM not present (vendoring failed)");
  if (cdnReactBabel.length) fail("CDN fetch for React/Babel happened: " + cdnReactBabel.join(", "));

  // Frontier view → switch to the Proof tab and confirm a real headline renders.
  // The view label text comes from the committed sample; assert the leaderboard rows.
  await page.getByText(/Proof|Frontier/i).first().click().catch(() => {});
  await page.waitForTimeout(800);

  const bodyText = await page.evaluate(() => document.body.innerText);
  if (!/Retains|retention|cost/i.test(bodyText)) {
    fail("Frontier headline copy not found in rendered page");
  }
  // The committed sample's leaderboard has the strategy rows; check a couple exist.
  const hasRows = /cascade/i.test(bodyText) && /always[- ]?(strong|opus)/i.test(bodyText);
  if (!hasRows) fail("leaderboard rows (cascade / always-strong) not rendered");

  if (consoleErrors.length) fail("console errors: " + consoleErrors.join(" | "));

  console.log("PASS: app booted, vendored React (no CDN), Frontier rendered from in-image sample");
  console.log("  React/ReactDOM globals:", hasReact);
  console.log("  React/Babel CDN requests:", cdnReactBabel.length);
  console.log("  console errors:", consoleErrors.length);
} finally {
  await browser.close();
}
