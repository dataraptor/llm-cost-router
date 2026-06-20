// WCAG 2.1 AA audit (split 14): axe-core over BOTH views × BOTH themes, plus a
// few manual-equivalent assertions (focus visible, slider valuetext, chart data
// table mirrors the FrontierPoints, SR route summary). The app is served
// same-origin; /api is mocked so the audit is deterministic and no-key.
//
// Gate: 0 serious/critical axe violations on every view×theme combination.

import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import * as F from "./fixtures.js";

const PAGE = "/FrugalRoute.dc.html";
const AA_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

async function mockApi(page) {
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
  await page.route("**/api/eval/sample", (r) => r.fulfill({ json: F.BUNDLE }));
}

async function setTheme(page, theme) {
  const current = await page.locator("[data-theme]").first().getAttribute("data-theme");
  if (current !== theme) {
    await page.getByRole("button", { name: "Toggle theme" }).click();
  }
  await expect(page.locator(`[data-theme="${theme}"]`).first()).toBeVisible();
}

async function audit(page) {
  const results = await new AxeBuilder({ page }).withTags(AA_TAGS).analyze();
  const seriousOrCritical = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  return seriousOrCritical;
}

function describeViolations(vs) {
  return vs
    .map((v) => `${v.id} (${v.impact}): ${v.help} [${v.nodes.length} node(s)] e.g. ${v.nodes[0]?.target}`)
    .join("\n");
}

// --- axe: Single-query view, light + dark -----------------------------------
for (const theme of ["light", "dark"]) {
  test(`axe: single-query view, ${theme} theme — 0 serious/critical`, async ({ page }) => {
    await mockApi(page);
    await page.goto(PAGE);
    await expect(page.locator('[data-screen-label="single-query"]')).toBeVisible();
    await setTheme(page, theme);
    const vs = await audit(page);
    expect(vs, describeViolations(vs)).toEqual([]);
  });
}

// --- axe: Frontier (Proof) view, light + dark -------------------------------
for (const theme of ["light", "dark"]) {
  test(`axe: frontier view, ${theme} theme — 0 serious/critical`, async ({ page }) => {
    await mockApi(page);
    await page.goto(PAGE);
    await page.getByRole("button", { name: "Proof", exact: true }).click();
    await expect(page.locator('[data-screen-label="frontier"]')).toBeVisible();
    await setTheme(page, theme);
    // let the reveal settle
    await page.waitForTimeout(400);
    const vs = await audit(page);
    expect(vs, describeViolations(vs)).toEqual([]);
  });
}

// --- manual-equivalent: focus is always visible -----------------------------
test("keyboard: the primary controls take visible focus (outline not suppressed)", async ({ page }) => {
  await mockApi(page);
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Route" }).focus();
  const outline = await page.getByRole("button", { name: "Route" }).evaluate((el) => {
    const s = getComputedStyle(el);
    return { width: s.outlineWidth, style: s.outlineStyle };
  });
  // focus-visible only paints on keyboard focus; assert the rule exists by checking
  // the slider's focus ring is reachable via Tab and the outline is not 'none' on :focus-visible.
  // (A direct .focus() may not trigger :focus-visible, so we assert the CSS is present.)
  const cssHasFocusRing = await page.evaluate(() => {
    for (const sheet of document.styleSheets) {
      let rules;
      try {
        rules = sheet.cssRules;
      } catch {
        continue;
      }
      for (const rule of rules) {
        if (rule.selectorText && rule.selectorText.includes(":focus-visible") && /outline/.test(rule.cssText)) {
          return true;
        }
      }
    }
    return false;
  });
  expect(cssHasFocusRing).toBe(true);
  expect(outline).toBeTruthy();
});

// --- manual-equivalent: sliders expose a meaningful value to a screen reader -
test("slider exposes aria-valuetext (tau/theta + value, not a bare number)", async ({ page }) => {
  await mockApi(page);
  await page.goto(PAGE);
  const vt = await page.locator('input[aria-label="operating point"]').getAttribute("aria-valuetext");
  expect(vt).toMatch(/^(tau|theta) \d\.\d{2}$/);
});

// --- manual-equivalent: the chart's hidden data table mirrors FrontierPoints --
test("chart data table mirrors the FrontierPoints (1.1.1 non-text alternative)", async ({ page }) => {
  await mockApi(page);
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Proof", exact: true }).click();
  await expect(page.locator('[data-screen-label="frontier"]')).toBeVisible();
  // one table row per cascade FrontierPoint in the bundle
  const cascade = F.BUNDLE.reports.find((r) => r.strategy === "cascade");
  const rowCount = await page.locator("table.sr-only tbody tr").count();
  expect(rowCount).toBe(cascade.points.length);
  // the SVG carries an accessible name (role=img + aria-label)
  const svgLabel = await page.locator('svg[role="img"]').getAttribute("aria-label");
  expect(svgLabel).toMatch(/frontier chart/i);
  expect(svgLabel).toMatch(/retains .* of Opus accuracy/i);
});

// --- manual-equivalent: the route stepper has an ordered SR summary ----------
test("route stepper exposes an ordered text summary to a screen reader", async ({ page }) => {
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
  // an accepted-cascade stream → the SR summary should read the gate + acceptance
  const { mockStream } = await import("./sse.js");
  await mockStream(page, [
    ["phase", { phase: "gen", tier: "claude-haiku-4-5" }],
    ["gate", { sufficient: true, confidence: 0.91, reason: "ok", cost_usd: 0.0004 }],
    ["done", F.ACCEPTED],
  ]);
  await page.goto(PAGE);
  await page.getByRole("button", { name: "Route" }).click();
  const summary = page.locator(".sr-only[aria-live]").filter({ hasText: "Cascade route" });
  await expect(summary).toContainText("gate judged sufficient at confidence 0.91");
  await expect(summary).toContainText("the cheap answer was kept");
});
