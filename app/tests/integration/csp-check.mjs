// CSP compatibility check (split 14). Loads the served page (default: the built
// app container) in headless chromium and asserts the Content-Security-Policy
// does NOT block the dc-runtime: no `securitypolicyviolation` events, React mounts
// (the header renders), and no CSP console errors. The API may be down — the boot
// fetch failing is fine; we only assert CSP didn't break script-eval/style/fonts.
//
// Usage: BASE=http://localhost:8099 node tests/integration/csp-check.mjs
import { chromium } from "@playwright/test";

const BASE = process.env.BASE || "http://localhost:8099";
const URL = `${BASE}/FrugalRoute.dc.html`;

const browser = await chromium.launch();
const page = await browser.newPage();

const violations = [];
const cspConsole = [];
await page.addInitScript(() => {
  window.addEventListener("securitypolicyviolation", (e) => {
    (window.__csp = window.__csp || []).push(
      `${e.violatedDirective} blocked ${e.blockedURI}`,
    );
  });
});
page.on("console", (msg) => {
  const t = msg.text();
  if (/content security policy|csp/i.test(t)) cspConsole.push(t);
});

await page.goto(URL, { waitUntil: "domcontentloaded" });
// Give React + the dc-runtime time to mount and eval the data-dc-script.
await page.waitForTimeout(1500);

// The brand header ("FrugalRoute") only renders if React mounted under the CSP.
const mounted = await page.getByText("FrugalRoute", { exact: false }).first().isVisible().catch(() => false);
const reported = await page.evaluate(() => window.__csp || []);
violations.push(...reported);

await browser.close();

const problems = [];
if (!mounted) problems.push("React did not mount (CSP may have blocked script-eval)");
if (violations.length) problems.push("CSP violations: " + violations.join("; "));
if (cspConsole.length) problems.push("CSP console: " + cspConsole.join("; "));

if (problems.length) {
  console.error("CSP CHECK FAILED:\n  " + problems.join("\n  "));
  process.exit(1);
}
console.log("CSP CHECK PASSED — app renders under CSP; 0 violations.");
