// Performance budget check (split 14). Measures the app's static asset weight and
// its render timing against documented budgets, deterministically and no-key (the
// app is served by the e2e static server; /api is mocked from the fixtures).
//
//   cd app && node tests/e2e/perf.mjs
//
// Budgets (recorded in ACCEPTANCE.md). Generous but real — they catch a regression
// like accidentally shipping an un-minified bundle or a multi-MB asset. Requires the
// chromium binary (npx playwright install chromium).

import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { readFile, stat } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import * as F from "./fixtures.js";

const PORT = 5710;
const HERE = fileURLToPath(new URL(".", import.meta.url));
const APP = fileURLToPath(new URL("../../", import.meta.url));

// --- budgets ---------------------------------------------------------------
const BUDGET = {
  rawJsKB: 600, // vendored React + ReactDOM + support.js + src/*.js, un-gzipped
  gzipJsKB: 220, // the same, gzipped (what the wire actually carries)
  htmlKB: 120, // the dc.html document
  firstRenderMs: 4000, // nav start → the brand header is visible (headless CI)
  frontierDrawMs: 4000, // Proof click → headline visible
};

const JS_ASSETS = [
  "vendor/react.production.min.js",
  "vendor/react-dom.production.min.js",
  "support.js",
  "src/config.js",
  "src/api.js",
  "src/format.js",
  "src/bridge.js",
];

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = [];
const rows = [];

function check(name, value, unit, budget) {
  const ok = value <= budget;
  rows.push(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(22)} ${String(value).padStart(8)} ${unit} (budget ${budget})`);
  if (!ok) fail.push(`${name}: ${value}${unit} > ${budget}${unit}`);
}

// --- 1. asset weight (from disk) -------------------------------------------
let rawJs = 0;
let gzJs = 0;
for (const rel of JS_ASSETS) {
  const buf = await readFile(APP + rel);
  rawJs += buf.length;
  gzJs += gzipSync(buf).length;
}
const htmlBytes = (await stat(APP + "FrugalRoute.dc.html")).size;
check("raw JS payload", Math.round(rawJs / 1024), "KB", BUDGET.rawJsKB);
check("gzipped JS payload", Math.round(gzJs / 1024), "KB", BUDGET.gzipJsKB);
check("dc.html document", Math.round(htmlBytes / 1024), "KB", BUDGET.htmlKB);

// --- 2. render timing (Playwright + mocked API) ----------------------------
const server = spawn("node", [HERE + "static-server.mjs", String(PORT)], { stdio: "ignore" });
try {
  await wait(800);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
  await page.route("**/api/eval/sample", (r) => r.fulfill({ json: F.BUNDLE }));

  const t0 = Date.now();
  await page.goto(`http://localhost:${PORT}/FrugalRoute.dc.html`, { waitUntil: "domcontentloaded" });
  await page.getByText("FrugalRoute", { exact: false }).first().waitFor({ state: "visible" });
  const firstRender = Date.now() - t0;
  check("first render", firstRender, "ms", BUDGET.firstRenderMs);

  const t1 = Date.now();
  await page.getByRole("button", { name: "Proof", exact: true }).click();
  await page.getByText(/Retains .* of Opus accuracy at .* of the cost/).waitFor({ state: "visible" });
  const frontierDraw = Date.now() - t1;
  check("frontier draw", frontierDraw, "ms", BUDGET.frontierDrawMs);

  await browser.close();
} finally {
  server.kill();
}

// --- report ----------------------------------------------------------------
console.log("\nFrugalRoute performance budgets\n" + "-".repeat(56));
console.log(rows.join("\n"));
console.log("-".repeat(56));
if (fail.length) {
  console.error("PERF CHECK FAILED:\n  " + fail.join("\n  "));
  process.exit(1);
}
console.log("PERF CHECK PASSED — all budgets met.");
