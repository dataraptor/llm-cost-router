// Capture the Frontier (Proof) view rendering the *committed* sample bundle to
// docs/frontier.png — the README's money shot (split-10 §3). Deterministic and
// no-key: the app is served by the e2e static server and /api is fulfilled from
// the committed sample_run.json on disk (the same artifact /api/eval/sample
// serves), so the screenshot is exactly what a stranger sees.
//
//   cd app && node tests/e2e/screenshot.mjs
//
// Requires the chromium binary (npx playwright install chromium).

import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import * as F from "./fixtures.js";

const PORT = 5610;
const HERE = fileURLToPath(new URL(".", import.meta.url));
const REPO = fileURLToPath(new URL("../../../", import.meta.url));
const SAMPLE = REPO + "api/src/frugalroute_api/data/sample_run.json";
const OUT = REPO + "docs/frontier.png";

const server = spawn("node", [HERE + "static-server.mjs", String(PORT)], { stdio: "ignore" });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

try {
  await wait(800); // let the static server bind
  const bundle = JSON.parse(await readFile(SAMPLE, "utf-8"));

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
  await page.route("**/api/config", (r) => r.fulfill({ json: F.CONFIG }));
  await page.route("**/api/examples", (r) => r.fulfill({ json: F.EXAMPLES }));
  await page.route("**/api/eval/sample", (r) => r.fulfill({ json: bundle }));

  await page.goto(`http://localhost:${PORT}/FrugalRoute.dc.html`);
  await page.getByRole("button", { name: "Proof", exact: true }).click();
  const frontier = page.locator('[data-screen-label="frontier"]');
  await frontier.waitFor({ state: "visible" });
  await page.getByText(/Retains .* of Opus accuracy at .* of the cost/).waitFor();
  await wait(1400); // let the §6 draw-in choreography settle

  await frontier.screenshot({ path: OUT });
  console.log("wrote", OUT);
  await browser.close();
} finally {
  server.kill();
}
