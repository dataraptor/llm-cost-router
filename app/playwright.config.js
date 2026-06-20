// Playwright e2e config (split-07). Serves app/ via the tiny static server and
// drives the Single-Query view; /api/* is mocked per-test with page.route().
//
// The browser binary is environment-gated: run `npx playwright install chromium`
// once. If the binary is absent the suite cannot launch (documented like the
// prior splits' env-gated live smokes); the node --test unit suite is the gate.

import { defineConfig, devices } from "@playwright/test";

const PORT = 5599;

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: /.*\.spec\.js/,
  timeout: 20000,
  expect: { timeout: 5000 },
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "off",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `node tests/e2e/static-server.mjs ${PORT}`,
    url: `http://localhost:${PORT}/FrugalRoute.dc.html`,
    reuseExistingServer: !process.env.CI,
    timeout: 15000,
  },
});
