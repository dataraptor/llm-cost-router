// Playwright config for the FULL-STACK integration pass (split-10 §2): the REAL
// api (uvicorn, native backend, no key) + the app served statically, exercised
// end-to-end against the COMMITTED sample bundle — no /api mocking.
//
// Two web servers boot together: uvicorn on API_PORT and the static app server on
// STATIC_PORT. The app reaches the api cross-origin via window.FRUGALROUTE_API
// (set per-test); the api's CORS defaults to "*", so the browser fetch is allowed.
//
// Browser binary is env-gated like the other e2e (npx playwright install chromium).
// Run from app/:  npx playwright test --config playwright.integration.config.js

import { defineConfig, devices } from "@playwright/test";
import { fileURLToPath } from "node:url";

const STATIC_PORT = 5621;
const API_PORT = 5622;
const REPO_ROOT = fileURLToPath(new URL("../", import.meta.url));

export default defineConfig({
  testDir: "./tests/integration",
  testMatch: /.*\.spec\.js/,
  timeout: 30000,
  expect: { timeout: 8000 },
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${STATIC_PORT}`,
    trace: "off",
  },
  // Exposed to the spec so it can point the app at the real api origin.
  metadata: { apiBase: `http://127.0.0.1:${API_PORT}/api` },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `python -m uvicorn frugalroute_api.app:app --host 127.0.0.1 --port ${API_PORT} --log-level warning`,
      cwd: REPO_ROOT,
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
      env: { FRUGALROUTE_BACKEND: "" }, // native backend → missing-key path is live
    },
    {
      command: `node tests/e2e/static-server.mjs ${STATIC_PORT}`,
      url: `http://localhost:${STATIC_PORT}/FrugalRoute.dc.html`,
      reuseExistingServer: !process.env.CI,
      timeout: 15000,
    },
  ],
});
