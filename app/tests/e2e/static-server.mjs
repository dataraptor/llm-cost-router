// A tiny static file server for the `app/` directory (e2e + manual smoke).
// No /api here — the Playwright tests intercept /api/* with `page.route()`, so
// the app is served same-origin and `config.js` resolves the default "/api".
//
//   node tests/e2e/static-server.mjs [port]   # default 5500

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const APP_DIR = fileURLToPath(new URL("../../", import.meta.url));
const PORT = Number(process.argv[2] || process.env.PORT || 5500);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
};

const server = createServer(async (req, res) => {
  try {
    let pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (pathname === "/") pathname = "/index.html";
    // Contain to APP_DIR (no path traversal).
    const filePath = normalize(join(APP_DIR, pathname));
    if (!filePath.startsWith(normalize(APP_DIR))) {
      res.writeHead(403).end("forbidden");
      return;
    }
    const info = await stat(filePath).catch(() => null);
    if (!info || !info.isFile()) {
      res.writeHead(404).end("not found");
      return;
    }
    const body = await readFile(filePath);
    res.writeHead(200, { "content-type": MIME[extname(filePath)] || "application/octet-stream" });
    res.end(body);
  } catch (err) {
    res.writeHead(500).end(String(err));
  }
});

server.listen(PORT, () => {
  console.log(`[static-server] app/ on http://localhost:${PORT}/`);
});
