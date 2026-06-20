// Forward "/" → the dc-runtime page, preserving any query string / hash so deep
// links survive. Lives in an external file (not inline) to satisfy the strict CSP
// (`script-src 'self'`, no 'unsafe-inline'); the <meta http-equiv="refresh"> in
// index.html is the no-JS fallback. See app/nginx.conf + README "Deploy".
location.replace("./FrugalRoute.dc.html" + location.search + location.hash);
