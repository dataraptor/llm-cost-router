// bridge.js — expose the ES modules to the dc-runtime logic on `window.FR`.
//
// The `<script data-dc-script>` in FrugalRoute.dc.html is evaluated by the
// dc-runtime via `new Function(...)`, so it cannot `import`. This module is loaded
// by the page (`<script type="module">`) and publishes the (statically-imported,
// fully-initialized) client + formatters as a global the logic reads at call time.
//
// It executes before the dc-runtime mounts the component (a local module resolves
// well before the React UMD bundle loads from the CDN), so `window.FR` is present
// by first render. The logic still guards defensively.

import * as api from "./api.js";
import * as format from "./format.js";
import { apiBaseUrl } from "./config.js";

window.FR = {
  // API client
  getConfig: api.getConfig,
  getExamples: api.getExamples,
  postRoute: api.postRoute,
  routeStreamUrl: api.routeStreamUrl,
  getEvalSample: api.getEvalSample,
  postEval: api.postEval,
  ApiError: api.ApiError,
  apiBaseUrl,
  // pure format/mapping helpers (namespaced so the logic reads `FR.format.*`)
  format,
};
