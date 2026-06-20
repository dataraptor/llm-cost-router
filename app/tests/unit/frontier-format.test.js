// Unit tests for the Frontier mapping helpers (split-08 tests 1–5 + coverage).
// Pure functions over an /api/eval/sample-shaped bundle — `node --test`, no browser.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  toCurve,
  toBaselines,
  toOracle,
  toLeaderboardRows,
  toHeadline,
  interpCurve,
  chosenPoint,
  frontierXMax,
  pickReport,
  spreadStr,
  pctSpreadStr,
  frozenSplitNote,
  provChips,
} from "../../src/format.js";

// A clean §7-shaped bundle. always_strong = {q:1.0, cost:0.0065}; the cascade's
// lowest-cost ≥95%-retention point is τ=0.5 (cost 0.0039 = 60% of strong → 40% cut).
function makeBundle(over = {}) {
  const baselines = {
    always_cheap: { quality: 0.6, quality_spread: 0.0, cost: 0.0009, cost_spread: 2e-5 },
    always_strong: { quality: 1.0, quality_spread: 0.0, cost: 0.0065, cost_spread: 1.5e-4 },
    random: { quality: 0.75, quality_spread: 0.0, cost: 0.0037, cost_spread: 9e-5 },
  };
  const oracle = { quality: 1.0, quality_spread: 0.0, cost: 0.003 };
  const cascade = {
    strategy: "cascade",
    points: [
      { operating_param: 0.5, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.0039, cost_spread: 1e-4, escalation_rate: 0.375, n: 8 },
      { operating_param: 0.8, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.00475, cost_spread: 1.2e-4, escalation_rate: 0.5, n: 8 },
      { operating_param: 1.0, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.008, cost_spread: 2e-4, escalation_rate: 1.0, n: 8 },
    ],
    baselines,
    oracle,
    retention_at_target: 1.0,
    retention_at_target_spread: 0.012,
    cost_reduction_at_target: 0.4, // 1 - 0.0039/0.0065
    cost_reduction_at_target_spread: 0.0,
    n_refused: 0,
    prompt_version: "v1",
    model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
    n_runs: 3,
  };
  const predictive = {
    strategy: "predictive",
    points: [
      { operating_param: 0.4, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.0037, cost_spread: 9e-5, escalation_rate: 0.5, n: 8 },
      { operating_param: 0.6, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.003, cost_spread: 7e-5, escalation_rate: 0.375, n: 8 },
    ],
    baselines,
    oracle,
    retention_at_target: 1.0,
    retention_at_target_spread: 0.0,
    cost_reduction_at_target: 0.5384615384615385, // 1 - 0.003/0.0065
    cost_reduction_at_target_spread: 0.0,
    n_refused: 0,
    prompt_version: "v1",
    model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
    n_runs: 3,
  };
  return {
    reports: [cascade, predictive],
    benchmark: "gsm8k",
    frozen_split: { n_test: 8, n_calibration: 32, small_n: true },
    generated_at: "2026-06-20T00:00:00+00:00",
    ...over,
  };
}

// --- test 1: toCurve field-for-field, order preserved ----------------------
test("toCurve maps FrontierPoint → {param,q,cost,esc,n} field-for-field, order preserved (test 1)", () => {
  const cascade = pickReport(makeBundle(), "cascade");
  const curve = toCurve(cascade);
  assert.equal(curve.length, 3);
  assert.deepEqual(
    curve.map((p) => p.param),
    [0.5, 0.8, 1.0],
  );
  assert.deepEqual(curve[0], { param: 0.5, q: 1.0, qSpread: 0.0, cost: 0.0039, costSpread: 1e-4, esc: 0.375, n: 8 });
  assert.equal(curve[2].cost, 0.008);
  assert.equal(curve[2].esc, 1.0);
  // robust to a missing/empty report
  assert.deepEqual(toCurve(null), []);
  assert.deepEqual(toCurve({ points: "nope" }), []);
});

// --- test 2: leaderboard — 6 fixed-order rows, ours flags, oracle "—" -------
test("toLeaderboardRows → 6 fixed-order rows; ours flagged; oracle '—'; numbers match (test 2)", () => {
  const bundle = makeBundle();
  const rows = toLeaderboardRows(bundle);
  assert.equal(rows.length, 6);
  assert.deepEqual(
    rows.map((r) => r.name),
    ["always-cheap", "always-strong", "random @ cost", "FrugalRoute · cascade", "FrugalRoute · predictive", "oracle (ceiling)"],
  );
  // ours = only the two FrugalRoute rows
  assert.deepEqual(rows.map((r) => r.ours), [false, false, false, true, true, false]);
  assert.deepEqual(rows.map((r) => r.nameColor), [
    "var(--ink-900)", "var(--ink-900)", "var(--ink-900)", "var(--accent)", "var(--accent)", "var(--ink-900)",
  ]);

  // baseline numbers equal the report's baselines (vs always-strong)
  const cheap = rows[0];
  assert.equal(cheap.q, "0.60");
  assert.equal(cheap.cost, "0.0009");
  assert.equal(cheap.ret, "60%"); // 0.6/1.0
  assert.equal(cheap.cut, "86%"); // 1 - 0.0009/0.0065
  const strong = rows[1];
  assert.equal(strong.ret, "100%");
  assert.equal(strong.cut, "0%");
  const random = rows[2];
  assert.equal(random.ret, "75%");

  // our cascade row uses the report's headline retention/cost-reduction + chosen point
  const cas = rows[3];
  assert.equal(cas.q, "1.00");
  assert.equal(cas.cost, "0.0039"); // the τ=0.5 chosen point
  assert.equal(cas.ret, "100%"); // retention_at_target
  assert.equal(cas.cut, "40%"); // cost_reduction_at_target
  assert.equal(cas.note, "τ=0.50");
  const pred = rows[4];
  assert.equal(pred.note, "θ=0.60");
  assert.equal(pred.cut, "54%");

  // oracle retention/cut are a dash (a ceiling, not a target)
  const oracle = rows[5];
  assert.equal(oracle.ret, "—");
  assert.equal(oracle.cut, "—");
  assert.equal(oracle.q, "1.00");
  assert.equal(oracle.cost, "0.0030");

  // empty / 404 bundle → no rows (N/A)
  assert.deepEqual(toLeaderboardRows(null), []);
  assert.deepEqual(toLeaderboardRows({ reports: [] }), []);
});

// --- test 3: headline matches the report's at-target at the chosen point ----
test("toHeadline at the chosen τ matches retention_at_target / cost_reduction_at_target (test 3)", () => {
  const report = pickReport(makeBundle(), "cascade");
  const cp = chosenPoint(report);
  assert.equal(cp.param, 0.5); // lowest-cost point ≥95% retention

  const head = toHeadline(report, cp.param);
  assert.equal(head.hasData, true);
  assert.equal(head.belowBE, false);
  // retention matches retention_at_target
  assert.equal(head.retPctStr, Math.round(report.retention_at_target * 100) + "%");
  // cost-reduction (1 − cost-fraction) matches cost_reduction_at_target
  const costFrac = parseInt(head.costPctStr, 10);
  assert.equal(100 - costFrac, Math.round(report.cost_reduction_at_target * 100));
  // empty report → honest "—", never a fake zero
  const na = toHeadline(null, 0.8);
  assert.deepEqual(na, { retPctStr: "—", costPctStr: "—", belowBE: false, hasData: false });
});

// --- test 4: below break-even → belowBE true --------------------------------
test("toHeadline flips belowBE when interpolated cost exceeds always-strong (test 4)", () => {
  const report = pickReport(makeBundle(), "cascade");
  // τ=1.0 → cost 0.008 > always-strong 0.0065
  const high = toHeadline(report, 1.0);
  assert.equal(high.belowBE, true);
  // τ=0.5 → cost 0.0039 < 0.0065
  const low = toHeadline(report, 0.5);
  assert.equal(low.belowBE, false);
  // interpolation glides between points (a mid τ is between the endpoints' costs)
  const mid = interpCurve(toCurve(report), 0.65);
  assert.ok(mid.cost > 0.0039 && mid.cost < 0.00475);
});

// --- test 5: PLOT.xmax derives from data ------------------------------------
test("frontierXMax derives from data — larger costs widen the axis; stays finite (test 5)", () => {
  const small = frontierXMax([{ cost: 0.001 }, { cost: 0.004 }]);
  const large = frontierXMax([{ cost: 0.001 }, { cost: 0.02 }]);
  assert.ok(large > small, "a larger max cost must widen xmax");
  assert.ok(Number.isFinite(small) && small > 0);
  assert.ok(Math.abs(large - 0.02 * 1.07) < 1e-12); // max × 1.07 headroom
  // empty / degenerate input → a positive floor so the axis ticks stay valid
  assert.ok(frontierXMax([]) >= 0.0086);
  assert.ok(frontierXMax(null) >= 0.0086);
  // sx-style scaling stays finite with the derived xmax
  const xmax = frontierXMax([{ cost: 0.008 }]);
  const sx = (c) => 64 + (c / xmax) * 560;
  assert.ok(Number.isFinite(sx(0.008)) && sx(0.008) <= 64 + 560);
});

// --- baselines / oracle / random polyline -----------------------------------
test("toBaselines builds the cheap/strong points + the cheap→random→strong polyline", () => {
  const report = pickReport(makeBundle(), "cascade");
  const b = toBaselines(report);
  assert.deepEqual(b.cheap, { q: 0.6, cost: 0.0009, qSpread: 0.0 });
  assert.deepEqual(b.strong, { q: 1.0, cost: 0.0065, qSpread: 0.0 });
  assert.equal(b.random.length, 3);
  assert.deepEqual(b.random[0], b.cheap);
  assert.deepEqual(b.random[2], b.strong);
  assert.equal(b.random[1].cost, 0.0037); // the random midpoint
  const o = toOracle(report);
  assert.deepEqual(o, { q: 1.0, cost: 0.003, qSpread: 0.0 });
});

// --- chosenPoint replicates the engine's target selection -------------------
test("chosenPoint: lowest-cost ≥target retention, else highest retention", () => {
  // all qualify → lowest cost wins (τ=0.5)
  const report = pickReport(makeBundle(), "cascade");
  assert.equal(chosenPoint(report).param, 0.5);

  // none qualify → highest retention wins
  const losing = {
    strategy: "cascade",
    baselines: { always_strong: { quality: 1.0, cost: 0.0065 } },
    points: [
      { operating_param: 0.5, quality: 0.5, cost_usd_per_query: 0.002 },
      { operating_param: 1.0, quality: 0.8, cost_usd_per_query: 0.006 },
    ],
  };
  assert.equal(chosenPoint(losing).param, 1.0); // 0.8 > 0.5 retention
  assert.equal(chosenPoint({ points: [] }), null);
});

// --- spread + caption formatters --------------------------------------------
test("spreadStr / pctSpreadStr / frozenSplitNote format the load-bearing ± honestly", () => {
  assert.equal(spreadStr(0.02), "±.02");
  assert.equal(spreadStr(0.0), "±.00");
  assert.equal(spreadStr(0.144), "±.14");
  assert.equal(pctSpreadStr(0.012), "±1.2%");
  assert.equal(pctSpreadStr(0.0), "±0.0%");

  assert.deepEqual(frozenSplitNote(makeBundle()), { n: "8", wideCI: " · wide CI" });
  assert.deepEqual(frozenSplitNote(makeBundle({ frozen_split: { n_test: 120, small_n: false } })), {
    n: "120",
    wideCI: "",
  });
  assert.deepEqual(frozenSplitNote(null), { n: "", wideCI: "" });
});

// --- provenance n_runs chip (Frontier) --------------------------------------
test("provChips appends the n_runs chip when a report is given (split-07 callers unchanged)", () => {
  const config = {
    prompt_version: "v1",
    model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
    pricing_pinned_date: "2026-06-19",
  };
  // single-query callers pass only config → still three chips
  assert.equal(provChips(config).length, 3);
  // frontier passes the bundle's report → adds n_runs
  const report = pickReport(makeBundle(), "cascade");
  const chips = provChips(config, report);
  assert.equal(chips.length, 4);
  assert.equal(chips[3], "n_runs: 3");
});

// --- adversarial (R10): a losing cascade + an empty predictive curve --------
test("R10: losing-region cascade draws; empty predictive curve degrades (no crash)", () => {
  const losingCascade = {
    strategy: "cascade",
    points: [
      { operating_param: 0.5, quality: 0.9, quality_spread: 0.0, cost_usd_per_query: 0.004, cost_spread: 0, escalation_rate: 0.4, n: 8 },
      { operating_param: 1.0, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.009, cost_spread: 0, escalation_rate: 1.0, n: 8 }, // above strong cost
    ],
    baselines: {
      always_cheap: { quality: 0.6, quality_spread: 0, cost: 0.001, cost_spread: 0 },
      always_strong: { quality: 1.0, quality_spread: 0, cost: 0.007, cost_spread: 0 },
      random: { quality: 0.7, quality_spread: 0, cost: 0.004, cost_spread: 0 },
    },
    oracle: { quality: 1.0, quality_spread: 0, cost: 0.003 },
    retention_at_target: 1.0,
    retention_at_target_spread: 0.0,
    cost_reduction_at_target: 1 - 0.009 / 0.007, // negative — costs MORE than Opus
    cost_reduction_at_target_spread: 0.0,
    n_refused: 2,
    prompt_version: "v1",
    model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
    n_runs: 3,
  };
  const emptyPredictive = { ...losingCascade, strategy: "predictive", points: [], retention_at_target: null, cost_reduction_at_target: null };
  const bundle = { reports: [losingCascade, emptyPredictive], benchmark: "gsm8k", frozen_split: { n_test: 8, small_n: true } };

  // the losing region is honestly surfaced at τ=1.0
  assert.equal(toHeadline(losingCascade, 1.0).belowBE, true);

  // the empty predictive curve degrades to no line + a dashed leaderboard row
  assert.deepEqual(toCurve(emptyPredictive), []);
  const rows = toLeaderboardRows(bundle);
  assert.equal(rows.length, 6);
  const predRow = rows[4];
  assert.equal(predRow.name, "FrugalRoute · predictive");
  assert.equal(predRow.q, "—"); // no chosen point
  assert.equal(predRow.ret, "—"); // null retention → honest dash, no NaN
  assert.ok(!JSON.stringify(rows).includes("NaN"));
  // the cascade row still renders its real (negative) headline cut
  assert.match(rows[3].cut, /-?\d+%/);
});
