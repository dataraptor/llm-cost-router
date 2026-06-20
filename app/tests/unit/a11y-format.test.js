// Unit tests for the split-14 accessibility text helpers (pure, no browser).
// Runner: `node --test`. These back the SR live regions, the chart data table,
// and the slider aria-valuetext — assert the strings are accurate and honest.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  routeStepperSummary,
  savedAnnouncement,
  sliderValueText,
  chartSummary,
  chartDataRows,
  savingsBand,
} from "../../src/format.js";

test("routeStepperSummary: idle → no route", () => {
  assert.equal(routeStepperSummary({ strategy: "cascade", phase: "idle" }), "No route yet.");
});

test("routeStepperSummary: cascade accepted reads the gate verdict + acceptance", () => {
  const s = routeStepperSummary({
    strategy: "cascade",
    phase: "done",
    gate: { sufficient: true, confidence: 0.92 },
    result: { escalated: false, tierName: "Haiku 4.5" },
  });
  assert.match(s, /Cascade route/);
  assert.match(s, /gate judged sufficient at confidence 0\.92/);
  assert.match(s, /the cheap answer was kept \(tier Haiku 4\.5\)/);
});

test("routeStepperSummary: cascade escalated says escalated, not accepted", () => {
  const s = routeStepperSummary({
    strategy: "cascade",
    phase: "done",
    gate: { sufficient: false, confidence: 0.4 },
    result: { escalated: true, tierName: "Opus 4.8" },
  });
  assert.match(s, /gate judged insufficient at confidence 0\.40/);
  assert.match(s, /escalated to Opus 4\.8/);
  assert.doesNotMatch(s, /accepted/);
});

test("routeStepperSummary: cheap refusal is surfaced", () => {
  const s = routeStepperSummary({
    strategy: "cascade",
    phase: "done",
    gate: null,
    result: { escalated: true, tierName: "Opus 4.8", cheapRefusedEscalated: true },
  });
  assert.match(s, /cheap tier refused/);
});

test("routeStepperSummary: predictive has no gate, reads P(strong)", () => {
  const s = routeStepperSummary({
    strategy: "predictive",
    phase: "done",
    result: { escalated: true, tierName: "Opus 4.8", pStrong: 0.73 },
  });
  assert.match(s, /Predictive route/);
  assert.match(s, /P\(strong\)=0\.73/);
  assert.doesNotMatch(s, /gate/);
});

test("savedAnnouncement: positive savings says 'saved' + up%, no raw symbols", () => {
  const band = savingsBand(0.02, 0.02, 4, 0.007);
  const s = savedAnnouncement(band, 4);
  assert.match(s, /saved \$0\.0200 over 4 runs/);
  assert.match(s, /up 71%|up \d+%/);
  assert.doesNotMatch(s, /[▲▼−]/); // spoken words, not glyphs
});

test("savedAnnouncement: a loss says 'spent' + down% (honest, never hidden)", () => {
  const band = savingsBand(-0.01, 0.01, 2, 0.007);
  const s = savedAnnouncement(band, 2);
  assert.match(s, /spent \$0\.0100/);
  assert.match(s, /down/);
});

test("savedAnnouncement: singular run grammar", () => {
  const band = savingsBand(0.005, 0.005, 1, 0.007);
  assert.match(savedAnnouncement(band, 1), /over 1 run\b/);
});

test("sliderValueText: tau vs theta spelled out", () => {
  assert.equal(sliderValueText("τ", 0.8), "tau 0.80");
  assert.equal(sliderValueText("θ", 0.6), "theta 0.60");
});

test("chartSummary: N/A state names that no data is loaded", () => {
  const s = chartSummary(null, { hasData: false });
  assert.match(s, /No eval run is loaded/);
});

test("chartSummary: populated reads retention + cost fraction", () => {
  const s = chartSummary({}, { hasData: true, retPctStr: "100%", costPctStr: "60%", belowBE: false });
  assert.match(s, /retains 100% of Opus accuracy at 60% of the cost/);
  assert.match(s, /data table follows/);
  assert.doesNotMatch(s, /below break-even/);
});

test("chartSummary: losing region is named honestly", () => {
  const s = chartSummary({}, { hasData: true, retPctStr: "100%", costPctStr: "120%", belowBE: true });
  assert.match(s, /below break-even/);
  assert.match(s, /costs more than always using Opus/);
});

test("chartDataRows: mirrors FrontierPoints as plain strings (no NaN)", () => {
  const report = {
    points: [
      { operating_param: 0.5, quality: 0.93, cost_usd_per_query: 0.0012, escalation_rate: 0.1, n: 84 },
      { operating_param: 1.0, quality: 0.98, cost_usd_per_query: 0.008, escalation_rate: 1.0, n: 84 },
    ],
  };
  const rows = chartDataRows(report);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0], { param: "0.50", q: "0.930", cost: "$0.0012", esc: "10%", n: "84" });
  assert.deepEqual(rows[1], { param: "1.00", q: "0.980", cost: "$0.0080", esc: "100%", n: "84" });
  for (const r of rows) for (const v of Object.values(r)) assert.doesNotMatch(v, /NaN|undefined/);
});

test("chartDataRows: empty report → empty rows (N/A, never a fake row)", () => {
  assert.deepEqual(chartDataRows(null), []);
  assert.deepEqual(chartDataRows({ points: [] }), []);
});
