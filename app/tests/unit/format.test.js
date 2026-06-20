// Unit tests for the pure format/mapping helpers (split-07 tests 1–4 + coverage).
// Runner: `node --test` (no browser, no network).

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  num,
  safe4,
  money,
  latency,
  clamp01,
  accumulateSavings,
  savingsBand,
  tierName,
  isStrongTier,
  alwaysOpus,
  mapResult,
  mapExamples,
  provChips,
  buildSteps,
  normalizeSteps,
} from "../../src/format.js";

// --- numbers ---------------------------------------------------------------

test("safe4 / money: 0.0088 → '$0.0088' (test 1)", () => {
  assert.equal(safe4(0.0088), "0.0088");
  assert.equal(money(0.0088), "$0.0088");
  assert.equal(money(0.0018), "$0.0018");
});

test("number helpers guard NaN/undefined/Infinity (no $undefined/NaN)", () => {
  assert.equal(safe4(undefined), "0.0000");
  assert.equal(safe4(NaN), "0.0000");
  assert.equal(safe4("not a number"), "0.0000");
  assert.equal(money(Infinity), "$0.0000");
  assert.equal(num("3.5"), 3.5);
  assert.equal(num(undefined, 7), 7);
});

test("latency formats 2dp with 's'; clamp01 bounds [0,1]", () => {
  assert.equal(latency(1.842), "1.84s");
  assert.equal(latency(undefined), "0.00s");
  assert.equal(clamp01(0.5), 0.5);
  assert.equal(clamp01(-3), 0);
  assert.equal(clamp01(9), 1);
  assert.equal(clamp01(NaN), 0);
});

// --- savings tally (test 1 sign + test 2 accumulation) ---------------------

test("accumulateSavings = prev + (alwaysOpus - cost)", () => {
  assert.ok(Math.abs(accumulateSavings(0, 0.0018, 0.007) - 0.0052) < 1e-12);
  assert.ok(Math.abs(accumulateSavings(0.0052, 0.0088, 0.007) - 0.0034) < 1e-12);
});

test("savingsBand: positive → green ▲ (test 1/2)", () => {
  const b = savingsBand(0.0052, 0.0052, 1, 0.007);
  assert.equal(b.str, "$0.0052");
  assert.equal(b.color, "var(--accent)");
  assert.ok(b.pctStr.startsWith("▲ "));
  assert.equal(b.pct, Math.round((0.0052 / 0.007) * 100));
});

test("savingsBand: negative total → clay ▼ (escalation honesty, test 2)", () => {
  const b = savingsBand(-0.0018, -0.0018, 1, 0.007);
  assert.equal(b.str, "−$0.0018");
  assert.equal(b.color, "var(--alert)");
  assert.ok(b.pctStr.startsWith("▼ "));
  assert.ok(b.pct < 0);
});

test("savingsBand: zero runs → 0% (no divide-by-zero)", () => {
  const b = savingsBand(0, 0, 0, 0.007);
  assert.equal(b.str, "$0.0000");
  assert.equal(b.pct, 0);
  assert.equal(b.pctStr, "▲ 0%");
});

// --- tiers ------------------------------------------------------------------

test("tierName / isStrongTier / alwaysOpus", () => {
  assert.equal(tierName("claude-haiku-4-5"), "Haiku 4.5");
  assert.equal(tierName("claude-opus-4-8"), "Opus 4.8");
  assert.equal(tierName("gpt-5.5"), "gpt-5.5");
  assert.equal(tierName("unknown-model"), "unknown-model");
  const tiers = ["claude-haiku-4-5", "claude-opus-4-8"];
  assert.equal(isStrongTier("claude-opus-4-8", tiers), true);
  assert.equal(isStrongTier("claude-haiku-4-5", tiers), false);
  assert.equal(isStrongTier("claude-opus-4-8", null), true); // fallback
  assert.equal(alwaysOpus({ always_strong_cost_ref_usd: 0.0065 }), 0.0065);
  assert.equal(alwaysOpus(null), 0.007); // §5 fallback
});

// --- mapResult: cost-breakdown label + exceeds (test 3) --------------------

const TIERS = ["claude-haiku-4-5", "claude-opus-4-8"];

test("mapResult: accepted-cheap → label '= Haiku + gate', not exceeding (test 3)", () => {
  const r = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "The answer is 72.",
      correct: null,
      gate: { sufficient: true, confidence: 0.91, reason: "commits to a number" },
      p_strong: null,
      refused: false,
      cost_usd: 0.0018,
      latency_s: 0.9,
      cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  assert.equal(r.costLabel, "= Haiku + gate");
  assert.equal(r.exceeds, false);
  assert.equal(r.strong, false);
  assert.equal(r.tierName, "Haiku 4.5");
  assert.equal(r.hasAnswer, true);
  assert.equal(r.cheapRefusedEscalated, false);
  assert.equal(r.refusalSurfaced, false);
});

test("mapResult: escalated → label '= Haiku + gate + Opus', exceeds reflected (test 3)", () => {
  const r = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-opus-4-8",
      escalated: true,
      answer: "The answer is 15.",
      gate: { sufficient: false, confidence: 0.58, reason: "hedged" },
      refused: false,
      cost_usd: 0.0088,
      latency_s: 1.8,
      cost_breakdown: { label: "= Haiku + gate + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
    },
    TIERS,
  );
  assert.equal(r.costLabel, "= Haiku + gate + Opus");
  assert.equal(r.exceeds, true);
  assert.equal(r.strong, true);
});

test("mapResult: predictive single call (test 3)", () => {
  const r = mapResult(
    {
      strategy: "predictive",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "ok",
      correct: null,
      p_strong: 0.2,
      decision_margin: -0.4,
      refused: false,
      cost_usd: 0.0014,
      latency_s: 0.7,
      cost_breakdown: { label: "= Haiku", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  assert.equal(r.costLabel, "= Haiku");
  assert.equal(r.pStrong, 0.2);
  assert.equal(r.decisionMargin, -0.4);
  assert.equal(r.correct, null);
});

test("mapResult: cheap-refusal escalation → conservative chip signal", () => {
  // router.py: cheap refused → escalated, gate=null, refused=true, strong answers.
  const r = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-opus-4-8",
      escalated: true,
      answer: "Here is the answer.",
      gate: null,
      refused: true,
      cost_usd: 0.0084,
      cost_breakdown: { label: "= Haiku + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
    },
    TIERS,
  );
  assert.equal(r.cheapRefusedEscalated, true);
  assert.equal(r.refusalSurfaced, false); // strong DID answer
  assert.equal(r.hasAnswer, true);
});

test("mapResult: answering tier refused → answer empty → refusalSurfaced", () => {
  const r = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-opus-4-8",
      escalated: true,
      answer: "", // llm.call returns "" on a refusal
      gate: { sufficient: false, confidence: 0.1, reason: "doubt" },
      refused: true,
      cost_usd: 0.0086,
      cost_breakdown: { label: "= Haiku + gate + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
    },
    TIERS,
  );
  assert.equal(r.refusalSurfaced, true);
  assert.equal(r.hasAnswer, false);
});

test("mapResult: null/garbage input → null (no crash)", () => {
  assert.equal(mapResult(null, TIERS), null);
  assert.equal(mapResult(undefined, TIERS), null);
  const r = mapResult({}, TIERS); // empty object: every field defaulted, no throw
  assert.equal(r.cost, 0);
  assert.equal(r.answer, "");
  assert.equal(r.costLabel, "");
});

// --- examples / provenance --------------------------------------------------

test("mapExamples renames benchmark→bench, keeps id/label/query (test 8)", () => {
  const out = mapExamples([
    { id: "gsm8k-1", benchmark: "gsm8k", label: "g1", query: "q1", gold: 5 },
    null,
    { id: "mmlu-1", benchmark: "mmlu", label: "m1", query: "q2" },
  ]);
  assert.equal(out.length, 2);
  assert.deepEqual(out[0], { id: "gsm8k-1", bench: "gsm8k", label: "g1", query: "q1" });
  assert.equal(out[1].bench, "mmlu");
  assert.deepEqual(mapExamples(null), []);
});

test("provChips sourced from /config (no hardcoded literals)", () => {
  const chips = provChips({
    prompt_version: "v3",
    model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
    pricing_pinned_date: "2026-06-19",
  });
  assert.deepEqual(chips, [
    "prompt_version: v3",
    "tiers: haiku-4-5 → opus-4-8",
    "pricing pinned 2026-06-19",
  ]);
  assert.deepEqual(provChips(null), []);
});

// --- stepper (test 4) -------------------------------------------------------

test("buildSteps cascade accepted → Opus node hollow (test 4)", () => {
  const steps = buildSteps({
    strategy: "cascade",
    phase: "done",
    result: { escalated: false },
  });
  assert.equal(steps.length, 3);
  assert.deepEqual(steps.map((s) => s.label), ["Haiku 4.5", "gate", "Opus 4.8"]);
  assert.equal(steps[2].solid, false); // Opus hollow when accepted
  const n = normalizeSteps(steps);
  assert.equal(n[2].hollow, true);
});

test("buildSteps cascade escalated → Opus node solid (test 4)", () => {
  const steps = buildSteps({
    strategy: "cascade",
    phase: "done",
    result: { escalated: true },
  });
  assert.equal(steps[2].solid, true);
  assert.equal(steps[1].solid, true); // gate ran
});

test("buildSteps predictive → no gate node (test 4 / R6)", () => {
  const steps = buildSteps({
    strategy: "predictive",
    phase: "done",
    result: { escalated: true },
  });
  assert.equal(steps.length, 3);
  assert.deepEqual(steps.map((s) => s.label), ["embed", "classify", "Opus 4.8"]);
  assert.ok(!steps.some((s) => s.label === "gate"));
});

test("normalizeSteps shapes solid/hollow/hasLink/sub", () => {
  const n = normalizeSteps([{ label: "x", solid: true, hasLink: true, sub: "judging…" }]);
  assert.deepEqual(n[0], { label: "x", sub: "judging…", solid: true, hollow: false, hasLink: true });
});
