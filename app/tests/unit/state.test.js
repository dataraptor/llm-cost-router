// Pure-logic coverage of all 8 single-query states + the adversarial R10 inputs
// (split-07 §4 state table). `deriveSingleQuery` is the function the dc-runtime's
// renderVals() overlays, so these assert the exact bindings the template renders —
// a browser-free proof that every state renders correctly (complements the e2e).

import { test } from "node:test";
import assert from "node:assert/strict";

import { deriveSingleQuery, mapResult } from "../../src/format.js";

const TIERS = ["claude-haiku-4-5", "claude-opus-4-8"];
const CONFIG = { model_tiers: TIERS, always_strong_cost_ref_usd: 0.007 };

function ctx(over) {
  return {
    strategy: "cascade",
    phase: "idle",
    result: null,
    gate: null,
    costDisplay: 0,
    tau: 0.8,
    config: CONFIG,
    error: null,
    ...over,
  };
}

// 1. Idle ------------------------------------------------------------------
test("state: idle — hollow stepper, idle card, no answer/gate/cost/error", () => {
  const v = deriveSingleQuery(ctx());
  assert.equal(v.showIdle, true);
  assert.equal(v.showAnswer, false);
  assert.equal(v.showGate, false);
  assert.equal(v.showCost, false);
  assert.equal(v.showError, false);
});

// 2. Loading ---------------------------------------------------------------
test("state: loading (gen) — no answer text yet, cost label 'routing…'", () => {
  const v = deriveSingleQuery(ctx({ phase: "gen" }));
  assert.equal(v.showIdle, false);
  assert.equal(v.showAnswer, false); // never a fabricated/candidate answer
  assert.equal(v.showCost, true);
  assert.equal(v.costBreakdown, "routing…");
  assert.equal(v.costStr, "0.0000");
});

// 3. Accepted at cheap -----------------------------------------------------
test("state: accepted-cheap — gate sufficient, no loss, Haiku hollow tier badge", () => {
  const result = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "The answer is 72.",
      gate: { sufficient: true, confidence: 0.91, reason: "commits" },
      refused: false,
      cost_usd: 0.0018,
      latency_s: 0.9,
      cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  const v = deriveSingleQuery(ctx({ phase: "done", result, gate: result.gate, costDisplay: 0.0018 }));
  assert.equal(v.showAnswer, true);
  assert.equal(v.answerText, "The answer is 72.");
  assert.equal(v.answerHeading, "Accepted at Haiku 4.5");
  assert.equal(v.showGate, true);
  assert.equal(v.gateVerdict, "sufficient");
  assert.equal(v.gateConfStr, "0.91");
  assert.equal(v.showLoss, false);
  assert.equal(v.tierSolid, false); // cheap accepted → hollow badge
  assert.equal(v.showCorrectNote, true); // correctness never shown (live)
  assert.equal(v.costStr, "0.0018");
  assert.equal(v.costBreakdown, "= Haiku + gate");
});

// 4. Escalated (honest loss) ----------------------------------------------
test("state: escalated — gate insufficient, solid Opus, honest loss flagged", () => {
  const result = mapResult(
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
  const v = deriveSingleQuery(ctx({ phase: "done", result, gate: result.gate, costDisplay: 0.0088 }));
  assert.equal(v.answerHeading, "Answer · escalated");
  assert.equal(v.gateVerdict, "insufficient");
  assert.equal(v.tierSolid, true); // strong → solid badge
  assert.equal(v.showLoss, true); // ⚠ more than Opus-only
  assert.equal(v.costBreakdown, "= Haiku + gate + Opus");
});

test("state: escalated but NOT exceeding always-Opus → no false loss warning", () => {
  const result = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-opus-4-8",
      escalated: true,
      answer: "ok",
      gate: { sufficient: false, confidence: 0.4, reason: "x" },
      refused: false,
      cost_usd: 0.0069,
      cost_breakdown: { label: "= Haiku + gate + Opus", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  const v = deriveSingleQuery(ctx({ phase: "done", result, gate: result.gate }));
  assert.equal(v.showLoss, false);
});

// 5. Predictive (no gate node, margin, no checkmark) -----------------------
test("state: predictive — margin shown, NO gate, correctness never shown (R6)", () => {
  const result = mapResult(
    {
      strategy: "predictive",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "predicted answer",
      correct: null,
      p_strong: 0.18,
      decision_margin: -0.42,
      refused: false,
      cost_usd: 0.0014,
      cost_breakdown: { label: "= Haiku", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  const v = deriveSingleQuery(
    ctx({ strategy: "predictive", phase: "done", result, gate: null, tau: 0.6, costDisplay: 0.0014 }),
  );
  assert.equal(v.showMargin, true);
  assert.equal(v.pStrongStr, "0.18");
  assert.equal(v.showGate, false); // no gate node/card
  assert.equal(v.answerHeading, "Answer · predicted");
  assert.equal(v.showCorrectNote, true);
  assert.equal(v.showLoss, false); // loss is cascade-only
});

// 6. Refusal (cheap refused → conservative chip) ---------------------------
test("state: cheap refusal → conservative chip, strong answer surfaced", () => {
  const result = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-opus-4-8",
      escalated: true,
      answer: "Here is the real answer.",
      gate: null,
      refused: true,
      cost_usd: 0.0084,
      cost_breakdown: { label: "= Haiku + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
    },
    TIERS,
  );
  const v = deriveSingleQuery(ctx({ phase: "done", result, gate: null }));
  assert.equal(v.showRefuseChip, true);
  assert.equal(v.showRefuseStrong, false); // strong actually answered
  assert.equal(v.answerText, "Here is the real answer.");
});

// 6b. Refusal (answering tier refused → alert card) ------------------------
test("state: strong refusal → alert-edged card 'surfaced, not hidden', no fabricated answer", () => {
  const result = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-opus-4-8",
      escalated: true,
      answer: "", // refusal → empty
      gate: { sufficient: false, confidence: 0.2, reason: "doubt" },
      refused: true,
      cost_usd: 0.0086,
      cost_breakdown: { label: "= Haiku + gate + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
    },
    TIERS,
  );
  const v = deriveSingleQuery(ctx({ phase: "done", result, gate: result.gate }));
  assert.equal(v.showRefuseStrong, true);
  assert.equal(v.answerBorder, "var(--alert)");
  assert.match(v.answerText, /refusal/); // honest placeholder, never fabricated
  assert.ok(!v.answerText.includes("undefined"));
});

// 7. Missing key (blocking card + escape hatch) ----------------------------
test("state: missing-key → blocking card naming the key + 'View the Proof' hatch", () => {
  const v = deriveSingleQuery(
    ctx({ phase: "idle", error: { type: "missing-key", message: "Set ANTHROPIC_API_KEY (or Azure)…" } }),
  );
  assert.equal(v.showError, true);
  assert.equal(v.showErrorProof, true); // escape hatch present
  assert.match(v.errorMessage, /ANTHROPIC_API_KEY/);
  assert.equal(v.showIdle, false); // error replaces idle
  assert.equal(v.showAnswer, false);
});

// 8. API error (inline card, no escape-hatch button) -----------------------
test("state: api-error → inline card naming the cause, Route re-enabled", () => {
  const v = deriveSingleQuery(
    ctx({ phase: "idle", error: { type: "api-error", message: "The model backend returned an error." } }),
  );
  assert.equal(v.showError, true);
  assert.equal(v.showErrorProof, false);
  assert.match(v.errorMessage, /backend/);
});

// --- Adversarial (R10) ----------------------------------------------------
test("R10: empty answer (not a refusal) → honest empty, never fabricated/NaN", () => {
  const result = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "",
      gate: { sufficient: true, confidence: 0.9, reason: "x" },
      refused: false,
      cost_usd: 0,
      cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  const v = deriveSingleQuery(ctx({ phase: "done", result, gate: result.gate, costDisplay: 0 }));
  assert.equal(v.answerText, "(no answer returned)");
  assert.equal(v.costStr, "0.0000"); // cost_usd 0 → honest, not NaN
  assert.ok(!String(v.answerText).includes("undefined"));
});

test("R10: malformed gate (missing confidence) → 0.00, no NaN", () => {
  const v = deriveSingleQuery(
    ctx({
      phase: "done",
      result: { escalated: false, hasAnswer: true, answer: "x", tierName: "Haiku 4.5", strong: false, cost: 0.001 },
      gate: { reason: "no confidence field here" }, // sufficient & confidence missing
    }),
  );
  assert.equal(v.gateConfStr, "0.00");
  assert.equal(v.gateConfPct, 0);
  assert.equal(v.gateVerdict, "insufficient"); // missing sufficient → falsy
  assert.ok(!v.gateConfStr.includes("NaN"));
});

test("R10: costDisplay NaN/undefined → '0.0000' (never $undefined)", () => {
  const v = deriveSingleQuery(ctx({ phase: "gen", costDisplay: undefined }));
  assert.equal(v.costStr, "0.0000");
});

// --- Streaming candidate preview (split 09) -------------------------------
test("stream: candidate during gate → italic ink-400 preview, no correctness/refusal", () => {
  const v = deriveSingleQuery(
    ctx({ phase: "gate", candidate: { answer: "April: 48. May: 24. Total = 72.", tier: "claude-haiku-4-5" } }),
  );
  assert.equal(v.showAnswer, true);
  assert.equal(v.answerText, "April: 48. May: 24. Total = 72.");
  assert.match(v.answerHeading, /Candidate · Haiku 4\.5/);
  assert.equal(v.answerStyle, "italic");
  assert.equal(v.answerColor, "var(--ink-400)");
  assert.equal(v.tierHollow, true); // cheap candidate isn't the strong tier
  assert.equal(v.latencyStr, ""); // no latency until done
  assert.equal(v.showCorrectNote, false); // provisional → no correctness note yet
  assert.equal(v.showRefuseStrong, false);
});

test("stream: the real result replaces the candidate on done", () => {
  const result = mapResult(
    {
      strategy: "cascade",
      tier_used: "claude-haiku-4-5",
      escalated: false,
      answer: "The answer is 72.",
      gate: { sufficient: true, confidence: 0.91, reason: "commits" },
      refused: false,
      cost_usd: 0.0018,
      latency_s: 0.9,
      cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
    },
    TIERS,
  );
  // Even if a stale candidate is still in ctx, a present result wins.
  const v = deriveSingleQuery(
    ctx({ phase: "done", result, gate: result.gate, candidate: { answer: "stale", tier: "claude-haiku-4-5" } }),
  );
  assert.equal(v.answerText, "The answer is 72.");
  assert.equal(v.answerHeading, "Accepted at Haiku 4.5");
  assert.equal(v.showCorrectNote, true);
});

test("stream: a candidate is suppressed while an error is showing", () => {
  const v = deriveSingleQuery(
    ctx({ phase: "gate", candidate: { answer: "x", tier: "claude-haiku-4-5" }, error: { type: "api-error", message: "boom" } }),
  );
  assert.equal(v.showError, true);
  assert.equal(v.showAnswer, false); // error trumps the candidate preview
});
