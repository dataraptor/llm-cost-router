// format.js — pure mapping/format helpers extracted from the dc-runtime logic
// (split-07 §2/§3). NO DOM, NO fetch — unit-tested headless. The embedded
// `data-dc-script` calls these via `window.FR.format.*`; tests import them directly.
//
// Everything here is defensive: a malformed/partial API payload (missing field,
// NaN cost, empty answer) must degrade to an honest value, never `NaN`/`$undefined`
// and never a fabricated answer (split-07 R10).

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

/** A finite number or the fallback (guards undefined/NaN/Infinity/strings). */
export function num(value, fallback = 0) {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** 4-dp money string WITHOUT the `$` (the template renders the `$` literally). */
export function safe4(value) {
  return num(value).toFixed(4);
}

/** 4-dp money string WITH the `$` (e.g. `money(0.0088) === "$0.0088"`). */
export function money(value) {
  return "$" + safe4(value);
}

/** Latency seconds, 2-dp, with the `s` suffix (e.g. `"1.84s"`). */
export function latency(value) {
  return num(value).toFixed(2) + "s";
}

/** Clamp to [0,1]; non-finite → 0 (gate confidence may arrive malformed). */
export function clamp01(value) {
  const n = num(value, 0);
  return n < 0 ? 0 : n > 1 ? 1 : n;
}

// ---------------------------------------------------------------------------
// Savings tally (running "saved vs always-Opus")
// ---------------------------------------------------------------------------

/** Add one route's contribution: `prevTotal + (alwaysOpus - cost)`. */
export function accumulateSavings(prevTotal, cost, alwaysOpus) {
  return num(prevTotal) + (num(alwaysOpus) - num(cost));
}

/**
 * The header savings band, derived distributionally-honest:
 * negative total renders clay with a `▼`, positive renders green with `▲`.
 *
 * @returns {{str:string,color:string,pct:number,pctStr:string}}
 */
export function savingsBand(total, display, runs, alwaysOpus) {
  const t = num(total);
  const neg = t < -1e-9;
  const str = (neg ? "−$" : "$") + Math.abs(num(display)).toFixed(4);
  const a = num(alwaysOpus);
  const r = num(runs);
  const pct = r > 0 && a > 0 ? Math.round((t / (r * a)) * 100) : 0;
  const pctStr = (pct < 0 ? "▼ " : "▲ ") + Math.abs(pct) + "%";
  return {
    str,
    color: neg ? "var(--alert)" : "var(--accent)",
    pct,
    pctStr,
  };
}

// ---------------------------------------------------------------------------
// Tiers
// ---------------------------------------------------------------------------

const TIER_NAMES = {
  "claude-haiku-4-5": "Haiku 4.5",
  "claude-sonnet-4-6": "Sonnet 4.6",
  "claude-opus-4-8": "Opus 4.8",
  "gpt-5.5": "gpt-5.5",
};

/** Human display name for a model id (falls back to the raw id). */
export function tierName(id) {
  return TIER_NAMES[id] || id || "";
}

/** Is `id` the strongest tier? (last of the ordered `tiers`, else Opus). */
export function isStrongTier(id, tiers) {
  if (Array.isArray(tiers) && tiers.length > 0) return id === tiers[tiers.length - 1];
  return id === "claude-opus-4-8";
}

/** Always-Opus per-query cost reference from /config, with the §5 fallback. */
export function alwaysOpus(config) {
  return config ? num(config.always_strong_cost_ref_usd, 0.007) : 0.007;
}

// ---------------------------------------------------------------------------
// API → local view shape
// ---------------------------------------------------------------------------

/**
 * Map an `/api/route` RouteResponse onto the flat shape the template binds.
 * Faithful: only the engine's fields are used; the cost-breakdown label/exceeds
 * come straight from the API (split-06 reports the total cost only, no per-term USD).
 *
 * Refusal disambiguation (router.py semantics):
 *   - cheap refused → escalated, `gate === null`, `refused === true` (strong answers)
 *     → conservative chip "Haiku refused → escalated".
 *   - the answering tier itself refused → `refused === true` and `answer === ""`
 *     (llm.call returns text="" on a refusal) → alert-edged "surfaced, not hidden" card.
 */
export function mapResult(api, tiers) {
  if (!api || typeof api !== "object") return null;
  const tier = typeof api.tier_used === "string" ? api.tier_used : "";
  const answer = typeof api.answer === "string" ? api.answer : "";
  const hasAnswer = answer.trim().length > 0;
  const refused = api.refused === true;
  const escalated = api.escalated === true;
  const cascade = api.strategy === "cascade";
  const cb = api.cost_breakdown && typeof api.cost_breakdown === "object" ? api.cost_breakdown : {};
  return {
    strategy: typeof api.strategy === "string" ? api.strategy : "",
    tier,
    tierName: tierName(tier),
    strong: isStrongTier(tier, tiers),
    answer,
    hasAnswer,
    cost: num(api.cost_usd),
    latency: num(api.latency_s),
    escalated,
    refused,
    // The displayed answer is itself a refusal (no real answer to show).
    refusalSurfaced: refused && !hasAnswer,
    // Cascade escalated specifically because the cheap tier refused (gate skipped).
    cheapRefusedEscalated: cascade && escalated && api.gate == null && refused,
    gate: api.gate && typeof api.gate === "object" ? api.gate : null,
    pStrong: api.p_strong == null ? null : num(api.p_strong, null),
    decisionMargin: api.decision_margin == null ? null : num(api.decision_margin, null),
    correct: api.correct == null ? null : api.correct,
    costLabel: typeof cb.label === "string" ? cb.label : "",
    exceeds: cb.exceeds_always_strong === true,
  };
}

/** Picker entries — only id/benchmark/label/query (answers are NOT served). */
export function mapExamples(list) {
  if (!Array.isArray(list)) return [];
  return list
    .filter((e) => e && typeof e === "object")
    .map((e) => ({ id: e.id, bench: e.benchmark, label: e.label, query: e.query }));
}

/** Provenance footer chips, sourced entirely from /config. */
export function provChips(config) {
  if (!config || typeof config !== "object") return [];
  const chips = [];
  if (config.prompt_version) chips.push("prompt_version: " + config.prompt_version);
  const tiers = Array.isArray(config.model_tiers)
    ? config.model_tiers.map((t) => String(t).replace(/^claude-/, "")).join(" → ")
    : "";
  if (tiers) chips.push("tiers: " + tiers);
  if (config.pricing_pinned_date) chips.push("pricing pinned " + config.pricing_pinned_date);
  return chips;
}

// ---------------------------------------------------------------------------
// Route stepper
// ---------------------------------------------------------------------------

/**
 * Build the 3-node route stepper for the current phase/strategy/result.
 * Cascade: Haiku → gate → Opus (Opus stays hollow unless escalated/escalating).
 * Predictive: embed → classify → predicted-tier (NO gate node — split-07 R6).
 */
export function buildSteps({ strategy, phase, result, cheapName = "Haiku 4.5", strongName = "Opus 4.8" }) {
  const ph = phase;
  const r = result;
  if (strategy === "cascade") {
    const gateOn = ph === "gate" || ph === "escalate" || ph === "done";
    const opusOn = (ph === "done" && r && r.escalated) || ph === "escalate";
    return [
      { label: cheapName, solid: ph !== "idle", sub: ph === "gen" ? "generating…" : "", hasLink: false },
      { label: "gate", solid: gateOn, sub: ph === "gate" ? "judging…" : "", hasLink: true },
      { label: strongName, solid: opusOn, sub: ph === "escalate" ? "escalating…" : "", hasLink: true },
    ];
  }
  const strong = r ? !!r.escalated : false;
  return [
    { label: "embed", solid: ph !== "idle", sub: ph === "embed" ? "embedding…" : "", hasLink: false },
    { label: "classify", solid: ph === "classify" || ph === "done", sub: ph === "classify" ? "predicting…" : "", hasLink: true },
    { label: strong ? strongName : cheapName, solid: ph === "done" && strong, sub: "", hasLink: true },
  ];
}

/** Shape steps for the template's `solid`/`hollow`/`hasLink`/`sub` bindings. */
export function normalizeSteps(steps) {
  return (Array.isArray(steps) ? steps : []).map((s) => ({
    label: s.label,
    sub: s.sub || "",
    solid: !!s.solid,
    hollow: !s.solid,
    hasLink: !!s.hasLink,
  }));
}

// ---------------------------------------------------------------------------
// Single-query view model (the 8 states) — pure, fully unit-testable
// ---------------------------------------------------------------------------

/**
 * Derive every single-query binding from the live/mock state.
 *
 * @param {object} ctx
 * @param {string}      ctx.strategy   "cascade" | "predictive"
 * @param {string}      ctx.phase      "idle"|"gen"|"embed"|"gate"|"classify"|"escalate"|"done"
 * @param {object|null} ctx.result     mapped RouteResult (from {@link mapResult}) or null
 * @param {object|null} ctx.gate       the gate verdict to display (cascade)
 * @param {number}      ctx.costDisplay tweened cost (USD)
 * @param {number}      ctx.tau        the operating-point slider value (τ or θ)
 * @param {object|null} ctx.config     /config payload (for the always-Opus ref)
 * @param {object|null} ctx.error      normalized ApiError {type,message} or null
 */
export function deriveSingleQuery(ctx) {
  const { strategy, phase, result: r, gate, costDisplay, tau, config, error } = ctx;
  const isC = strategy === "cascade";
  const ph = phase;
  const isError = !!error;
  const missingKey = isError && error.type === "missing-key";

  // --- error card -----------------------------------------------------------
  const errorTitle = missingKey ? "Live routing unavailable" : "Routing error";
  const errorMessage = isError
    ? error.message || "The router could not complete this request."
    : "";

  // --- gate / margin --------------------------------------------------------
  const showGate = !isError && isC && !!gate && (ph === "gate" || ph === "escalate" || ph === "done");
  const showMargin = !isError && !isC && ph === "done" && !!r;
  const gateConf = clamp01(gate && gate.confidence);
  const ps = clamp01(r && r.pStrong);
  const tauPct = Math.round(clamp01(tau) * 100);

  // --- answer ---------------------------------------------------------------
  const showAnswer = !isError && !!r;
  let answerText = "";
  let answerHeading = "Answer";
  let answerBorder = "var(--line)";
  let answerColor = "var(--ink-900)";
  let answerStyle = "normal";
  let tierNameStr = "";
  let tierSolid = false;
  let latencyStr = "";
  if (r) {
    tierNameStr = r.tierName;
    tierSolid = !!r.strong;
    latencyStr = latency(r.latency);
    const baseHeading = isC
      ? r.escalated
        ? "Answer · escalated"
        : "Accepted at " + r.tierName
      : "Answer · predicted";
    if (r.refusalSurfaced) {
      answerBorder = "var(--alert)";
      answerColor = "var(--alert)";
      answerHeading = "Answer · refusal surfaced";
      answerText = r.hasAnswer ? r.answer : "(no answer — the model returned a refusal)";
    } else if (r.hasAnswer) {
      answerText = r.answer;
      answerHeading = baseHeading;
    } else {
      // Empty answer, not a refusal (degenerate/adversarial) — honest, never faked.
      answerText = "(no answer returned)";
      answerColor = "var(--ink-400)";
      answerStyle = "italic";
      answerHeading = baseHeading;
    }
  }

  // --- cost -----------------------------------------------------------------
  const showCost = !isError && (!!r || ph !== "idle");
  let costBreakdown = "";
  if (r) costBreakdown = r.costLabel || "";
  else if (ph !== "idle") costBreakdown = "routing…";
  // Honest escalation loss: only when an escalated cascade actually cost more.
  const showLoss = !!r && isC && r.escalated && r.exceeds;

  return {
    // gate
    showGate,
    gateVerdict: gate && gate.sufficient ? "sufficient" : "insufficient",
    gateConfStr: gateConf.toFixed(2),
    gateConfPct: Math.round(gateConf * 100),
    gateReason: gate ? gate.reason || "" : "",
    tauTickPct: tauPct,
    // predictive margin
    showMargin,
    pStrongStr: ps.toFixed(2),
    pStrongPct: Math.round(ps * 100),
    // refusal chip (cheap refused → conservative escalation)
    showRefuseChip: !!r && !!r.cheapRefusedEscalated,
    // answer
    showAnswer,
    answerText,
    answerHeading,
    answerBorder,
    answerColor,
    answerStyle,
    tierName: tierNameStr,
    tierSolid,
    tierHollow: !tierSolid,
    latencyStr,
    showRefuseStrong: !!r && !!r.refusalSurfaced,
    showCorrectNote: !!r, // correct is None in live mode — never a checkmark
    // cost
    showCost,
    costStr: safe4(costDisplay),
    costBreakdown,
    showLoss,
    // idle / error
    showIdle: ph === "idle" && !r && !isError,
    showError: isError,
    errorTitle,
    errorMessage,
    showErrorProof: missingKey,
  };
}
