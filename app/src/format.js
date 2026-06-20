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

/**
 * Provenance footer chips, sourced from /config (and, on the Frontier, the eval
 * report for `n_runs`). `report` is optional so the split-07 single-query callers
 * keep their three chips; passing a bundle's report adds the `n_runs` chip (split-08).
 */
export function provChips(config, report) {
  if (!config || typeof config !== "object") return [];
  const chips = [];
  if (config.prompt_version) chips.push("prompt_version: " + config.prompt_version);
  const tiers = Array.isArray(config.model_tiers)
    ? config.model_tiers.map((t) => String(t).replace(/^claude-/, "")).join(" → ")
    : "";
  if (tiers) chips.push("tiers: " + tiers);
  if (config.pricing_pinned_date) chips.push("pricing pinned " + config.pricing_pinned_date);
  if (report && report.n_runs != null) chips.push("n_runs: " + report.n_runs);
  return chips;
}

// ---------------------------------------------------------------------------
// Route stepper
// ---------------------------------------------------------------------------

/**
 * Build the 3-node route stepper for the current phase/strategy/result.
 * Cascade: Haiku → gate → Opus (Opus stays hollow unless escalated/escalating).
 * Predictive: embed → classify → predicted-tier (NO gate node — split-07 R6).
 *
 * `retry` (split 09) is an optional `{stage}` from a streamed 429-backoff event;
 * the matching node shows a "retrying (rate-limited)…" sublabel (overriding the
 * phase sublabel) so the stream surfaces the wait honestly, never an error toast.
 */
export function buildSteps({ strategy, phase, result, retry, cheapName = "Haiku 4.5", strongName = "Opus 4.8" }) {
  const ph = phase;
  const r = result;
  const retryStage = retry && retry.stage ? retry.stage : null;
  const sub = (stage, normal) => (retryStage === stage ? "retrying (rate-limited)…" : normal);
  if (strategy === "cascade") {
    const gateOn = ph === "gate" || ph === "escalate" || ph === "done";
    const opusOn = (ph === "done" && r && r.escalated) || ph === "escalate";
    return [
      { label: cheapName, solid: ph !== "idle", sub: sub("gen", ph === "gen" ? "generating…" : ""), hasLink: false },
      { label: "gate", solid: gateOn, sub: sub("gate", ph === "gate" ? "judging…" : ""), hasLink: true },
      { label: strongName, solid: opusOn, sub: sub("escalate", ph === "escalate" ? "escalating…" : ""), hasLink: true },
    ];
  }
  const strong = r ? !!r.escalated : false;
  return [
    { label: "embed", solid: ph !== "idle", sub: sub("embed", ph === "embed" ? "embedding…" : ""), hasLink: false },
    { label: "classify", solid: ph === "classify" || ph === "done", sub: sub("classify", ph === "classify" ? "predicting…" : ""), hasLink: true },
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
 * @param {object|null} ctx.candidate  streamed cheap-answer preview {answer,tier} or null
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
  // Streamed cheap-answer preview (split 09): shown ink-400 italic while the gate
  // is still judging, then replaced by the real result on `done`. Never a final answer.
  const cand = ctx.candidate && ctx.candidate.answer ? ctx.candidate : null;
  const showCandidate = !isError && !r && !!cand;

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
  const showAnswer = !isError && (!!r || showCandidate);
  let answerText = "";
  let answerHeading = "Answer";
  let answerBorder = "var(--line)";
  let answerColor = "var(--ink-900)";
  let answerStyle = "normal";
  let tierNameStr = "";
  let tierSolid = false;
  let latencyStr = "";
  if (!r && showCandidate) {
    // Candidate preview — the cheap answer, clearly provisional (italic, ink-400),
    // no latency/correctness yet. The gate is still deciding.
    tierNameStr = tierName(cand.tier);
    answerText = cand.answer;
    answerHeading = "Candidate · " + tierName(cand.tier);
    answerColor = "var(--ink-400)";
    answerStyle = "italic";
  }
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

// ---------------------------------------------------------------------------
// Frontier view model (split 08) — map the /api/eval/sample EvalReport bundle
// onto the chart/leaderboard shapes the FROZEN SVG geometry already consumes.
// Pure + unit-tested; the dc-runtime's renderVals() feeds these into sx/sy/etc.
// ---------------------------------------------------------------------------

/** Percent as a rounded integer string ("0.732" → "73%"); null/NaN → "—". */
function pctInt(value) {
  if (value == null) return "—";
  const n = Number(value);
  return Number.isFinite(n) ? Math.round(n * 100) + "%" : "—";
}

/** Quality 2-dp string ("0.95"); guards NaN. */
function qStr(value) {
  return num(value).toFixed(2);
}

/** A "±.02"-style spread suffix (2-dp, leading zero stripped). */
export function spreadStr(value) {
  const s = Math.abs(num(value)).toFixed(2);
  return "±" + s.replace(/^0(?=\.)/, "");
}

/** A "±1.2%"-style percentage spread for the headline caption. */
export function pctSpreadStr(value) {
  return "±" + (Math.abs(num(value)) * 100).toFixed(1) + "%";
}

/** The cascade/predictive report for a strategy, or null. */
export function pickReport(bundle, strategy) {
  const reports = bundle && Array.isArray(bundle.reports) ? bundle.reports : [];
  return reports.find((r) => r && r.strategy === strategy) || null;
}

/** FrontierPoint[] → the chart's point shape (preserve the geometry's field names). */
export function toCurve(report) {
  const pts = report && Array.isArray(report.points) ? report.points : [];
  return pts.map((p) => ({
    param: num(p && p.operating_param),
    q: num(p && p.quality),
    qSpread: num(p && p.quality_spread),
    cost: num(p && p.cost_usd_per_query),
    costSpread: num(p && p.cost_spread),
    esc: num(p && p.escalation_rate),
    n: num(p && p.n),
  }));
}

/**
 * report.baselines → `{cheap, strong, random:[cheap,randomPoint,strong], randomPoint}`.
 * `random` is the polyline the dashed baseline path draws (cheap → random → strong).
 */
export function toBaselines(report) {
  const b = (report && report.baselines) || {};
  const pt = (o) => ({ q: num(o && o.quality), cost: num(o && o.cost), qSpread: num(o && o.quality_spread) });
  const cheap = pt(b.always_cheap);
  const strong = pt(b.always_strong);
  const randomPoint = pt(b.random);
  return { cheap, strong, randomPoint, random: [cheap, randomPoint, strong] };
}

/** report.oracle → `{q, cost, qSpread}` (the unachievable ceiling). */
export function toOracle(report) {
  const o = (report && report.oracle) || {};
  return { q: num(o.quality), cost: num(o.cost), qSpread: num(o.quality_spread) };
}

/**
 * The frontier point the headline is taken from — replicates the engine's
 * `metrics.cost_reduction_at_target`: among points at/above the retention target,
 * the lowest-cost one; else the highest-retention point. Returns null when there
 * are no points / no usable strong reference.
 */
export function chosenPoint(report, target = 0.95) {
  const curve = toCurve(report);
  const strongQ = num(report && report.baselines && report.baselines.always_strong && report.baselines.always_strong.quality);
  if (!curve.length || !(strongQ > 0)) return null;
  const scored = curve
    .map((p) => ({ p, ret: p.q / strongQ }))
    .filter((s) => Number.isFinite(s.ret));
  if (!scored.length) return null;
  const qualifying = scored.filter((s) => s.ret >= target);
  const pick = qualifying.length
    ? qualifying.reduce((a, b) => (b.p.cost < a.p.cost ? b : a))
    : scored.reduce((a, b) => (b.ret > a.ret ? b : a));
  return pick.p;
}

/**
 * Interpolate a curve at operating point `t` (mirrors the dc-runtime's `interp`).
 * Used by {@link toHeadline} so the pure headline test doesn't need the component.
 */
export function interpCurve(pts, t) {
  const arr = Array.isArray(pts) ? pts : [];
  const tt = num(t);
  if (!arr.length) return { param: tt, q: 0, cost: 0, esc: 0, n: 0 };
  if (tt <= arr[0].param) return arr[0];
  const last = arr[arr.length - 1];
  if (tt >= last.param) return last;
  for (let i = 1; i < arr.length; i++) {
    if (tt <= arr[i].param) {
      const a = arr[i - 1];
      const b = arr[i];
      const denom = b.param - a.param;
      const f = denom !== 0 ? (tt - a.param) / denom : 0;
      return {
        param: tt,
        q: a.q + (b.q - a.q) * f,
        cost: a.cost + (b.cost - a.cost) * f,
        esc: a.esc + (b.esc - a.esc) * f,
        n: b.n,
      };
    }
  }
  return last;
}

/**
 * The headline at the slider's operating point: retention (% of strong accuracy)
 * and cost (% of strong cost) interpolated along the cascade curve, plus `belowBE`
 * (cost above the always-strong reference → the losing region). Empty/undefined →
 * an honest "—"/`hasData:false` (the N/A state), never a fake zero.
 */
export function toHeadline(report, frontierTau) {
  const curve = toCurve(report);
  const base = toBaselines(report);
  const strongQ = base.strong.q;
  const strongCost = base.strong.cost;
  if (!curve.length || !(strongQ > 0) || !(strongCost > 0)) {
    return { retPctStr: "—", costPctStr: "—", belowBE: false, hasData: false };
  }
  const pt = interpCurve(curve, frontierTau);
  return {
    retPctStr: Math.round((pt.q / strongQ) * 100) + "%",
    costPctStr: Math.round((pt.cost / strongCost) * 100) + "%",
    belowBE: pt.cost > strongCost + 1e-9,
    hasData: true,
  };
}

/** Largest cost in the bundle (points + baselines + oracle), ×1.07 headroom. */
export function frontierXMax(points) {
  const arr = Array.isArray(points) ? points : [];
  let max = 0;
  for (const p of arr) {
    const c = num(p && p.cost);
    if (c > max) max = c;
  }
  const headroom = max * 1.07;
  // Floor keeps the fixed $0–$.008 axis ticks meaningful for an empty/tiny run.
  return headroom > 0.001 ? headroom : 0.0086;
}

/** One FrugalRoute (cascade/predictive) leaderboard row from its report. */
function strategyRow(name, report, symbol) {
  const cp = report ? chosenPoint(report) : null;
  return {
    name,
    ours: true,
    note: cp ? symbol + "=" + num(cp.param).toFixed(2) : symbol + "=—",
    q: cp ? qStr(cp.q) : "—",
    spread: cp ? spreadStr(cp.qSpread) : "",
    cost: cp ? safe4(cp.cost) : "—",
    ret: report ? pctInt(report.retention_at_target) : "—",
    cut: report ? pctInt(report.cost_reduction_at_target) : "—",
  };
}

/**
 * The six fixed-order leaderboard rows (§11 / the engine's `format_leaderboard`):
 * always-cheap · always-strong · random · cascade · predictive · oracle. Baseline
 * retention/cut are measured against always-strong; the two FrugalRoute rows use
 * the report's headline `retention_at_target`/`cost_reduction_at_target`; the
 * oracle's retention/cut are "—" (a ceiling, not a target).
 */
export function toLeaderboardRows(bundle) {
  const cascade = pickReport(bundle, "cascade");
  const predictive = pickReport(bundle, "predictive");
  const ref = cascade || predictive;
  if (!ref) return [];
  const b = ref.baselines || {};
  const cheap = b.always_cheap || {};
  const strong = b.always_strong || {};
  const rand = b.random || {};
  const oracle = ref.oracle || {};
  const strongQ = num(strong.quality);
  const strongCost = num(strong.cost);
  const retOf = (q) => (strongQ > 0 ? pctInt(num(q) / strongQ) : "—");
  const cutOf = (cost) => (strongCost > 0 ? pctInt(1 - num(cost) / strongCost) : "—");

  const rows = [
    {
      name: "always-cheap", ours: false, note: "cost floor",
      q: qStr(cheap.quality), spread: spreadStr(cheap.quality_spread),
      cost: safe4(cheap.cost), ret: retOf(cheap.quality), cut: cutOf(cheap.cost),
    },
    {
      name: "always-strong", ours: false, note: "reference",
      q: qStr(strong.quality), spread: spreadStr(strong.quality_spread),
      cost: safe4(strong.cost), ret: retOf(strong.quality), cut: cutOf(strong.cost),
    },
    {
      name: "random @ cost", ours: false, note: "matched-cost chance",
      q: qStr(rand.quality), spread: spreadStr(rand.quality_spread),
      cost: safe4(rand.cost), ret: retOf(rand.quality), cut: cutOf(rand.cost),
    },
    strategyRow("FrugalRoute · cascade", cascade, "τ"),
    strategyRow("FrugalRoute · predictive", predictive, "θ"),
    {
      name: "oracle (ceiling)", ours: false, note: "uses ground truth",
      q: qStr(oracle.quality), spread: spreadStr(oracle.quality_spread),
      cost: safe4(oracle.cost), ret: "—", cut: "—",
    },
  ];
  return rows.map((r) => ({ ...r, nameColor: r.ours ? "var(--accent)" : "var(--ink-900)" }));
}

/** The leaderboard's frozen-split caption pieces, from the bundle (split-08 §2a). */
export function frozenSplitNote(bundle) {
  const fs = (bundle && bundle.frozen_split) || {};
  const n = fs.n_test == null ? "" : String(fs.n_test);
  return { n, wideCI: fs.small_n ? " · wide CI" : "" };
}
