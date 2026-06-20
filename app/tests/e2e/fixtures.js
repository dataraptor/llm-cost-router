// §06-shaped mock payloads for the e2e (split-07). These mirror the FROZEN HTTP
// contract (api/src/frugalroute_api/schemas.py) so the UI is exercised against
// exactly what the real API returns.

export const CONFIG = {
  prompt_version: "v3",
  model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
  strategies: ["cascade", "predictive"],
  pricing: {
    "claude-haiku-4-5": { input_per_mtok: 1.0, output_per_mtok: 5.0 },
    "claude-opus-4-8": { input_per_mtok: 5.0, output_per_mtok: 25.0 },
  },
  always_strong_cost_ref_usd: 0.007,
  defaults: { tau: 0.8, theta: 0.6 },
  pricing_pinned_date: "2026-06-19",
};

export const EXAMPLES = [
  {
    id: "gsm8k-1142",
    benchmark: "gsm8k",
    label: "gsm8k #1142",
    query:
      "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?",
  },
  { id: "gsm8k-882", benchmark: "gsm8k", label: "gsm8k #882 · hard", query: "A robe takes 2 bolts…" },
  { id: "refuse-1", benchmark: "gsm8k", label: "edge · cheap refuses", query: "Walk me through how to…" },
];

export const ACCEPTED = {
  query: EXAMPLES[0].query,
  strategy: "cascade",
  tier_used: "claude-haiku-4-5",
  escalated: false,
  answer: "April: 48. May: 24. Total = 72.\nThe answer is 72.",
  correct: null,
  gate: { sufficient: true, confidence: 0.91, reason: "Commits to a single number." },
  p_strong: null,
  refused: false,
  cost_usd: 0.0018,
  latency_s: 0.92,
  prompt_version: "v3",
  decision_margin: null,
  cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
};

export const ESCALATED = {
  query: EXAMPLES[1].query,
  strategy: "cascade",
  tier_used: "claude-opus-4-8",
  escalated: true,
  answer: "Robe 1: 3. Robe 2: 12. Total = 15.\nThe answer is 15.",
  correct: null,
  gate: { sufficient: false, confidence: 0.58, reason: "Hedged between two totals." },
  p_strong: null,
  refused: false,
  cost_usd: 0.0088,
  latency_s: 1.87,
  prompt_version: "v3",
  decision_margin: null,
  cost_breakdown: { label: "= Haiku + gate + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
};

export const STRONG_REFUSAL = {
  query: EXAMPLES[2].query,
  strategy: "cascade",
  tier_used: "claude-opus-4-8",
  escalated: true,
  answer: "", // llm.call returns "" on a refusal
  correct: null,
  gate: { sufficient: false, confidence: 0.0, reason: "doubt" },
  p_strong: null,
  refused: true,
  cost_usd: 0.0086,
  latency_s: 1.4,
  prompt_version: "v3",
  decision_margin: null,
  cost_breakdown: { label: "= Haiku + gate + Opus", always_strong_usd: 0.007, exceeds_always_strong: true },
};

export const PREDICTIVE = {
  query: EXAMPLES[0].query,
  strategy: "predictive",
  tier_used: "claude-haiku-4-5",
  escalated: false,
  answer: "Predicted upfront. The answer is 72.",
  correct: null,
  gate: null,
  p_strong: 0.18,
  refused: false,
  cost_usd: 0.0014,
  latency_s: 0.83,
  prompt_version: "v3",
  decision_margin: 0.18 - 0.6,
  cost_breakdown: { label: "= Haiku", always_strong_usd: 0.007, exceeds_always_strong: false },
};

export const MISSING_KEY = {
  error: {
    type: "missing-key",
    message:
      "No model backend is configured. Set ANTHROPIC_API_KEY (or the Azure OpenAI credentials) to run live routing, or view the precomputed proof at GET /api/eval/sample.",
    detail: "ANTHROPIC_API_KEY is not set.",
  },
};

export const API_ERROR = {
  error: { type: "api-error", message: "The model backend returned an error.", detail: "503 upstream" },
};

// --- Frontier (split-08): the /api/eval/sample bundle -----------------------
// Shaped exactly like api/src/frugalroute_api/data/sample_run.json. always-strong
// = {q:1.0, cost:0.0065}; the cascade's lowest-cost ≥95%-retention point is τ=0.5
// (cost 0.0039 = 60% of strong → 40% cut), so the default settled headline reads
// "Retains 100% of Opus accuracy at 60% of the cost."
const FRONTIER_BASELINES = {
  always_cheap: { quality: 0.6, quality_spread: 0.0, cost: 0.0009, cost_spread: 2e-5 },
  always_strong: { quality: 1.0, quality_spread: 0.0, cost: 0.0065, cost_spread: 1.5e-4 },
  random: { quality: 0.75, quality_spread: 0.0, cost: 0.0037, cost_spread: 9e-5 },
};
const FRONTIER_ORACLE = { quality: 1.0, quality_spread: 0.0, cost: 0.003 };

export const BUNDLE = {
  reports: [
    {
      strategy: "cascade",
      points: [
        { operating_param: 0.5, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.0039, cost_spread: 1e-4, escalation_rate: 0.375, n: 8 },
        { operating_param: 0.8, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.00475, cost_spread: 1.2e-4, escalation_rate: 0.5, n: 8 },
        { operating_param: 1.0, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.008, cost_spread: 2e-4, escalation_rate: 1.0, n: 8 },
      ],
      baselines: FRONTIER_BASELINES,
      oracle: FRONTIER_ORACLE,
      retention_at_target: 1.0,
      retention_at_target_spread: 0.012,
      cost_reduction_at_target: 0.4,
      cost_reduction_at_target_spread: 0.0,
      n_refused: 0,
      prompt_version: "v1",
      model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
      n_runs: 3,
    },
    {
      strategy: "predictive",
      points: [
        { operating_param: 0.4, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.0037, cost_spread: 9e-5, escalation_rate: 0.5, n: 8 },
        { operating_param: 0.6, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.003, cost_spread: 7e-5, escalation_rate: 0.375, n: 8 },
      ],
      baselines: FRONTIER_BASELINES,
      oracle: FRONTIER_ORACLE,
      retention_at_target: 1.0,
      retention_at_target_spread: 0.0,
      cost_reduction_at_target: 0.5384615384615385,
      cost_reduction_at_target_spread: 0.0,
      n_refused: 0,
      prompt_version: "v1",
      model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
      n_runs: 3,
    },
  ],
  benchmark: "gsm8k",
  frozen_split: { n_test: 8, n_calibration: 32, small_n: true },
  generated_at: "2026-06-20T00:00:00+00:00",
};

// Adversarial (R10 / §8): a cascade frontier with a point ABOVE always-strong cost
// (a real losing region), an EMPTY predictive curve, and n_refused > 0. The default
// settled headline must honestly read "Below break-even — costs more than Opus here."
export const BUNDLE_LOSING = {
  reports: [
    {
      strategy: "cascade",
      points: [
        { operating_param: 0.5, quality: 0.9, quality_spread: 0.01, cost_usd_per_query: 0.004, cost_spread: 1e-4, escalation_rate: 0.4, n: 8 },
        { operating_param: 1.0, quality: 1.0, quality_spread: 0.0, cost_usd_per_query: 0.009, cost_spread: 2e-4, escalation_rate: 1.0, n: 8 },
      ],
      baselines: {
        always_cheap: { quality: 0.55, quality_spread: 0.0, cost: 0.001, cost_spread: 0.0 },
        always_strong: { quality: 1.0, quality_spread: 0.0, cost: 0.007, cost_spread: 0.0 },
        random: { quality: 0.72, quality_spread: 0.0, cost: 0.004, cost_spread: 0.0 },
      },
      oracle: { quality: 1.0, quality_spread: 0.0, cost: 0.003 },
      retention_at_target: 1.0,
      retention_at_target_spread: 0.0,
      cost_reduction_at_target: 1 - 0.009 / 0.007, // negative — costs MORE than Opus
      cost_reduction_at_target_spread: 0.0,
      n_refused: 2,
      prompt_version: "v1",
      model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
      n_runs: 3,
    },
    {
      strategy: "predictive",
      points: [], // empty curve — must degrade gracefully (no dashed line, no crash)
      baselines: {
        always_cheap: { quality: 0.55, quality_spread: 0.0, cost: 0.001, cost_spread: 0.0 },
        always_strong: { quality: 1.0, quality_spread: 0.0, cost: 0.007, cost_spread: 0.0 },
        random: { quality: 0.72, quality_spread: 0.0, cost: 0.004, cost_spread: 0.0 },
      },
      oracle: { quality: 1.0, quality_spread: 0.0, cost: 0.003 },
      retention_at_target: null,
      retention_at_target_spread: null,
      cost_reduction_at_target: null,
      cost_reduction_at_target_spread: null,
      n_refused: 2,
      prompt_version: "v1",
      model_tiers: ["claude-haiku-4-5", "claude-opus-4-8"],
      n_runs: 3,
    },
  ],
  benchmark: "gsm8k",
  frozen_split: { n_test: 8, n_calibration: 32, small_n: true },
  generated_at: "2026-06-20T00:00:00+00:00",
};

// Adversarial (R10): empty answer + malformed gate (no confidence) + cost 0.
export const ADVERSARIAL = {
  query: EXAMPLES[0].query,
  strategy: "cascade",
  tier_used: "claude-haiku-4-5",
  escalated: false,
  answer: "",
  correct: null,
  gate: { reason: "no confidence field" },
  p_strong: null,
  refused: false,
  cost_usd: 0,
  latency_s: 0,
  prompt_version: "v3",
  decision_margin: null,
  cost_breakdown: { label: "= Haiku + gate", always_strong_usd: 0.007, exceeds_always_strong: false },
};
