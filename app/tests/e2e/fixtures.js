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
