// SSE helpers for the e2e mock of GET /api/route/stream (split 09). They build a
// `text/event-stream` body from frames and register a Playwright route that
// fulfills the stream — so the served app is driven by a real EventSource exactly
// as it would be against the live API.

const CHEAP = "claude-haiku-4-5";
const STRONG = "claude-opus-4-8";

/** Serialize ordered `[event, data]` frames into an SSE body. */
export function sseBody(frames) {
  return frames.map(([ev, data]) => `event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`).join("");
}

/**
 * A minimal but ordered cascade/predictive event sequence whose terminal `done`
 * carries `result`. The intermediate choreography (candidate/gate/cost/escalate)
 * is included so the stepper + cost assemble visibly; the exact tiers follow the
 * fixture. For an error fixture (`{error:{...}}`) a single `error` frame is emitted.
 */
export function framesForResult(result) {
  if (result && result.error) {
    return [["error", { type: result.error.type || "api-error", message: result.error.message || "error" }]];
  }
  const isPred = result.strategy === "predictive";
  if (isPred) {
    return [
      ["phase", { phase: "embed", tier: null }],
      ["phase", { phase: "classify", tier: result.tier_used }],
      ["cost", { cost_usd_cumulative: result.cost_usd }],
      ["done", result],
    ];
  }
  const frames = [["phase", { phase: "gen", tier: CHEAP }]];
  if (result.refused && result.answer === "" && !result.escalated) {
    // answering (cheap) tier refused — straight to a refusal then done
    frames.push(["refusal", { stage: "cheap", message: "refusal" }]);
  } else {
    frames.push(["candidate", { answer: "April: 48. May: 24. Total = 72.", tier: CHEAP, cost_usd: 0.0014 }]);
    frames.push(["cost", { cost_usd_cumulative: 0.0014 }]);
    frames.push(["phase", { phase: "gate", tier: CHEAP }]);
    const g = result.gate || { sufficient: !result.escalated, confidence: 0.9, reason: "judged" };
    frames.push(["gate", { sufficient: !!g.sufficient, confidence: g.confidence ?? 0.9, reason: g.reason || "judged", cost_usd: 0.0004 }]);
  }
  if (result.escalated) {
    frames.push(["phase", { phase: "escalate", tier: STRONG }]);
    if (result.refused && result.answer === "") frames.push(["refusal", { stage: "strong", message: "refusal" }]);
  }
  frames.push(["cost", { cost_usd_cumulative: result.cost_usd }]);
  frames.push(["done", result]);
  return frames;
}

/** Register a Playwright route that fulfills the stream endpoint with `frames`. */
export async function mockStream(page, frames) {
  await page.route("**/api/route/stream**", (r) =>
    r.fulfill({ status: 200, contentType: "text/event-stream", body: sseBody(frames) }),
  );
}

/** Register a stream route that fails the transport (forces the postRoute fallback). */
export async function failStream(page) {
  await page.route("**/api/route/stream**", (r) => r.abort());
}
