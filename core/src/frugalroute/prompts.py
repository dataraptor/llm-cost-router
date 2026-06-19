"""Canonical prompts (build-spec §6), pinned behind ``PROMPT_VERSION``.

The gate system prompt and the per-benchmark generation prompts are reproduced
**verbatim** from §6 — do not paraphrase. ``PROMPT_VERSION`` is recorded in every
``RouteResult`` and ``EvalReport`` for reproducibility; bump it only when a prompt
string here actually changes.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

# Cascade quality gate (Haiku 4.5) — system prompt, verbatim from build-spec §6.
GATE_SYSTEM = """You are a strict answer-quality gate in a model-routing system. You are given a
QUESTION and a candidate ANSWER produced by a fast, inexpensive model. Decide
whether that answer is reliable enough to return as-is, or whether the question
should be escalated to a stronger, more expensive model.

Judge the ANSWER on its own merits against the QUESTION:
- sufficient = true ONLY if it is well-reasoned, internally consistent, directly
  answers the question, and shows no sign of a mistake, guess, or confusion. When
  the task has a verifiable form (a number, a single choice, a definite fact), the
  answer must commit to exactly one and justify it.
- sufficient = false if the reasoning is missing, hand-wavy, self-contradictory,
  hedged ("it could be X or Y"), or reads like a plausible guess.

Be conservative: when in genuine doubt, return sufficient = false. Escalating an
easy question wastes a little money; returning a wrong cheap answer is the failure
this gate exists to prevent.

confidence: your probability (0.0-1.0) that the ANSWER is correct.
reason: one sentence."""


def gate_user(question: str, answer: str) -> str:
    """Build the gate's user content (build-spec §6)."""
    return f"QUESTION:\n{question}\n\nANSWER:\n{answer}"


# Per-benchmark generation system prompts (build-spec §6). The same generation
# prompt is used across all tiers for a given benchmark — the model is the
# variable under test, not the prompt. Each ends with the exact answer-format
# line the objective grader (Appendix A) parses.
GEN_SYSTEM: dict[str, str] = {
    "gsm8k": (
        "Solve the problem. Show brief working, then end with exactly one line: "
        '"The answer is <number>."'
    ),
    "mmlu": (
        "Answer the multiple-choice question. Respond with exactly one line: "
        '"The answer is <A|B|C|D>."'
    ),
}
