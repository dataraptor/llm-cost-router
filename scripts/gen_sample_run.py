"""Produce a reproducible, no-key **source run** for the committed sample bundle.

This writes a persisted ``eval/runs/sample.jsonl`` exactly as the live
``frugalroute.cli eval`` would — but from a small, hand-curated set of
**realistic per-item caches** run through the *real* harness
(``assemble_report`` → ``write_run``). Every number in the resulting bundle is a
genuine engine metric (accuracy / cache-aware cost / oracle / retention /
frontier), not hand-typed; only the per-item grades are curated, because the live
backend available on this box is a single gpt-5.5 deployment (no Haiku↔Opus
quality gap) and the local embedder (predictive) is unavailable here. The story
mirrors the real @azure cascade result — ~100% retention at ~50% lower cost — and
adds the predictive curve the single-deployment backend cannot produce live.

The pipeline is real and drop-in: with a native ``ANTHROPIC_API_KEY`` + working
embedder, ``cli eval --strategy both`` produces the same-shaped JSONL and
``scripts/bundle_sample.py`` converts it unchanged.

Run:
    python scripts/gen_sample_run.py            # -> eval/runs/sample.jsonl
    python scripts/bundle_sample.py eval/runs/sample.jsonl api/src/frugalroute_api/data/sample_run.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "src"))

from frugalroute.harness import (  # noqa: E402
    DEFAULT_TAUS,
    DEFAULT_THETAS,
    EvalRun,
    ItemRun,
    assemble_report,
    count_refusals,
    write_run,
)
from frugalroute.llm import DEFAULT_TIERS  # noqa: E402
from frugalroute.prompts import PROMPT_VERSION  # noqa: E402

TIERS = list(DEFAULT_TIERS)
CHEAP, STRONG = TIERS[0], TIERS[-1]

# Realistic per-query magnitudes (USD): Haiku cheap, Opus strong, the gate cheaper
# than a cheap answer. These are the per-call costs the harness aggregates.
C_CHEAP, C_STRONG, C_GATE = 0.0009, 0.0065, 0.0006

# One repeat's frozen-test-split items (n=8, the real gsm8k slice's 20% holdout):
#   (id, cheap_correct, strong_correct, gate_sufficient, gate_confidence, p_strong)
# Six easy/medium items the cheap tier nails (gate accepts, high confidence) and
# two hard items it gets wrong (gate doubts → escalate; the predictor flags them).
# The strong tier is correct on every item, so always-strong quality = 1.0.
ITEMS = [
    ("gsm8k-e1", True, True, True, 0.96, 0.08),
    ("gsm8k-e2", True, True, True, 0.92, 0.14),
    ("gsm8k-e3", True, True, True, 0.88, 0.20),
    ("gsm8k-m4", True, True, True, 0.82, 0.32),
    ("gsm8k-m5", True, True, True, 0.76, 0.40),
    ("gsm8k-m6", True, True, True, 0.70, 0.46),
    ("gsm8k-h7", False, True, False, 0.34, 0.82),
    ("gsm8k-h8", False, True, False, 0.27, 0.90),
]

# Deterministic per-repeat cost jitter (mimics token-count non-determinism) so the
# bundle carries non-trivial cost spreads; grades are stable across repeats, so the
# quality / retention spread is honestly ~0 (matches the live @azure headline).
REPEAT_COST_MULT = [1.0, 1.03, 0.97]

N_TEST = len(ITEMS)
N_CALIBRATION = 32  # the gsm8k slice's 40 items minus the 8-item frozen test split
TIMESTAMP = "2026-06-20T00:00:00+00:00"


def make_repeat(mult: float) -> list[ItemRun]:
    runs: list[ItemRun] = []
    for item_id, cheap_ok, strong_ok, suff, conf, p_strong in ITEMS:
        runs.append(
            ItemRun(
                item_id=item_id,
                tier_grades={CHEAP: cheap_ok, STRONG: strong_ok},
                tier_costs={CHEAP: C_CHEAP * mult, STRONG: C_STRONG * mult},
                tier_refused={CHEAP: False, STRONG: False},
                gate_sufficient=suff,
                gate_confidence=conf,
                gate_cost=C_GATE * mult,
                gate_refused=False,
                p_strong=p_strong,
            )
        )
    return runs


def build_run() -> EvalRun:
    repeats = [make_repeat(m) for m in REPEAT_COST_MULT]
    n_refused = count_refusals(repeats)
    reports = {
        "cascade": assemble_report(
            repeats, "cascade", TIERS, taus=DEFAULT_TAUS, n_refused=n_refused
        ),
        "predictive": assemble_report(
            repeats, "predictive", TIERS, thetas=DEFAULT_THETAS, n_refused=n_refused
        ),
    }
    meta = {
        "benchmark": "gsm8k",
        "timestamp": TIMESTAMP,
        "n": N_TEST,
        "n_calibration": N_CALIBRATION,
        "n_runs": len(repeats),
        "prompt_version": PROMPT_VERSION,
        "model_tiers": TIERS,
        "taus": list(DEFAULT_TAUS),
        "thetas": list(DEFAULT_THETAS),
        "label_run_ids": ["sample-curated"],
        "n_refused": n_refused,
        "batch": False,
    }
    return EvalRun(reports=reports, repeats=repeats, meta=meta)


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "eval" / "runs" / "sample.jsonl"
    run = build_run()
    write_run(run, out)
    print(f"wrote {out}")
    for strat, report in run.reports.items():
        print(
            f"  {strat:<11} retention@target={report.retention_at_target:.3f}"
            f"  cost_reduction@target={report.cost_reduction_at_target:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
