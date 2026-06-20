"""Honesty + losing-region acceptance gates (split 14, test 16).

Run with NO key — every check uses the committed sample bundle and the engine's
own pure functions. Exits non-zero if any honesty guarantee is broken:

  1. Honest headline — carries a spread (±/+/-), 'n=', 'frozen split'; never
     an un-negated 'free quality' claim (bundle headlines + README).
  2. The demo CAN show a loss — the committed frontier surfaces a losing-region
     point (cost > always-Opus) with a negative cost-reduction (§8 made visible).
  3. 0 items → N/A end to end — an empty eval reports N/A, never a fake number.
  4. Refusals are surfaced — a refused route round-trips refused=True (not hidden).

    python scripts/acceptance_checks.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO / "api" / "src" / "frugalroute_api" / "data" / "sample_run.json"
README_PATH = REPO / "README.md"

# Import the engine (installed editable; core/ on the path either way).
sys.path.insert(0, str(REPO / "core" / "src"))

from frugalroute import metrics  # noqa: E402
from frugalroute.harness import format_headline, report_from_dict, run_eval  # noqa: E402
from frugalroute.models import (  # noqa: E402
    GateVerdict,
    RouteResult,
    route_result_from_dict,
    route_result_to_dict,
)

# The headlines carry ±/§/→; force UTF-8 stdout so a Windows cp1252 console
# doesn't crash on them (CI/Linux is UTF-8 already).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

_failures: list[str] = []
_passes: list[str] = []


def _ok(name: str, detail: str = "") -> None:
    _passes.append(f"PASS  {name}" + (f" — {detail}" if detail else ""))


def _bad(name: str, detail: str) -> None:
    _failures.append(f"FAIL  {name} — {detail}")


def _load_bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


# --- 1. Honest headline -----------------------------------------------------
def check_honest_headline() -> None:
    bundle = _load_bundle()
    n_test = int((bundle.get("frozen_split") or {}).get("n_test", 0))
    benchmark = bundle.get("benchmark", "gsm8k")
    seen_any = False
    for rdict in bundle["reports"]:
        report = report_from_dict(rdict)
        headline = format_headline(report, benchmark=benchmark, n_test=n_test)
        low = headline.lower()
        if "free quality" in low:
            _bad("honest headline", f"says 'free quality': {headline!r}")
            return
        # The headline for a real (non-empty) report must carry the distributional
        # markers; an N/A report legitimately omits them.
        if "n/a" not in low:
            seen_any = True
            # The engine renders the distributional spread as ASCII '+/-' (a
            # deliberate Windows-console choice, split-03); '±' is the same signal.
            if "±" not in headline and "+/-" not in headline:
                _bad("honest headline", f"no spread marker (±/+/-) in {headline!r}")
                return
            for marker in ("n=", "frozen split"):
                if marker not in headline:
                    _bad("honest headline", f"missing {marker!r} in {headline!r}")
                    return
    if not seen_any:
        _bad("honest headline", "no non-empty report produced a headline")
        return
    # The README must also be honest: any mention of 'free quality' must be a
    # disclaimer (negated), never an affirmative claim. A disclaimer is a *positive*
    # honesty signal; an un-negated claim fails.
    import re

    readme = README_PATH.read_text(encoding="utf-8").lower()
    for m in re.finditer(r"free quality", readme):
        pre = readme[max(0, m.start() - 30) : m.start()]
        if not re.search(r"\b(not|never|no|isn't|without)\b", pre):
            _bad("honest headline", "README makes an un-negated 'free quality' claim")
            return
    _ok(
        "honest headline",
        "spread (±/+/-), n=, frozen split present; 'free quality' only ever disclaimed",
    )


# --- 2. The demo can show a loss (losing region) ----------------------------
def check_losing_region() -> None:
    bundle = _load_bundle()
    for rdict in bundle["reports"]:
        report = report_from_dict(rdict)
        if not report.points:
            continue
        strong_c = report.baselines["always_strong"]["cost"]
        strong_q = report.baselines["always_strong"]["quality"]
        losing = [p for p in report.points if p.cost_usd_per_query > strong_c + 1e-12]
        if not losing:
            continue
        # At a losing point the cost-reduction vs always-strong is negative.
        worst = max(losing, key=lambda p: p.cost_usd_per_query)
        reduction = metrics.cost_reduction(worst.cost_usd_per_query, strong_c)
        if reduction >= 0:
            _bad(
                "losing region",
                f"point cost {worst.cost_usd_per_query:.4f} > strong {strong_c:.4f} "
                f"but cost_reduction {reduction:.3f} is not negative",
            )
            return
        _ok(
            "losing region",
            f"{report.strategy} @ {worst.operating_param:.2f}: "
            f"${worst.cost_usd_per_query:.4f} > always-Opus ${strong_c:.4f} "
            f"(cut {reduction * 100:.0f}%, quality {strong_q:.2f}) — §8 shown, not hidden",
        )
        return
    _bad(
        "losing region",
        "no committed frontier point exceeds always-Opus — the demo can't show a loss",
    )


# --- 3. 0 items → N/A end to end --------------------------------------------
def check_empty_is_na() -> None:
    # n=0 runs ZERO generations → no key needed (split-05 dec d).
    run = run_eval("gsm8k", strategy="cascade", repeats=1, n=0)
    report = next(iter(run.reports.values()))
    if math.isfinite(report.retention_at_target) or math.isfinite(
        report.cost_reduction_at_target
    ):
        _bad(
            "empty → N/A",
            "retention/cost-reduction are finite on a 0-item eval (should be NaN)",
        )
        return
    headline = format_headline(report, benchmark="gsm8k", n_test=0)
    if "N/A" not in headline:
        _bad("empty → N/A", f"headline is not N/A: {headline!r}")
        return
    _ok("empty → N/A", f"0-item eval → NaN metrics, headline {headline!r}")


# --- 4. Refusals are surfaced (never hidden) --------------------------------
def check_refusal_surfaced() -> None:
    refused = RouteResult(
        query="q",
        strategy="cascade",
        tier_used="claude-opus-4-8",
        escalated=True,
        answer="",  # a refusal yields no answer
        correct=None,
        gate=GateVerdict(sufficient=False, confidence=0.0, reason="gate refused"),
        p_strong=None,
        refused=True,
        cost_usd=0.0089,
        latency_s=0.5,
        prompt_version="v1",
    )
    round_tripped = route_result_from_dict(route_result_to_dict(refused))
    if not round_tripped.refused:
        _bad(
            "refusal surfaced",
            "refused flag lost on round-trip (would hide the refusal)",
        )
        return
    if round_tripped.answer != "":
        _bad("refusal surfaced", "a fabricated answer appeared on a refusal")
        return
    _ok(
        "refusal surfaced",
        "refused=True + empty answer round-trip intact (surfaced, not faked)",
    )


def main() -> int:
    for check in (
        check_honest_headline,
        check_losing_region,
        check_empty_is_na,
        check_refusal_surfaced,
    ):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - report, don't crash the gate
            _bad(check.__name__, f"raised {type(exc).__name__}: {exc}")

    print("FrugalRoute acceptance — honesty + losing-region gates")
    print("-" * 60)
    for line in _passes:
        print(line)
    for line in _failures:
        print(line)
    print("-" * 60)
    if _failures:
        print(
            f"ACCEPTANCE CHECKS FAILED ({len(_failures)} of {len(_passes) + len(_failures)})."
        )
        return 1
    print(f"ALL {len(_passes)} ACCEPTANCE CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
