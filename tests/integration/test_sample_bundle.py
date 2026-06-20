"""Split-10 tests 5-6 + 9: the committed sample bundle and the README (no key).

Tests 5-6 assert the committed sample (what `/api/eval/sample` serves with no
key/network) is a genuine harness product: it satisfies the §06 bundle + §7
``EvalReport`` schemas, carries full provenance, and tells an honest cost-quality
story (FrugalRoute up-and-left of the random baseline; an operating point at or
above the target retention; non-trivial spread; a visible losing region).

Test 9 is the README check: the honest-headline guard + leaderboard numbers that
match the committed sample.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from frugalroute.harness import report_from_dict, report_to_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_SAMPLE = (
    REPO_ROOT / "api" / "src" / "frugalroute_api" / "data" / "sample_run.json"
)
README = REPO_ROOT / "README.md"

BUNDLE_KEYS = {"reports", "benchmark", "frozen_split", "generated_at"}
EVAL_REPORT_KEYS = {
    "strategy",
    "points",
    "baselines",
    "oracle",
    "retention_at_target",
    "retention_at_target_spread",
    "cost_reduction_at_target",
    "cost_reduction_at_target_spread",
    "n_refused",
    "prompt_version",
    "model_tiers",
    "n_runs",
}
POINT_KEYS = {
    "operating_param",
    "quality",
    "quality_spread",
    "cost_usd_per_query",
    "cost_spread",
    "escalation_rate",
    "n",
}


@pytest.fixture(scope="module")
def bundle() -> dict:
    return json.loads(COMMITTED_SAMPLE.read_text(encoding="utf-8"))


# --- Test 5: schema + provenance --------------------------------------------
def test_committed_sample_exists() -> None:
    assert COMMITTED_SAMPLE.exists(), f"committed sample missing at {COMMITTED_SAMPLE}"


def test_bundle_satisfies_contract(bundle: dict) -> None:
    assert BUNDLE_KEYS <= set(bundle)
    strategies = {r["strategy"] for r in bundle["reports"]}
    assert {"cascade", "predictive"} <= strategies, (
        "both curves required for the Frontier"
    )
    fs = bundle["frozen_split"]
    assert {"n_test", "n_calibration", "small_n"} <= set(fs)
    assert fs["n_test"] > 0


def test_reports_satisfy_eval_report_schema(bundle: dict) -> None:
    for report in bundle["reports"]:
        assert EVAL_REPORT_KEYS <= set(report)
        for point in report["points"]:
            assert POINT_KEYS <= set(point)
        assert {"always_cheap", "always_strong", "random"} <= set(report["baselines"])
        assert {"quality", "cost"} <= set(report["oracle"])
        # Round-trips losslessly through the engine's own (de)serializers (no drift).
        assert report_to_dict(report_from_dict(report)) == report


def test_provenance_complete(bundle: dict) -> None:
    # Build-spec §11: a real frozen run records what produced it.
    assert bundle.get("run_id"), "missing run_id provenance"
    assert bundle.get("prompt_version"), "missing prompt_version provenance"
    assert bundle.get("generated_at"), "missing generated_at"
    for report in bundle["reports"]:
        assert report["prompt_version"]
        assert report["model_tiers"]
        assert report["n_runs"] >= 1


# --- Test 6: honest cost-quality story --------------------------------------
def test_frontier_up_and_left_of_random(bundle: dict) -> None:
    """At least one FrugalRoute point beats the random baseline (higher quality,
    lower cost) — the whole point of the demo."""
    for report in bundle["reports"]:
        random = report["baselines"]["random"]
        up_and_left = [
            p
            for p in report["points"]
            if p["quality"] >= random["quality"]
            and p["cost_usd_per_query"] < random["cost"]
        ]
        assert up_and_left, f"{report['strategy']} never beats random — no story"


def test_operating_point_meets_target_or_flagged(bundle: dict) -> None:
    """The headline operating point clears the 95% retention target (or the run
    honestly reports the closest point it could reach)."""
    cascade = next(r for r in bundle["reports"] if r["strategy"] == "cascade")
    ret = cascade["retention_at_target"]
    # A real float (never faked); meets target or is honestly the best reachable.
    assert isinstance(ret, float)
    assert ret >= 0.95 or ret == max(p["quality"] for p in cascade["points"])


def test_spread_present(bundle: dict) -> None:
    """Distributional reporting (§9): cost varies across the R repeats → spread > 0."""
    cascade = next(r for r in bundle["reports"] if r["strategy"] == "cascade")
    assert cascade["n_runs"] >= 2
    assert any(p["cost_spread"] > 0 for p in cascade["points"]), (
        "no cost spread — not distributional"
    )


def test_losing_region_is_representable(bundle: dict) -> None:
    """The cascade can cost MORE than always-strong at an aggressive threshold — the
    honest break-even cliff (§8) must be visible, not hidden."""
    cascade = next(r for r in bundle["reports"] if r["strategy"] == "cascade")
    strong_cost = cascade["baselines"]["always_strong"]["cost"]
    assert any(p["cost_usd_per_query"] > strong_cost for p in cascade["points"]), (
        "no point above always-strong cost — the losing region is hidden"
    )


# --- Test 9: README honesty + numbers ---------------------------------------
def test_readme_exists_and_is_honest() -> None:
    assert README.exists(), "root README.md missing"
    text = README.read_text(encoding="utf-8")
    low = text.lower()
    # The honest headline must carry distributional language.
    assert "frozen split" in low
    assert "±" in text
    assert re.search(r"n\s*=\s*\d", text), "README headline missing n="
    # Never *claims* free quality — it explicitly rejects the free-lunch framing.
    assert 'not "free quality"' in low or "free lunch" in low, (
        "README must disclaim free quality"
    )
    assert not re.search(r"(?<!not )(?<!never )free quality (?:at|on|,|\.)", low), (
        "README appears to claim free quality"
    )


def test_readme_sections_present() -> None:
    low = README.read_text(encoding="utf-8").lower()
    for section in [
        "architecture",
        "leaderboard",
        "break-even",
        "limitation",
        "quickstart",
        "provenance",
    ]:
        assert section in low, f"README missing the {section!r} section"


def test_readme_screenshot_asset_present() -> None:
    text = README.read_text(encoding="utf-8")
    refs = re.findall(r"\]\(([^)]+\.(?:png|svg|jpg|jpeg))\)", text)
    assert refs, "README references no screenshot/image asset"
    assert any((REPO_ROOT / ref).exists() for ref in refs), (
        f"no screenshot asset found on disk ({refs})"
    )


def test_readme_leaderboard_numbers_match_sample() -> None:
    """The README's leaderboard $/query figures are the committed sample's, to the
    cent-of-a-milli — so the doc and the served proof can never disagree."""
    bundle = json.loads(COMMITTED_SAMPLE.read_text(encoding="utf-8"))
    cascade = next(r for r in bundle["reports"] if r["strategy"] == "cascade")
    text = README.read_text(encoding="utf-8")
    for key in ("always_cheap", "always_strong", "random"):
        cost = cascade["baselines"][key]["cost"]
        assert f"{cost:.4f}" in text, f"README missing {key} cost ${cost:.4f}"
    oracle_cost = cascade["oracle"]["cost"]
    assert f"{oracle_cost:.4f}" in text, (
        f"README missing oracle cost ${oracle_cost:.4f}"
    )
