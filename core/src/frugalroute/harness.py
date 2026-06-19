"""The eval harness — trace the Pareto frontier (build-spec §11, split 05).

This is the shared source of truth behind ``cli eval`` and ``eval/run.py``. It
implements the **run-once-then-re-threshold** rule that makes the whole sweep both
cheap and correct: for each item it generates **each tier exactly once** and runs
the cascade gate **exactly once** (the gate sees the real cheap answer; its verdict
does not depend on τ), caching the per-item results. Every frontier point is then
pure arithmetic over that cache —

- **cascade @ τ:** accept iff ``sufficient ∧ confidence ≥ τ`` → cheap grade and
  ``cost = c_cheap + c_gate``; else strong grade and ``c_cheap + c_gate + c_strong``;
- **predictive @ θ:** strong iff ``p_strong > θ`` else cheap → that tier's cached
  grade + cost.

so the cost is ``items × tiers × R`` generations + ``items × R`` gate calls,
**independent of the grid size**, and the sweep is never confounded by per-call
non-determinism (§9). The live ``route()`` path is not used to build the frontier;
the harness drives generation/gating directly so it can cache and re-threshold.

The pure pieces — re-thresholding, distributional aggregation, report
(de)serialization, and rendering — are no-key testable on synthetic per-item
caches; only :func:`collect_repeat` / :func:`run_eval` touch the live backend.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frugalroute import metrics
from frugalroute.benchmarks import BenchItem, frozen_split, grade, load_benchmark
from frugalroute.gate import gate
from frugalroute.generate import generate
from frugalroute.llm import DEFAULT_TIERS, cost_usd
from frugalroute.models import EvalReport, FrontierPoint
from frugalroute.prompts import PROMPT_VERSION

# --- Operating-point grids (configurable; --quick uses the coarse variants). ---
DEFAULT_TAUS: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_THETAS: list[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
QUICK_TAUS: list[float] = [0.5, 0.8, 1.0]
QUICK_THETAS: list[float] = [0.4, 0.6]

DEFAULT_REPEATS = 3
DEFAULT_TARGET_RETENTION = 0.95

# Batch API discount on the eval generations (build-spec §10): 50% off. Applied to
# the per-call cost; NEVER on the live router path (batch is async).
BATCH_DISCOUNT = 0.5

CASCADE = "cascade"
PREDICTIVE = "predictive"

# A re-threshold function: (runs, tiers, operating_param) -> (quality, cost, escal.).
PointFn = Callable[[Sequence["ItemRun"], Sequence[str], float], tuple[float, float, float]]


# ----------------------------------------------------------------------------
# The per-item cache (one repeat = list[ItemRun]) — re-thresholded for every point
# ----------------------------------------------------------------------------
@dataclass
class ItemRun:
    """One item's cached generation/gate/predict results for a single repeat.

    The per-tier maps hold each tier's graded answer; ``gate_*`` is the single
    cascade gate over the cheap answer (``gate_sufficient is None`` when the cheap
    tier refused, so the gate was skipped and any τ escalates); ``p_strong`` is the
    predictive classifier's P(needs strong) (``None`` when predictive wasn't run).
    """

    item_id: str
    tier_grades: dict[str, bool]
    tier_costs: dict[str, float]
    tier_refused: dict[str, bool]
    gate_sufficient: bool | None = None
    gate_confidence: float = 0.0
    gate_cost: float = 0.0
    gate_refused: bool = False
    p_strong: float | None = None


def cascade_point(
    runs: Sequence[ItemRun], tiers: Sequence[str], tau: float
) -> tuple[float, float, float]:
    """``(quality, mean_cost, escalation_rate)`` of the cascade at threshold ``tau``.

    Pure re-threshold over the cache (mirrors ``router._route_cascade`` exactly):
    a cheap refusal escalates without a gate (``c_cheap + c_strong``); otherwise
    accept iff ``sufficient ∧ confidence ≥ tau`` (``c_cheap + c_gate``) else
    escalate (``c_cheap + c_gate + c_strong``). Empty ``runs`` → all ``nan``.
    """
    cheap, strong = tiers[0], tiers[-1]
    correct: list[bool] = []
    costs: list[float] = []
    escalated: list[bool] = []
    for run in runs:
        if run.tier_refused.get(cheap, False):
            costs.append(run.tier_costs[cheap] + run.tier_costs[strong])
            correct.append(run.tier_grades[strong])
            escalated.append(True)
            continue
        accept = run.gate_sufficient is True and run.gate_confidence >= tau
        if accept:
            costs.append(run.tier_costs[cheap] + run.gate_cost)
            correct.append(run.tier_grades[cheap])
            escalated.append(False)
        else:
            costs.append(run.tier_costs[cheap] + run.gate_cost + run.tier_costs[strong])
            correct.append(run.tier_grades[strong])
            escalated.append(True)
    return metrics.accuracy(correct), metrics.mean_cost(costs), metrics.accuracy(escalated)


def predictive_point(
    runs: Sequence[ItemRun], tiers: Sequence[str], theta: float
) -> tuple[float, float, float]:
    """``(quality, mean_cost, escalation_rate)`` of the predictive router at ``theta``.

    Pure re-threshold: route to strong iff ``p_strong > theta`` else cheap, then
    use that single tier's cached grade + cost (no gate, no double spend). Empty
    ``runs`` → all ``nan``.
    """
    cheap, strong = tiers[0], tiers[-1]
    correct: list[bool] = []
    costs: list[float] = []
    escalated: list[bool] = []
    for run in runs:
        to_strong = run.p_strong is not None and run.p_strong > theta
        chosen = strong if to_strong else cheap
        correct.append(run.tier_grades[chosen])
        costs.append(run.tier_costs[chosen])
        escalated.append(to_strong)
    return metrics.accuracy(correct), metrics.mean_cost(costs), metrics.accuracy(escalated)


def _grades(runs: Sequence[ItemRun]) -> list[dict[str, bool]]:
    return [run.tier_grades for run in runs]


def _costs(runs: Sequence[ItemRun]) -> list[dict[str, float]]:
    return [run.tier_costs for run in runs]


def count_refusals(repeats: Sequence[Sequence[ItemRun]]) -> int:
    """Total refusal events (any tier generation or gate) across all repeats (§17)."""
    total = 0
    for repeat in repeats:
        for run in repeat:
            total += sum(1 for refused in run.tier_refused.values() if refused)
            total += 1 if run.gate_refused else 0
    return total


# ----------------------------------------------------------------------------
# Report assembly (pure) — aggregate the R repeats into one EvalReport
# ----------------------------------------------------------------------------
def assemble_report(
    repeats: Sequence[Sequence[ItemRun]],
    strategy: str,
    tiers: Sequence[str],
    *,
    taus: Sequence[float] | None = None,
    thetas: Sequence[float] | None = None,
    n_refused: int | None = None,
    target_retention: float = DEFAULT_TARGET_RETENTION,
) -> EvalReport:
    """Aggregate R per-item caches into a §7 ``EvalReport`` with mean ± spread.

    Sweeps the operating grid (``taus`` for cascade, ``thetas`` for predictive),
    computes every ``FrontierPoint`` + the four baselines + the oracle per repeat,
    and reports each as **mean ± population-stdev over the R repeats**. The headline
    (``retention_at_target`` / ``cost_reduction_at_target``) is the lowest-cost
    point at/above ``target_retention`` on the mean frontier, with its spread taken
    over the repeats at that same operating point. Pure: no I/O, no key.
    """
    point_fn: PointFn
    if strategy == CASCADE:
        params = list(taus) if taus is not None else DEFAULT_TAUS
        point_fn = cascade_point
    elif strategy == PREDICTIVE:
        params = list(thetas) if thetas is not None else DEFAULT_THETAS
        point_fn = predictive_point
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; expected {CASCADE!r} or {PREDICTIVE!r}.")

    n_items = len(repeats[0]) if repeats else 0

    baselines_per_repeat = [metrics.baselines(_grades(r), _costs(r), tiers) for r in repeats]
    oracle_per_repeat = [metrics.oracle(_grades(r), _costs(r), tiers) for r in repeats]
    baseline_agg = _aggregate_baselines(baselines_per_repeat)
    oracle_agg = _aggregate_oracle(oracle_per_repeat)

    points: list[FrontierPoint] = []
    per_param_repeat: dict[float, list[tuple[float, float]]] = {}
    for param in params:
        qualities: list[float] = []
        param_costs: list[float] = []
        escalations: list[float] = []
        for repeat in repeats:
            quality, cost, escalation = point_fn(repeat, tiers, param)
            qualities.append(quality)
            param_costs.append(cost)
            escalations.append(escalation)
        per_param_repeat[param] = list(zip(qualities, param_costs, strict=True))
        q_mean, q_spread = metrics.mean_spread(qualities)
        c_mean, c_spread = metrics.mean_spread(param_costs)
        esc_mean, _ = metrics.mean_spread(escalations)
        points.append(
            FrontierPoint(
                operating_param=param,
                quality=q_mean,
                quality_spread=q_spread,
                cost_usd_per_query=c_mean,
                cost_spread=c_spread,
                escalation_rate=esc_mean,
                n=n_items,
            )
        )

    strong_quality = baseline_agg["always_strong"]["quality"]
    strong_cost = baseline_agg["always_strong"]["cost"]
    headline = metrics.cost_reduction_at_target(
        points, strong_quality, strong_cost, target_retention
    )
    ret_spread, cr_spread = _headline_spread(headline, per_param_repeat, baselines_per_repeat)

    refused = n_refused if n_refused is not None else count_refusals(repeats)
    return EvalReport(
        strategy=strategy,
        points=points,
        baselines=baseline_agg,
        oracle=oracle_agg,
        retention_at_target=float(headline["retention"]),
        retention_at_target_spread=ret_spread,
        cost_reduction_at_target=float(headline["cost_reduction"]),
        cost_reduction_at_target_spread=cr_spread,
        n_refused=refused,
        prompt_version=PROMPT_VERSION,
        model_tiers=list(tiers),
        n_runs=len(repeats),
    )


def _aggregate_baselines(
    per_repeat: Sequence[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Mean ± spread of each baseline's quality and cost over the repeats."""
    names = ("always_cheap", "always_strong", "random")
    out: dict[str, dict[str, float]] = {}
    for name in names:
        q_mean, q_spread = metrics.mean_spread([b[name]["quality"] for b in per_repeat])
        c_mean, c_spread = metrics.mean_spread([b[name]["cost"] for b in per_repeat])
        out[name] = {
            "quality": q_mean,
            "quality_spread": q_spread,
            "cost": c_mean,
            "cost_spread": c_spread,
        }
    return out


def _aggregate_oracle(per_repeat: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean ± spread of the oracle quality, and the mean oracle cost (the ceiling)."""
    q_mean, q_spread = metrics.mean_spread([o["quality"] for o in per_repeat])
    c_mean, _ = metrics.mean_spread([o["cost"] for o in per_repeat])
    return {"quality": q_mean, "quality_spread": q_spread, "cost": c_mean}


def _headline_spread(
    headline: dict[str, float | bool],
    per_param_repeat: dict[float, list[tuple[float, float]]],
    baselines_per_repeat: Sequence[dict[str, dict[str, float]]],
) -> tuple[float, float]:
    """Spread of retention / cost-reduction over the repeats at the chosen point."""
    param = headline["operating_param"]
    if not isinstance(param, float) or not math.isfinite(param) or param not in per_param_repeat:
        return metrics.NAN, metrics.NAN
    retentions: list[float] = []
    reductions: list[float] = []
    for (quality, cost), baseline in zip(
        per_param_repeat[param], baselines_per_repeat, strict=True
    ):
        strong_q = baseline["always_strong"]["quality"]
        strong_c = baseline["always_strong"]["cost"]
        retentions.append(metrics.retention(quality, strong_q))
        reductions.append(metrics.cost_reduction(cost, strong_c))
    _, ret_spread = metrics.mean_spread([x for x in retentions if math.isfinite(x)])
    _, cr_spread = metrics.mean_spread([x for x in reductions if math.isfinite(x)])
    return ret_spread, cr_spread


# ----------------------------------------------------------------------------
# Generation collection — synchronous, or via the Batch API (50% off, @api)
# ----------------------------------------------------------------------------
@dataclass
class GenRequest:
    """One generation request, keyed by a stable ``custom_id`` for batch results."""

    custom_id: str
    model_id: str
    query: str
    benchmark: str


@dataclass
class GenResult:
    """One generation outcome (cost already discounted in batch mode)."""

    custom_id: str
    model_id: str
    text: str
    refused: bool
    cost_usd: float
    usage: dict[str, int]


@dataclass
class RawBatchItem:
    """A raw batch result the runner returns — keyed by ``custom_id``, any order."""

    custom_id: str
    usage: dict[str, int]
    text: str = ""
    refused: bool = False


BatchRunner = Callable[[Sequence[GenRequest]], Sequence[RawBatchItem]]


def run_generations(
    requests: Sequence[GenRequest],
    *,
    client: Any = None,
    batch: bool = False,
    batch_runner: BatchRunner | None = None,
) -> dict[str, GenResult]:
    """Run a set of generation requests synchronously or through the Batch API.

    Returns ``{custom_id: GenResult}``. In **batch** mode the costs are computed
    from each returned usage and multiplied by :data:`BATCH_DISCOUNT` (the 50%
    Batch-API discount), and results are re-associated **by ``custom_id``**, never
    by position — so a runner that returns items out of order is handled correctly
    (build-spec §10). A ``batch_runner`` is required in batch mode (the live
    Anthropic Batches runner is wired by the caller; tests inject a fake).
    """
    by_id = {req.custom_id: req for req in requests}
    out: dict[str, GenResult] = {}

    if batch:
        if batch_runner is None:
            raise ValueError("batch=True requires a batch_runner.")
        for item in batch_runner(requests):
            req = by_id.get(item.custom_id)
            if req is None:
                raise KeyError(f"Batch result has unknown custom_id {item.custom_id!r}.")
            usage = item.usage
            full = cost_usd(
                req.model_id,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            )
            out[item.custom_id] = GenResult(
                custom_id=item.custom_id,
                model_id=req.model_id,
                text="" if item.refused else item.text,
                refused=item.refused,
                cost_usd=BATCH_DISCOUNT * full,
                usage=usage,
            )
        missing = [req.custom_id for req in requests if req.custom_id not in out]
        if missing:
            raise KeyError(f"Batch run missing results for custom_ids: {missing}.")
        return out

    for req in requests:
        result = generate(client, req.model_id, req.query, req.benchmark)
        out[req.custom_id] = GenResult(
            custom_id=req.custom_id,
            model_id=req.model_id,
            text=result.text,
            refused=result.refused,
            cost_usd=result.cost_usd,
            usage=result.usage,
        )
    return out


def _custom_id(item_id: str, tier: str) -> str:
    return f"{item_id}::{tier}"


def collect_repeat(
    client: Any,
    items: Sequence[BenchItem],
    tiers: Sequence[str],
    benchmark: str,
    *,
    need_gate: bool,
    router: Any = None,
    embedder: Any = None,
    batch: bool = False,
    batch_runner: BatchRunner | None = None,
) -> list[ItemRun]:
    """Collect one repeat's per-item cache (@api): each tier once, gate once.

    Generates every tier for every item (synchronously or batched), grades each
    answer, runs the cascade gate over the cheap answer once when ``need_gate``,
    and records ``p_strong`` from ``router`` when given. This single pass serves the
    baselines, the oracle, **and** both strategies' frontiers (the run-once rule).
    """
    requests = [
        GenRequest(_custom_id(item.id, tier), tier, item.question, benchmark)
        for item in items
        for tier in tiers
    ]
    generated = run_generations(requests, client=client, batch=batch, batch_runner=batch_runner)

    cheap = tiers[0]
    runs: list[ItemRun] = []
    for item in items:
        tier_grades: dict[str, bool] = {}
        tier_costs: dict[str, float] = {}
        tier_refused: dict[str, bool] = {}
        for tier in tiers:
            result = generated[_custom_id(item.id, tier)]
            tier_refused[tier] = result.refused
            tier_costs[tier] = result.cost_usd
            tier_grades[tier] = (not result.refused) and grade(benchmark, result.text, item.gold)

        gate_sufficient: bool | None = None
        gate_confidence = 0.0
        gate_cost = 0.0
        gate_refused = False
        if need_gate:
            cheap_result = generated[_custom_id(item.id, cheap)]
            if not cheap_result.refused:
                outcome = gate(client, item.question, cheap_result.text, gate_model=cheap)
                gate_cost = outcome.cost_usd
                gate_refused = outcome.refused
                gate_sufficient = outcome.verdict.sufficient
                gate_confidence = outcome.verdict.confidence

        p_strong: float | None = None
        if router is not None:
            _tier, p_strong = router.predict_tier(item.question, embedder=embedder)

        runs.append(
            ItemRun(
                item_id=item.id,
                tier_grades=tier_grades,
                tier_costs=tier_costs,
                tier_refused=tier_refused,
                gate_sufficient=gate_sufficient,
                gate_confidence=gate_confidence,
                gate_cost=gate_cost,
                gate_refused=gate_refused,
                p_strong=p_strong,
            )
        )
    return runs


# ----------------------------------------------------------------------------
# The full eval orchestrator (@api)
# ----------------------------------------------------------------------------
@dataclass
class EvalRun:
    """A full eval run: per-strategy reports + the per-item provenance cache."""

    reports: dict[str, EvalReport]
    repeats: list[list[ItemRun]]
    meta: dict[str, Any]


def run_eval(
    benchmark: str,
    *,
    strategy: str = "both",
    repeats: int = DEFAULT_REPEATS,
    taus: Sequence[float] | None = None,
    thetas: Sequence[float] | None = None,
    client: Any = None,
    embedder: Any = None,
    router: Any = None,
    n: int | None = None,
    tiers: Sequence[str] = DEFAULT_TIERS,
    batch: bool = False,
    batch_runner: BatchRunner | None = None,
    data_dir: str | Path | None = None,
    timestamp: str = "unstamped",
) -> EvalRun:
    """Run the eval on the **frozen test split** and assemble the report(s) (@api).

    Loads ``benchmark``, holds out the frozen test split, and (for the predictive
    strategy) trains a router on the **calibration** split if one is not supplied —
    leakage-free. Collects ``repeats`` per-item caches over the test split, then
    assembles one :class:`~frugalroute.models.EvalReport` per requested strategy
    (mean ± spread). ``timestamp`` is injected (not read from the clock) so runs are
    reproducible/testable. The data collection is the only @api part.
    """
    wants_cascade = strategy in (CASCADE, "both")
    wants_predictive = strategy in (PREDICTIVE, "both")
    tiers = list(tiers)

    items = load_benchmark(benchmark, n=n, data_dir=data_dir)
    calibration, test = frozen_split(items)

    label_run_ids: list[str] = []
    if wants_predictive and router is None:
        router = _train_router(client, calibration, tiers, benchmark, embedder=embedder)
    if router is not None:
        label_run_ids = list(getattr(router, "label_run_ids", []))

    repeats_cache: list[list[ItemRun]] = []
    for _ in range(repeats):
        repeats_cache.append(
            collect_repeat(
                client,
                test,
                tiers,
                benchmark,
                need_gate=wants_cascade,
                router=router if wants_predictive else None,
                embedder=embedder,
                batch=batch,
                batch_runner=batch_runner,
            )
        )

    n_refused = count_refusals(repeats_cache)
    reports: dict[str, EvalReport] = {}
    if wants_cascade:
        reports[CASCADE] = assemble_report(
            repeats_cache, CASCADE, tiers, taus=taus, n_refused=n_refused
        )
    if wants_predictive:
        reports[PREDICTIVE] = assemble_report(
            repeats_cache, PREDICTIVE, tiers, thetas=thetas, n_refused=n_refused
        )

    meta = {
        "benchmark": benchmark,
        "timestamp": timestamp,
        "n": len(test),
        "n_calibration": len(calibration),
        "n_runs": repeats,
        "prompt_version": PROMPT_VERSION,
        "model_tiers": tiers,
        "taus": list(taus) if taus is not None else DEFAULT_TAUS,
        "thetas": list(thetas) if thetas is not None else DEFAULT_THETAS,
        "label_run_ids": label_run_ids,
        "n_refused": n_refused,
        "batch": batch,
    }
    return EvalRun(reports=reports, repeats=repeats_cache, meta=meta)


def _train_router(
    client: Any,
    calibration: Sequence[BenchItem],
    tiers: Sequence[str],
    benchmark: str,
    *,
    embedder: Any = None,
) -> Any:
    """Train a predictive router on the calibration split (leakage-free, @api).

    Imported lazily so the no-key path (which injects a router or skips predictive)
    never pulls sklearn / the embedder.
    """
    from frugalroute.classifier import (
        DEFAULT_EMBEDDER,
        PredictiveRouter,
        generate_labels,
        train,
    )
    from frugalroute.embed import embed

    label_runs = generate_labels(client, calibration, tiers, benchmark)
    labels = [run.label for run in label_runs]
    embeddings = embed([item.question for item in calibration], embedder=embedder)
    clf = train(embeddings, labels, tiers)
    return PredictiveRouter(
        clf=clf,
        tiers=list(tiers),
        embedder_name=DEFAULT_EMBEDDER,
        prompt_version=PROMPT_VERSION,
        label_run_ids=sorted({run.run_id for run in label_runs}),
    )


# ----------------------------------------------------------------------------
# Persistence — gitignored eval/runs/<ts>.jsonl (round-trips, with provenance)
# ----------------------------------------------------------------------------
def report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Serialize an ``EvalReport`` to a JSON-ready dict (all §7 + spread fields)."""
    return {
        "strategy": report.strategy,
        "points": [
            {
                "operating_param": p.operating_param,
                "quality": p.quality,
                "quality_spread": p.quality_spread,
                "cost_usd_per_query": p.cost_usd_per_query,
                "cost_spread": p.cost_spread,
                "escalation_rate": p.escalation_rate,
                "n": p.n,
            }
            for p in report.points
        ],
        "baselines": report.baselines,
        "oracle": report.oracle,
        "retention_at_target": report.retention_at_target,
        "retention_at_target_spread": report.retention_at_target_spread,
        "cost_reduction_at_target": report.cost_reduction_at_target,
        "cost_reduction_at_target_spread": report.cost_reduction_at_target_spread,
        "n_refused": report.n_refused,
        "prompt_version": report.prompt_version,
        "model_tiers": report.model_tiers,
        "n_runs": report.n_runs,
    }


def report_from_dict(data: dict[str, Any]) -> EvalReport:
    """Rebuild an ``EvalReport`` from :func:`report_to_dict` output."""
    points = [FrontierPoint(**point) for point in data["points"]]
    return EvalReport(
        strategy=data["strategy"],
        points=points,
        baselines=data["baselines"],
        oracle=data["oracle"],
        retention_at_target=data["retention_at_target"],
        retention_at_target_spread=data["retention_at_target_spread"],
        cost_reduction_at_target=data["cost_reduction_at_target"],
        cost_reduction_at_target_spread=data["cost_reduction_at_target_spread"],
        n_refused=data["n_refused"],
        prompt_version=data["prompt_version"],
        model_tiers=data["model_tiers"],
        n_runs=data["n_runs"],
    )


def write_run(run: EvalRun, path: str | Path) -> Path:
    """Persist an ``EvalRun`` to ``path`` as JSONL (meta + reports + per-item rows).

    Each line is a typed record: one ``meta`` line, one ``report`` line per
    strategy, then one ``item`` line per (repeat, item, tier) carrying the grade +
    cost (the reproducibility provenance, build-spec §11). Creates parent dirs.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[dict[str, Any]] = [{"type": "meta", **run.meta}]
    for report in run.reports.values():
        lines.append({"type": "report", **report_to_dict(report)})
    for repeat_idx, repeat in enumerate(run.repeats):
        for item in repeat:
            for tier, correct in item.tier_grades.items():
                lines.append(
                    {
                        "type": "item",
                        "repeat": repeat_idx,
                        "item_id": item.item_id,
                        "tier": tier,
                        "grade": correct,
                        "cost": item.tier_costs[tier],
                        "refused": item.tier_refused[tier],
                    }
                )
    with target.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    return target


def read_run(path: str | Path) -> dict[str, Any]:
    """Read a persisted run JSONL into ``{meta, reports, items}`` (round-trips writes)."""
    meta: dict[str, Any] = {}
    reports: dict[str, EvalReport] = {}
    items: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            kind = record.pop("type")
            if kind == "meta":
                meta = record
            elif kind == "report":
                report = report_from_dict(record)
                reports[report.strategy] = report
            elif kind == "item":
                items.append(record)
    return {"meta": meta, "reports": reports, "items": items}


# ----------------------------------------------------------------------------
# Rendering (ASCII-only) — the frontier table, leaderboard, and honest headline
# ----------------------------------------------------------------------------
_SHORT_NAMES: dict[str, str] = {
    "claude-haiku-4-5": "Haiku",
    "claude-sonnet-4-6": "Sonnet",
    "claude-opus-4-8": "Opus",
    "gpt-5.5": "gpt-5.5",
}


def _short(model_id: str) -> str:
    return _SHORT_NAMES.get(model_id, model_id)


def _pct(value: float) -> str:
    """Percent with 1 decimal, or ``N/A`` for a nan sentinel (§17)."""
    return "N/A" if not math.isfinite(value) else f"{value * 100:.1f}%"


def _usd(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"${value:.6f}"


def _pm(value: float) -> str:
    """A ``+/-spread`` suffix (ASCII), or empty when the spread is nan."""
    return "" if not math.isfinite(value) else f" +/-{value * 100:.1f}%"


def format_headline(report: EvalReport, *, benchmark: str, n_test: int) -> str:
    """The one honest headline line (never 'free quality'; N/A when undefined)."""
    strong = _short(report.model_tiers[-1])
    ret = report.retention_at_target
    cut = report.cost_reduction_at_target
    if not math.isfinite(ret) or not math.isfinite(cut):
        return (
            f"FrugalRoute ({report.strategy}) on {benchmark}: N/A "
            f"(no operating point / empty split, n={n_test})."
        )
    param = "tau" if report.strategy == CASCADE else "theta"
    point = _headline_point(report)
    op = "?" if point is None else f"{point.operating_param:.2f}"
    ret_str = f"{_pct(ret)}{_pm(report.retention_at_target_spread)}"
    cut_str = f"{_pct(cut)}{_pm(report.cost_reduction_at_target_spread)}"
    return (
        f"FrugalRoute ({report.strategy}) retains {ret_str} of {strong} accuracy "
        f"at {cut_str} lower cost "
        f"(n={n_test}, frozen split, {report.strategy} @ {param}={op})."
    )


def _headline_point(report: EvalReport) -> FrontierPoint | None:
    """The frontier point the headline was taken from (lowest-cost ≥ target)."""
    strong_q = report.baselines["always_strong"]["quality"]
    strong_c = report.baselines["always_strong"]["cost"]
    head = metrics.cost_reduction_at_target(
        report.points, strong_q, strong_c, DEFAULT_TARGET_RETENTION
    )
    param = head["operating_param"]
    if not isinstance(param, float) or not math.isfinite(param):
        return None
    for point in report.points:
        if point.operating_param == param:
            return point
    return None


def format_frontier_table(report: EvalReport) -> str:
    """The per-operating-point frontier table for one strategy (ASCII)."""
    param = "tau" if report.strategy == CASCADE else "theta"
    header = f"  {param:>5}  {'quality':>16}  {'$/query':>10}  {'escal.':>7}  {'n':>4}"
    rows = [f"Frontier ({report.strategy}):", header]
    for point in metrics.frontier_points(report.points):
        quality = f"{_pct(point.quality)}{_pm(point.quality_spread)}"
        rows.append(
            f"  {point.operating_param:>5.2f}  {quality:>16}  "
            f"{_usd(point.cost_usd_per_query):>10}  {_pct(point.escalation_rate):>7}  {point.n:>4}"
        )
    return "\n".join(rows)


def format_leaderboard(reports: dict[str, EvalReport], tiers: Sequence[str]) -> str:
    """The §14-order leaderboard across baselines, both strategies, and the oracle."""
    any_report = next(iter(reports.values()))
    base = any_report.baselines
    strong_q = base["always_strong"]["quality"]
    strong_c = base["always_strong"]["cost"]

    header = (
        f"  {'method':<26}  {'quality':>16}  {'$/query':>10}  {'retention':>10}  {'cost-cut':>10}"
    )
    rows = ["Leaderboard:", header]

    def line(label: str, quality: float, q_spread: float, cost: float, ret: str, cut: str) -> str:
        q = f"{_pct(quality)}{_pm(q_spread)}"
        return f"  {label:<26}  {q:>16}  {_usd(cost):>10}  {ret:>10}  {cut:>10}"

    for name, label in (
        ("always_cheap", f"always-cheap ({_short(tiers[0])})"),
        ("always_strong", f"always-strong ({_short(tiers[-1])})"),
        ("random", "random"),
    ):
        entry = base[name]
        ret = _pct(metrics.retention(entry["quality"], strong_q))
        cut = _pct(metrics.cost_reduction(entry["cost"], strong_c))
        rows.append(line(label, entry["quality"], entry["quality_spread"], entry["cost"], ret, cut))

    for strategy in (CASCADE, PREDICTIVE):
        report = reports.get(strategy)
        if report is None:
            continue
        point = _headline_point(report)
        quality = point.quality if point is not None else metrics.NAN
        q_spread = point.quality_spread if point is not None else metrics.NAN
        cost = point.cost_usd_per_query if point is not None else metrics.NAN
        rows.append(
            line(
                f"FrugalRoute-{strategy}",
                quality,
                q_spread,
                cost,
                _pct(report.retention_at_target),
                _pct(report.cost_reduction_at_target),
            )
        )

    oracle = any_report.oracle
    rows.append(
        line(
            "oracle (ceiling)",
            oracle["quality"],
            oracle["quality_spread"],
            oracle["cost"],
            "-",
            "-",
        )
    )
    return "\n".join(rows)


def format_report(run: EvalRun) -> str:
    """The full ``cli eval`` output: frontier table(s) + leaderboard + headline(s)."""
    benchmark = str(run.meta.get("benchmark", "?"))
    n_test = int(run.meta.get("n", 0))
    blocks: list[str] = []
    for report in run.reports.values():
        blocks.append(format_frontier_table(report))
    blocks.append(format_leaderboard(run.reports, list(run.meta.get("model_tiers", DEFAULT_TIERS))))
    blocks.append(f"Frozen test split: n={n_test} (small - interpret the +/- intervals as wide).")
    refused = int(run.meta.get("n_refused", 0))
    if refused:
        blocks.append(f"n_refused: {refused} (refusal events surfaced during the run).")
    for report in run.reports.values():
        blocks.append(format_headline(report, benchmark=benchmark, n_test=n_test))
    return "\n\n".join(blocks)
