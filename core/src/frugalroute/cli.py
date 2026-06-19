"""Command-line interface: ``python -m frugalroute.cli`` (build-spec §13).

Subcommands:
  * ``route`` (split 03) — route one query via the cascade, or (split 04) via a
    trained predictive router.
  * ``train`` (split 04) — generate labels (@api), embed, fit a classifier, and
    save a :class:`~frugalroute.classifier.PredictiveRouter`.
  * ``eval`` (split 05) extends the same parser later.

The CLI is a thin presentation layer over :func:`frugalroute.router.route` and the
classifier helpers. Operational failures (missing key, a missing router/model,
a bad example/benchmark) surface as a clear one-line message and a non-zero exit
code — never a raw traceback. ``client`` and ``embedder`` are injectable test
seams (default: live backend / local embedder).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from typing import Any

from frugalroute.benchmarks import frozen_split, load_benchmark
from frugalroute.classifier import (
    DEFAULT_EMBEDDER,
    PredictiveRouter,
    generate_labels,
    load_router,
    save_router,
)
from frugalroute.classifier import train as train_classifier
from frugalroute.embed import embed
from frugalroute.examples import load_examples
from frugalroute.llm import DEFAULT_TIERS, cheap_tier, get_client, strong_tier
from frugalroute.models import RouteResult
from frugalroute.prompts import PROMPT_VERSION
from frugalroute.router import CASCADE, PREDICTIVE, route

# Friendly short names for the cost-breakdown line, keyed by model ID.
_SHORT_NAMES: dict[str, str] = {
    "claude-haiku-4-5": "Haiku",
    "claude-sonnet-4-6": "Sonnet",
    "claude-opus-4-8": "Opus",
    "gpt-5.5": "gpt-5.5",
}


def _short(model_id: str) -> str:
    """A short display name for a model ID (falls back to the ID itself)."""
    return _SHORT_NAMES.get(model_id, model_id)


def route_result_to_dict(result: RouteResult) -> dict[str, Any]:
    """Serialize a ``RouteResult`` to a JSON-ready dict (gate → plain dict)."""
    data = asdict(result)
    data["gate"] = result.gate.model_dump() if result.gate is not None else None
    return data


def _cost_breakdown(result: RouteResult) -> str:
    """One-line cost breakdown of the calls actually made.

    Cascade: ``Haiku``, ``Haiku + gate``, ``Haiku + gate + Opus`` (or ``Haiku +
    Opus`` on a cheap refusal that skipped the gate). Predictive: just the single
    tier that ran (no cheap call, no gate).
    """
    if result.strategy == PREDICTIVE:
        return _short(result.tier_used)
    parts: list[str] = [_short(cheap_tier())]
    if result.gate is not None:
        parts.append("gate")
    if result.escalated:
        parts.append(_short(strong_tier()))
    return " + ".join(parts)


def _print_human(result: RouteResult, *, theta: float | None = None) -> None:
    """Print the human-readable route summary (branching on strategy)."""
    print(f"Query:     {result.query}")
    print(f"Strategy:  {result.strategy}")
    print(f"Tier used: {result.tier_used}")
    print(f"Escalated: {'yes' if result.escalated else 'no'}")
    if result.strategy == PREDICTIVE:
        threshold = 0.5 if theta is None else theta
        margin = result.p_strong if result.p_strong is not None else float("nan")
        decision = "strong" if result.tier_used == strong_tier() else "cheap"
        print(f"Decision:  p_strong={margin:.2f} vs theta={threshold:.2f} -> {decision}")
    elif result.gate is not None:
        g = result.gate
        print(f"Gate:      sufficient={g.sufficient} confidence={g.confidence:.2f} - {g.reason}")
    else:
        print("Gate:      (skipped)")
    if result.refused:
        print("Refused:   yes (a tier returned a refusal - answer surfaced as-is)")
    print(f"Answer:    {result.answer!r}")
    print(f"Cost:      ${result.cost_usd:.6f}")
    print(f"Latency:   {result.latency_s:.3f}s")
    print(f"Breakdown: = {_cost_breakdown(result)}")


def _resolve_query(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve ``(query, benchmark)`` from ``--example ID`` or ``--query TEXT``.

    ``--example`` takes the example's own benchmark/query; ``--query`` uses the
    ``--benchmark`` flag. Raises ``ValueError`` with a clear message on a bad id.
    """
    if args.example is not None:
        for item in load_examples():
            if item["id"] == args.example:
                return item["query"], item["benchmark"]
        known = ", ".join(item["id"] for item in load_examples())
        raise ValueError(f"Unknown example id {args.example!r}. Known ids: {known}.")
    return args.query, args.benchmark


def _run_route(args: argparse.Namespace, client: Any, embedder: Any) -> int:
    """Execute the ``route`` subcommand. Returns a process exit code."""
    try:
        query, benchmark = _resolve_query(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    router: PredictiveRouter | None = None
    if args.strategy == PREDICTIVE:
        if not args.model:
            print(
                "error: --strategy predictive requires --model PATH (a trained router).",
                file=sys.stderr,
            )
            return 2
        try:
            router = load_router(args.model)
        except (FileNotFoundError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        result = route(
            query,
            strategy=args.strategy,
            benchmark=benchmark,
            tau=args.tau,
            theta=args.theta,
            client=client,
            router=router,
            embedder=embedder,
        )
    except ValueError as exc:
        # Bad strategy / missing router / degenerate prediction — user error.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        # e.g. missing ANTHROPIC_API_KEY — surface the clear message, not a trace.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(route_result_to_dict(result), indent=2))
    else:
        _print_human(result, theta=args.theta)
    return 0


def _run_train(args: argparse.Namespace, client: Any, embedder: Any) -> int:
    """Execute the ``train`` subcommand (@api + local embedder). Exit code."""
    try:
        items = load_benchmark(args.benchmark, n=args.n)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Train only on the calibration side of the frozen split (leakage-free: the
    # test side is held out for the eval in split 05).
    calibration, _test = frozen_split(items)
    if not calibration:
        print("error: no calibration items to train on (benchmark too small).", file=sys.stderr)
        return 2

    tiers = list(DEFAULT_TIERS)
    if client is None:
        try:
            client = get_client()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    label_runs = generate_labels(client, calibration, tiers, args.benchmark)
    labels = [run.label for run in label_runs]
    embeddings = embed([item.question for item in calibration], embedder=embedder)
    clf = train_classifier(embeddings, labels, tiers, kind=args.kind)
    run_ids = sorted({run.run_id for run in label_runs})
    router = PredictiveRouter(
        clf=clf,
        tiers=tiers,
        embedder_name=DEFAULT_EMBEDDER,
        prompt_version=PROMPT_VERSION,
        label_run_ids=run_ids,
    )

    out = args.out or f"models/{args.benchmark}.joblib"
    save_router(router, out)

    distribution = Counter(labels)
    dist_str = " ".join(f"{tier}={distribution.get(tier, 0)}" for tier in tiers)
    print(f"Trained predictive router on {len(calibration)} {args.benchmark} items.")
    print(f"Classifier: {router.clf_kind}")
    print(f"Label distribution: {dist_str}")
    print(f"Label runs: {', '.join(run_ids)}")
    print(f"Saved router to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser (``route`` + ``train``; ``eval`` added in split 05)."""
    parser = argparse.ArgumentParser(
        prog="frugalroute",
        description="FrugalRoute - route a query through the cost-optimizing router.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    route_p = sub.add_parser("route", help="Route a single query (cascade or predictive).")
    route_p.add_argument(
        "--strategy",
        choices=[CASCADE, PREDICTIVE],
        default=CASCADE,
        help="Routing strategy (default: cascade).",
    )
    target = route_p.add_mutually_exclusive_group(required=True)
    target.add_argument("--example", help="Run a bundled example by id (e.g. gsm8k-1142).")
    target.add_argument("--query", help="Run an arbitrary query string.")
    route_p.add_argument(
        "--benchmark",
        choices=["gsm8k", "mmlu"],
        default="gsm8k",
        help="Benchmark prompt to use for --query (ignored for --example).",
    )
    route_p.add_argument(
        "--tau", type=float, default=0.8, help="Cascade acceptance threshold tau (0..1)."
    )
    route_p.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Predictive decision threshold theta (route to strong iff p_strong > theta).",
    )
    route_p.add_argument(
        "--model", help="Path to a trained predictive router (required for --strategy predictive)."
    )
    route_p.add_argument("--json", action="store_true", help="Emit the RouteResult as JSON.")

    train_p = sub.add_parser("train", help="Train and save a predictive router (needs a key).")
    train_p.add_argument(
        "--benchmark", choices=["gsm8k", "mmlu"], required=True, help="Benchmark to train on."
    )
    train_p.add_argument("--n", type=int, default=None, help="Cap the number of items loaded.")
    train_p.add_argument(
        "--kind",
        choices=["logreg", "knn"],
        default="logreg",
        help="Classifier kind (default: logreg; knn is the §19-E fallback).",
    )
    train_p.add_argument(
        "--out", help="Output path for the router (default: models/<benchmark>.joblib)."
    )
    return parser


def main(argv: list[str] | None = None, *, client: Any = None, embedder: Any = None) -> int:
    """CLI entry point. ``client`` and ``embedder`` are injectable test seams.

    Returns a process exit code (0 on success). When invoked as a console script
    or ``python -m frugalroute.cli``, the return value is passed to ``sys.exit``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "route":
        return _run_route(args, client, embedder)
    if args.command == "train":
        return _run_train(args, client, embedder)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover - argparse guards
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
