"""Command-line interface: ``python -m frugalroute.cli`` (build-spec §13).

Split 03 ships the ``route`` subcommand (cascade strategy). ``train`` (split 04)
and ``eval`` (split 05) are added later behind the same argparse parser — the
subparser registration leaves a clean extension point.

The CLI is a thin presentation layer over :func:`frugalroute.router.route`: it
resolves an example or a raw query, runs the route, and prints either a
human-readable summary or round-trippable JSON. Operational failures (missing
key, an unimplemented strategy) surface as a clear one-line message and a
non-zero exit code — never a raw traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from frugalroute.examples import load_examples
from frugalroute.llm import cheap_tier, strong_tier
from frugalroute.models import RouteResult
from frugalroute.router import route

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
    """One-line cost breakdown, e.g. ``Haiku + gate`` or ``Haiku + gate + Opus``.

    The cheap call always ran; ``gate`` ran iff a verdict is present; the strong
    tier ran iff the route escalated (a cheap refusal escalates with no gate).
    Labels come from the default tier config, which is what the CLI routes with.
    """
    parts: list[str] = [_short(cheap_tier())]
    if result.gate is not None:
        parts.append("gate")
    if result.escalated:
        parts.append(_short(strong_tier()))
    return " + ".join(parts)


def _print_human(result: RouteResult) -> None:
    """Print the human-readable route summary."""
    print(f"Query:     {result.query}")
    print(f"Strategy:  {result.strategy}")
    print(f"Tier used: {result.tier_used}")
    print(f"Escalated: {'yes' if result.escalated else 'no'}")
    if result.gate is not None:
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


def _run_route(args: argparse.Namespace, client: Any) -> int:
    """Execute the ``route`` subcommand. Returns a process exit code."""
    try:
        query, benchmark = _resolve_query(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = route(
            query,
            strategy=args.strategy,
            benchmark=benchmark,
            tau=args.tau,
            client=client,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        # e.g. missing ANTHROPIC_API_KEY — surface the clear message, not a trace.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(route_result_to_dict(result), indent=2))
    else:
        _print_human(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. ``route`` is the only subcommand in split 03."""
    parser = argparse.ArgumentParser(
        prog="frugalroute",
        description="FrugalRoute — route a query through the cost-optimizing cascade.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    route_p = sub.add_parser("route", help="Route a single query (cascade strategy).")
    route_p.add_argument(
        "--strategy",
        choices=["cascade", "predictive"],
        default="cascade",
        help="Routing strategy (default: cascade; predictive arrives in split 04).",
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
    route_p.add_argument("--json", action="store_true", help="Emit the RouteResult as JSON.")
    return parser


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    """CLI entry point. ``client`` is an injectable test seam (default: live).

    Returns a process exit code (0 on success). When invoked as a console script
    or ``python -m frugalroute.cli``, the return value is passed to ``sys.exit``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "route":
        return _run_route(args, client)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover - argparse guards
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
