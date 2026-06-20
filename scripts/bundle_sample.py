"""Convert a persisted eval run (``eval/runs/*.jsonl``) into the committed sample
bundle that ``GET /api/eval/sample`` serves (split-10 §1).

The bundle shape is the frozen split-06 contract:

    {"reports": [<cascade>, <predictive>], "benchmark": str,
     "frozen_split": {"n_test", "n_calibration", "small_n"},
     "generated_at": str, "run_id": str, "prompt_version": str}

Every number is read straight back from the harness's own persisted run (via
``harness.read_run`` → ``report_to_dict``), so the served shape can never drift
from the engine's §7 ``EvalReport``. This is the spec's stage-2 converter:

    python -m frugalroute.cli eval --strategy both --benchmark gsm8k --out eval/runs/sample.jsonl
    python scripts/bundle_sample.py eval/runs/sample.jsonl api/src/frugalroute_api/data/sample_run.json

Run with no key — it only reads a finished run. The run itself (stage 1) is the
only @api part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Make `frugalroute` importable when run straight from the repo (no install needed
# beyond `pip install -e core`); harmless if it is already installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "src"))

from frugalroute.harness import read_run, report_to_dict  # noqa: E402

# Order the bundle's reports cascade-first, predictive-second (the Frontier draws
# the cascade curve solid and the predictive curve dashed in that order).
_STRATEGY_ORDER = {"cascade": 0, "predictive": 1}

# Mirrors api.config.SMALL_N_THRESHOLD (kept local so this script has no api dep).
_SMALL_N_THRESHOLD = 30


def _run_id(meta: dict[str, Any]) -> str:
    """A short, deterministic id for the run (provenance, build-spec §11).

    Hashed over the reproducibility-defining fields so the same run always yields
    the same id and a different prompt/benchmark/split yields a different one.
    """
    key = json.dumps(
        {
            "benchmark": meta.get("benchmark"),
            "prompt_version": meta.get("prompt_version"),
            "model_tiers": meta.get("model_tiers"),
            "n": meta.get("n"),
            "n_runs": meta.get("n_runs"),
            "timestamp": meta.get("timestamp"),
        },
        sort_keys=True,
    )
    return "run-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def build_bundle(
    run_path: str | Path, *, generated_at: str | None = None
) -> dict[str, Any]:
    """Read a persisted run and assemble the ``/api/eval/sample`` bundle dict."""
    run = read_run(run_path)
    meta = run["meta"]
    reports = run["reports"]
    if not reports:
        raise ValueError(f"{run_path} carries no reports — nothing to bundle.")

    ordered = sorted(
        reports.values(), key=lambda r: _STRATEGY_ORDER.get(r.strategy, 99)
    )

    n_test = int(meta.get("n", 0))
    n_calibration = int(meta.get("n_calibration", 0))
    return {
        "reports": [report_to_dict(r) for r in ordered],
        "benchmark": meta.get("benchmark"),
        "frozen_split": {
            "n_test": n_test,
            "n_calibration": n_calibration,
            "small_n": n_test < _SMALL_N_THRESHOLD,
        },
        "generated_at": generated_at or meta.get("timestamp"),
        "run_id": _run_id(meta),
        "prompt_version": meta.get("prompt_version"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run", help="persisted eval run JSONL (e.g. eval/runs/sample.jsonl)"
    )
    parser.add_argument(
        "out",
        help="committed bundle path (e.g. api/src/frugalroute_api/data/sample_run.json)",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="override the bundle's generated_at (default: the run's own timestamp)",
    )
    args = parser.parse_args(argv)

    bundle = build_bundle(args.run, generated_at=args.generated_at)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out}  ({len(bundle['reports'])} reports, run {bundle['run_id']})")
    for report in bundle["reports"]:
        print(
            f"  {report['strategy']:<11} retention@target={report['retention_at_target']}"
            f"  cost_reduction@target={report['cost_reduction_at_target']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
