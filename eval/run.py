"""Offline eval harness entry point (build-spec §15, split 05).

This is a thin shim over the engine's ``eval`` CLI so there is **one source of
truth** for the sweep logic (``frugalroute.harness``). Run it from the repo root
after ``pip install -e core``::

    python eval/run.py --quick --strategy both --benchmark gsm8k
    python eval/run.py --strategy cascade --benchmark gsm8k --repeats 3

It prints the frontier table + leaderboard + the honest headline and writes a
persisted run to ``eval/runs/<benchmark>-<ts>.jsonl`` (gitignored). Equivalent to
``python -m frugalroute.cli eval ...``.
"""

from __future__ import annotations

import sys

from frugalroute.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
