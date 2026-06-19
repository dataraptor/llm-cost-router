"""CLI ``eval`` subcommand — no-key integration over the cascade frontier.

A ``fake_client`` (one canned answer + gate verdict for every call) is injected via
the ``main(..., client=...)`` seam, so the whole eval pipeline — collect → assemble
→ render → persist — runs end-to-end with no key or network. Exercises the frontier
table, leaderboard, honest headline, the ``--grid`` override, and the ``--batch``
guard.
"""

from __future__ import annotations

import pytest

from frugalroute.cli import main
from frugalroute.harness import read_run
from frugalroute.models import GateVerdict


def _eval_client(fake_client, fake_usage):
    return fake_client(
        text="The answer is 72.",
        parsed_output=GateVerdict(sufficient=True, confidence=0.9, reason="committed"),
        usage=fake_usage(input_tokens=120, output_tokens=40),
    )


def test_cli_eval_cascade_quick(fake_client, fake_usage, capsys, tmp_path) -> None:
    client = _eval_client(fake_client, fake_usage)
    out_path = tmp_path / "run.jsonl"
    code = main(
        [
            "eval",
            "--quick",
            "--strategy",
            "cascade",
            "--benchmark",
            "gsm8k",
            "--out",
            str(out_path),
        ],
        client=client,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Frontier (cascade):" in out
    assert "Leaderboard:" in out
    assert "oracle (ceiling)" in out
    assert "Frozen test split:" in out
    assert "FrugalRoute (cascade) retains" in out
    assert "frozen split" in out
    assert "free quality" not in out  # honesty: never claim free quality

    # The run persisted and round-trips.
    assert out_path.exists()
    restored = read_run(out_path)
    assert "cascade" in restored["reports"]
    assert restored["meta"]["benchmark"] == "gsm8k"


def test_cli_eval_grid_override(fake_client, fake_usage, capsys, tmp_path) -> None:
    client = _eval_client(fake_client, fake_usage)
    out_path = tmp_path / "run.jsonl"
    code = main(
        [
            "eval",
            "--strategy",
            "cascade",
            "--benchmark",
            "gsm8k",
            "--repeats",
            "1",
            "--grid",
            "0.5,0.9",
            "--out",
            str(out_path),
        ],
        client=client,
    )
    out = capsys.readouterr().out
    assert code == 0
    # Only the two supplied operating points appear in the frontier table.
    restored = read_run(out_path)
    params = sorted(p.operating_param for p in restored["reports"]["cascade"].points)
    assert params == [0.5, 0.9]
    assert "0.50" in out and "0.90" in out


def test_cli_eval_batch_rejected_cleanly(capsys) -> None:
    # --batch needs the native Anthropic Batches backend → clean message, exit 2,
    # no traceback (and it returns before any client/key is required).
    code = main(["eval", "--batch", "--strategy", "cascade", "--benchmark", "gsm8k"])
    err = capsys.readouterr().err
    assert code == 2
    assert "--batch" in err


def test_cli_eval_missing_key_surfaces_message(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    code = main(["eval", "--quick", "--strategy", "cascade", "--benchmark", "gsm8k"])
    err = capsys.readouterr().err
    assert code == 1
    assert "ANTHROPIC_API_KEY is not set" in err


def test_cli_eval_both_degrades_to_cascade_without_embedder(
    fake_client, fake_usage, capsys, tmp_path, monkeypatch
) -> None:
    # --strategy both, embedder unavailable → degrade to a cascade-only report with
    # a clear note (never crash). Force the embedder to be unavailable.
    import sys

    def _no_embedder() -> object:
        raise OSError("embedder unavailable (test)")

    # `frugalroute.embed` the name is the re-exported function, so patch the real
    # submodule object that `_ensure_embedder` imports `get_embedder` from.
    monkeypatch.setattr(sys.modules["frugalroute.embed"], "get_embedder", _no_embedder)

    client = _eval_client(fake_client, fake_usage)
    out_path = tmp_path / "run.jsonl"
    code = main(
        ["eval", "--quick", "--strategy", "both", "--benchmark", "gsm8k", "--out", str(out_path)],
        client=client,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "predictive skipped" in captured.err
    assert "Frontier (cascade):" in captured.out
    assert "FrugalRoute (predictive)" not in captured.out


def test_cli_eval_predictive_without_embedder_errors(
    fake_client, fake_usage, capsys, monkeypatch
) -> None:
    import sys

    def _no_embedder() -> object:
        raise OSError("embedder unavailable (test)")

    # `frugalroute.embed` the name is the re-exported function, so patch the real
    # submodule object that `_ensure_embedder` imports `get_embedder` from.
    monkeypatch.setattr(sys.modules["frugalroute.embed"], "get_embedder", _no_embedder)

    client = _eval_client(fake_client, fake_usage)
    code = main(
        ["eval", "--quick", "--strategy", "predictive", "--benchmark", "gsm8k"], client=client
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "predictive eval needs the local embedder" in err


def test_cli_eval_unknown_benchmark_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["eval", "--benchmark", "nope"])
