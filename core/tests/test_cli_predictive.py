"""CLI predictive ``route`` + ``train`` wiring — tests 14-15 plus error paths.

No-key: a real (tiny) router is trained on synthetic embeddings and saved to a
temp ``.joblib``; the CLI loads it and routes with an injected fake embedder +
fake client, so the whole ``train`` → ``save`` → ``route`` surface is exercised
without a key or a model download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from frugalroute.classifier import PredictiveRouter, save_router, train
from frugalroute.cli import main

HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-4-8"
TIERS = [HAIKU, OPUS]
DIM = 4


class _FakeEmbedder:
    def __init__(self, vector) -> None:
        self._vector = np.asarray(vector, dtype=np.float32)

    def encode(self, queries, **_kwargs):
        return np.tile(self._vector, (len(list(queries)), 1))


def _saved_router(tmp_path):
    """Fit a tiny real logreg on two clusters and save it; return the path."""
    rng = np.random.RandomState(0)
    cheap = rng.randn(20, DIM) * 0.05 + np.array([-3.0, 0.0, 0.0, 0.0])
    strong = rng.randn(20, DIM) * 0.05 + np.array([3.0, 0.0, 0.0, 0.0])
    embeddings = np.vstack([cheap, strong]).astype(np.float32)
    labels = [HAIKU] * 20 + [OPUS] * 20
    clf = train(embeddings, labels, TIERS)
    router = PredictiveRouter(
        clf=clf,
        tiers=list(TIERS),
        embedder_name="fake",
        prompt_version="v1",
        label_run_ids=["labels-gsm8k-abc123"],
    )
    path = tmp_path / "gsm8k.joblib"
    save_router(router, path)
    return path


def test_predictive_route_json(tmp_path, fake_client, fake_usage, capsys) -> None:
    # 14. route --strategy predictive --json emits the full RouteResult with the
    # predictive fields populated (p_strong set, gate null).
    path = _saved_router(tmp_path)
    client = fake_client(
        text="The answer is 72.", usage=fake_usage(input_tokens=150, output_tokens=250)
    )
    embedder = _FakeEmbedder([3.0, 0.0, 0.0, 0.0])  # near the strong cluster
    code = main(
        [
            "route",
            "--strategy",
            "predictive",
            "--example",
            "gsm8k-1142",
            "--model",
            str(path),
            "--json",
        ],
        client=client,
        embedder=embedder,
    )
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "predictive"
    assert payload["gate"] is None
    assert payload["p_strong"] is not None and 0.0 <= payload["p_strong"] <= 1.0
    assert payload["tier_used"] in TIERS
    assert payload["query"].startswith("Natalia sold clips")
    # Exactly one model call was made.
    assert [m for m, _ in client.messages.calls] == ["create"]


def test_predictive_route_human_output(tmp_path, fake_client, fake_usage, capsys) -> None:
    # Human output uses the predictive decision line + single-tier breakdown.
    path = _saved_router(tmp_path)
    client = fake_client(usage=fake_usage(input_tokens=150, output_tokens=250))
    embedder = _FakeEmbedder([3.0, 0.0, 0.0, 0.0])
    code = main(
        [
            "route",
            "--strategy",
            "predictive",
            "--example",
            "gsm8k-1142",
            "--model",
            str(path),
            "--theta",
            "0.5",
        ],
        client=client,
        embedder=embedder,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Strategy:  predictive" in out
    assert "Decision:  p_strong=" in out
    assert "vs theta=0.50" in out
    # No gate line for the predictive path.
    assert "Gate:" not in out
    assert "Breakdown: = Opus" in out


def test_predictive_requires_model(scripted_client, capsys) -> None:
    code = main(
        ["route", "--strategy", "predictive", "--example", "gsm8k-1142"], client=scripted_client([])
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "requires --model" in err


def test_predictive_missing_model_file(tmp_path, scripted_client, capsys) -> None:
    missing = tmp_path / "nope.joblib"
    code = main(
        ["route", "--strategy", "predictive", "--example", "gsm8k-1142", "--model", str(missing)],
        client=scripted_client([]),
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "not found" in err


def test_predictive_degenerate_embedding_errors(tmp_path, fake_client, capsys) -> None:
    # R11 through the CLI: a NaN embedding → clean ValueError message, exit 2, no
    # model call and no traceback.
    path = _saved_router(tmp_path)
    client = fake_client()
    nan_embedder = _FakeEmbedder([float("nan"), 0.0, 0.0, 0.0])
    code = main(
        ["route", "--strategy", "predictive", "--example", "gsm8k-1142", "--model", str(path)],
        client=client,
        embedder=nan_embedder,
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "Degenerate embedding" in err
    assert client.messages.calls == []


def test_train_help_wires() -> None:
    # 15a. `train --help` parses and exits 0.
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0


def test_route_help_wires() -> None:
    # 15b. `route --help` (covers the predictive flags) parses and exits 0.
    with pytest.raises(SystemExit) as exc:
        main(["route", "--help"])
    assert exc.value.code == 0


def test_train_missing_key_surfaces_clear_message(monkeypatch, capsys) -> None:
    # train with no client and no key → clear message, exit 1, no traceback.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(["train", "--benchmark", "gsm8k", "--n", "4"])  # client=None → get_client()
    err = capsys.readouterr().err
    assert code == 1
    assert "ANTHROPIC_API_KEY is not set" in err


def test_train_unknown_benchmark_errors(scripted_client, capsys) -> None:
    with pytest.raises(SystemExit):
        # argparse rejects the choice before our handler runs.
        main(["train", "--benchmark", "bogus"], client=scripted_client([]))


def test_train_end_to_end_with_fakes(
    tmp_path, scripted_client, fake_response, fake_usage, capsys
) -> None:
    # Full no-key train: scripted generate answers + a fake embedder → saved router.
    from frugalroute.benchmarks import frozen_split, load_benchmark
    from frugalroute.classifier import load_router

    items = load_benchmark("gsm8k", n=10)
    calibration, _ = frozen_split(items)
    usage = fake_usage(input_tokens=10, output_tokens=10)
    # Two tiers per calibration item; answer "The answer is <gold>." so all grade
    # correct → labels collapse to the cheapest tier (DummyClassifier fallback).
    responses = []
    for item in calibration:
        for _ in TIERS:
            responses.append(fake_response(text=f"The answer is {item.gold}.", usage=usage))
    client = scripted_client(responses)
    embedder = _FakeEmbedder([1.0, 0.0, 0.0, 0.0])
    out = tmp_path / "gsm8k.joblib"

    code = main(
        ["train", "--benchmark", "gsm8k", "--n", "10", "--out", str(out)],
        client=client,
        embedder=embedder,
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert out.exists()
    assert "Trained predictive router" in printed
    assert "Label distribution:" in printed
    assert "Saved router to" in printed

    router = load_router(out)
    assert router.tiers == TIERS
    assert router.prompt_version == "v1"
    assert router.label_run_ids and router.label_run_ids[0].startswith("labels-gsm8k-")
