"""Classifier train/predict + θ frontier + save/load — tests 6-9 plus edges.

Synthetic embeddings only: two well-separated clusters labeled {cheap, strong},
so nothing here downloads a model or needs a key. A small ``_StubClf`` pins
``p_strong`` exactly where a test needs it (the θ sweep, R11 cases elsewhere).
"""

from __future__ import annotations

import numpy as np
import pytest

from frugalroute.classifier import (
    PredictiveRouter,
    load_router,
    save_router,
    train,
)

HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-4-8"
TIERS = [HAIKU, OPUS]
DIM = 4


class _StubClf:
    """A fixed-probability classifier: ``predict_proba`` ignores the input."""

    def __init__(self, classes: list[str], proba_row: list[float]) -> None:
        self.classes_ = np.array(classes)
        self._row = np.asarray(proba_row, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(self._row, (len(features), 1))


def _clusters(seed: int = 0, spread: float = 0.05, n: int = 25):
    """Two tight, well-separated clusters in DIM-space, labeled cheap/strong."""
    rng = np.random.RandomState(seed)
    cheap = rng.randn(n, DIM) * spread + np.array([-3.0, 0.0, 0.0, 0.0])
    strong = rng.randn(n, DIM) * spread + np.array([3.0, 0.0, 0.0, 0.0])
    embeddings = np.vstack([cheap, strong]).astype(np.float32)
    labels = [HAIKU] * n + [OPUS] * n
    return embeddings, labels


def _router(clf, tiers=TIERS) -> PredictiveRouter:
    return PredictiveRouter(
        clf=clf,
        tiers=list(tiers),
        embedder_name="fake-embedder",
        prompt_version="v1",
        label_run_ids=["labels-gsm8k-abc123"],
    )


def test_train_and_predict_separates_clusters() -> None:
    # 6. Logreg learns the split; a point near each cluster routes to that tier.
    embeddings, labels = _clusters()
    clf = train(embeddings, labels, TIERS)
    router = _router(clf)

    strong_tier, p_near_strong = router.predict_tier_from_embedding([3.0, 0.0, 0.0, 0.0])
    cheap_tier, p_near_cheap = router.predict_tier_from_embedding([-3.0, 0.0, 0.0, 0.0])

    assert strong_tier == OPUS and p_near_strong > 0.5
    assert cheap_tier == HAIKU and p_near_cheap < 0.5


def test_training_is_deterministic_for_fixed_random_state() -> None:
    # 6 (determinism). Same data + random_state → identical fitted predictions.
    embeddings, labels = _clusters()
    clf_a = train(embeddings, labels, TIERS, random_state=0)
    clf_b = train(embeddings, labels, TIERS, random_state=0)

    proba_a = np.asarray(clf_a.predict_proba(embeddings))
    proba_b = np.asarray(clf_b.predict_proba(embeddings))
    assert np.array_equal(proba_a, proba_b)


@pytest.mark.parametrize(
    ("theta", "expected"),
    [(0.6, OPUS), (0.69, OPUS), (0.7, HAIKU), (0.8, HAIKU)],
)
def test_theta_threshold_flips_routed_tier(theta: float, expected: str) -> None:
    # 7. With p_strong fixed at 0.70, route to strong iff p_strong > theta.
    router = _router(_StubClf(TIERS, [0.30, 0.70]))
    tier, p_strong = router.predict_tier_from_embedding([0.0, 0.0, 0.0, 0.0], theta=theta)
    assert p_strong == pytest.approx(0.70)
    assert tier == expected


def test_default_theta_is_half() -> None:
    # theta=None defaults to 0.5: p_strong 0.51 -> strong, 0.49 -> cheap.
    assert (
        _router(_StubClf(TIERS, [0.49, 0.51])).predict_tier_from_embedding([0.0] * DIM)[0] == OPUS
    )
    assert (
        _router(_StubClf(TIERS, [0.51, 0.49])).predict_tier_from_embedding([0.0] * DIM)[0] == HAIKU
    )


def test_save_load_round_trip(tmp_path) -> None:
    # 8. save_router -> load_router preserves predictions + all provenance.
    embeddings, labels = _clusters()
    clf = train(embeddings, labels, TIERS)
    router = _router(clf)
    path = tmp_path / "gsm8k.joblib"

    save_router(router, path)
    loaded = load_router(path)

    assert loaded.tiers == router.tiers
    assert loaded.embedder_name == router.embedder_name
    assert loaded.prompt_version == router.prompt_version
    assert loaded.label_run_ids == router.label_run_ids
    # Predictions survive the round-trip exactly.
    point = [3.0, 0.0, 0.0, 0.0]
    assert loaded.predict_tier_from_embedding(point) == router.predict_tier_from_embedding(point)


def test_save_creates_parent_dirs(tmp_path) -> None:
    embeddings, labels = _clusters()
    router = _router(train(embeddings, labels, TIERS))
    nested = tmp_path / "models" / "deep" / "gsm8k.joblib"
    save_router(router, nested)
    assert nested.exists()


def test_knn_fallback_trains_and_predicts() -> None:
    # 9. kNN fallback fits and returns a valid tier.
    embeddings, labels = _clusters()
    clf = train(embeddings, labels, TIERS, kind="knn")
    router = _router(clf)
    tier, _ = router.predict_tier_from_embedding([3.0, 0.0, 0.0, 0.0])
    assert tier in TIERS


def test_clf_kind_records_estimator() -> None:
    embeddings, labels = _clusters()
    assert _router(train(embeddings, labels, TIERS)).clf_kind == "LogisticRegression"
    assert _router(train(embeddings, labels, TIERS, kind="knn")).clf_kind == "KNeighborsClassifier"


def test_single_class_uses_constant_classifier() -> None:
    # Robustness: if every item labels to one tier, training still succeeds and
    # the predictor is well-defined (a constant), not a crash.
    embeddings = np.random.RandomState(1).randn(10, DIM).astype(np.float32)
    labels = [HAIKU] * 10
    clf = train(embeddings, labels, TIERS)
    router = _router(clf)
    assert router.clf_kind == "DummyClassifier"
    tier, p_strong = router.predict_tier_from_embedding([1.0, 2.0, 3.0, 4.0])
    # Only the cheap tier was ever the answer → strong never predicted.
    assert tier == HAIKU and p_strong == 0.0


def test_train_rejects_unknown_kind() -> None:
    embeddings, labels = _clusters()
    with pytest.raises(ValueError, match="Unknown classifier kind"):
        train(embeddings, labels, TIERS, kind="forest")


def test_train_rejects_empty_or_mismatched() -> None:
    with pytest.raises(ValueError, match="non-empty 2-D"):
        train(np.empty((0, DIM), dtype=np.float32), [], TIERS)
    with pytest.raises(ValueError, match="must match"):
        train(np.zeros((3, DIM), dtype=np.float32), [HAIKU, OPUS], TIERS)


def test_load_router_rejects_non_router(tmp_path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "a router"}, path)
    with pytest.raises(TypeError, match="does not contain a PredictiveRouter"):
        load_router(path)


def test_load_router_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_router(tmp_path / "nope.joblib")
